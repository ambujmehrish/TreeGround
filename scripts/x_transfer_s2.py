import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_eval as S2E  # noqa: E402
import s2_train as S2T  # noqa: E402
import v4_tree_value as V4  # noqa: E402
import x_transfer as X  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data.paco import CHAIN_TEMPLATE  # noqa: E402
from treeground.identity import qwen_static  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.mllm import SEL  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, ensure_deviation_log, load_config,  # noqa: E402
                            read_json, require_approved, resolve_run, runs_root, start_run, write_json)
from treeground.selector import Selector  # noqa: E402

# Transfer evaluation of S2 (deviation D17): x_transfer with the LoRA-adapted Qwen, its trained <SEL> row and the S2
# selector from the S2 checkpoint, tau_rej from the S2 eval run of that checkpoint. Node features and chain probes
# are computed as s2_eval recomputes them on PACO: [image; chain query without <SEL>] + teacher-forced answer
# '{object} <SEL> {part} <SEL>', layer-l states, fp16-rounded like the recomputed cache.

CODE = ["scripts/x_transfer_s2.py", "scripts/s2_train.py", "scripts/s2_eval.py", "treeground/s2.py",
        "treeground/lora.py"]


def chain_query(qid: str, obj: str, part: str) -> S2T.Query:
    return S2T.Query(qid, "chain", S2.user_text(CHAIN_TEMPLATE.format(object=obj, part=part)),
                     f"{obj} {SEL} {part} {SEL}", [])


def eval_run(cfg: dict, stage: str, s2_run: Path) -> Path:
    # the S2 eval run that evaluated this S2 run's configured checkpoint (latest such run unless pinned)
    ref = cfg["inputs"].get(f"{stage}_eval", "latest")
    if ref != "latest":
        runs = [resolve_run(cfg, f"{stage}_eval", ref, False)]
    else:
        runs = sorted(p for p in (runs_root(cfg) / f"{stage}_eval").glob("*") if (p / "metrics.json").exists())
    for run in reversed(runs):
        m, meta = read_json(run / "metrics.json"), read_json(run / "meta.json")
        if (not m["smoke"] and m["s2_run"] == s2_run.name
                and meta.get("checkpoint") == cfg["s2_eval"]["checkpoint"]):
            return run
    raise ProtocolError(f"no completed {stage}_eval run of {s2_run.name}/{cfg['s2_eval']['checkpoint']}")


def s2_inputs(cfg: dict) -> tuple[str, Path, Path, float]:
    stage = S2T.s2_stage(cfg)
    run = resolve_run(cfg, stage, cfg["inputs"].get(stage, "latest"), False)
    ev = eval_run(cfg, stage, run)
    return stage, run, ev, float(read_json(ev / "metrics.json")["chains"]["tau_rej_chain"])


def identity(cfg: dict, dataset: str, stage: str, run: Path, ev: Path) -> dict:
    h = hashlib.sha256()
    for rel in CODE:
        h.update((ROOT / rel).read_bytes())
    ident = X.identity(cfg, dataset, run)
    ident.update(kind="transfer_s2", s2_stage=stage, s2_eval_run=ev.name, checkpoint=cfg["s2_eval"]["checkpoint"],
                 code=ident["code"] + [h.hexdigest()])
    return ident


class ImageSource:
    # S2T.recompute's source interface over one in-memory transfer image and its capped node masks
    def __init__(self, processor, sel_id: int, image: np.ndarray, rles):
        self.processor, self.sel_id, self.rles = processor, sel_id, rles
        self.image = Image.fromarray(image)
        self.merge = processor.image_processor.merge_size
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ProtocolError("tokenizer has no pad token")

    def image_item(self, m, queries: list[S2T.Query]):
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, self.image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        return seqs, grid, mllm.sparse_coverages((M.decode(r) for r in self.rles), *grid)


@torch.no_grad()
def chain_scores(model, src: ImageSource, sel: Selector, queries: list[S2T.Query], layer: int, device: str,
                 batch_size: int) -> list[np.ndarray]:
    z, probes = S2T.recompute(model, src, None, queries, layer, device, batch_size)
    e = sel.nodes(z.half().to(device))
    return [torch.sigmoid(sel.scores(sel.queries(p.half()), e)).cpu().numpy() for p in probes]


@torch.no_grad()
def process_image(models, cfg, key, image, hw, chains, parts, cap):
    sam, sproc, model, processor, sel, sel_id, layer, device = models
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"], p["stability_offset"],
                            p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    if image.shape[:2] != hw:
        raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {hw}")
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    boxes, stab, src = out["boxes"][kept], out["stability"][kept], out["src_index"][kept]
    cov = []
    if parts:
        pi = X.gt_ious([r for _, r in parts], rles, boxes)
        best = pi.max(1) if pi.shape[1] else np.zeros(len(parts))
        cov += [{"image": key, "kind": "part", "label": lab, "area_bin": area_bin(X.gt_area(r)),
                 "best_iou": float(b)} for (lab, r), b in zip(parts, best)]
    full_tree = T.build_tree(rles, device) if rles else None
    c_idx = X.cap_nodes(rles, boxes, stab, src, cap)
    rc = [rles[i] for i in c_idx]
    bc = boxes[c_idx]
    if not chains:
        return cov, [], len(rles), len(rc)
    tree = T.build_tree(rc, device) if rc else T.from_parents(np.array([-1]))
    rand = [T.randomised_tree(tree, V4.tree_seed(s, key)) for s in cfg["v4"]["seeds"]]
    box_tree = T.build_box_tree(bc) if len(rc) else tree
    if rc:
        qs = [chain_query(qid, obj, part) for qid, obj, part, _, _ in chains]
        sigs = chain_scores(model, ImageSource(processor, sel_id, image, rc), sel, qs, layer, device,
                            cfg["transfer"]["probe_batch"])
    else:
        sigs = [np.zeros((2, 0)) for _ in chains]
    items = []
    for (qid, obj, part, orle, prle), sig in zip(chains, sigs):
        ious = X.gt_ious([orle, prle], rc, bc)
        full = X.gt_ious([orle, prle], rles, boxes)
        ga = np.array([X.gt_area(orle), X.gt_area(prle)])
        cov.append({"image": key, "kind": "chain", "label": qid.split("|")[-1], "area_bin": area_bin(ga[1]),
                    "best_iou": float(full[1].max()) if full.shape[1] else 0.0,
                    "object_iou": float(full[0].max()) if full.shape[1] else 0.0,
                    "feasible_uncapped": bool(full.shape[1] and T.max_iou_path(full_tree, full, X.THR) is not None),
                    "part_ratio": float(ga[1] / max(1, ga[0]))})
        items.append({"qid": qid, "image": key, "sha1": key, "tree": tree, "rand": rand, "box": box_tree,
                      "ious": ious, "sig": sig, "part_ratio": float(ga[1] / max(1, ga[0]))})
    return cov, items, len(rles), len(rc)


def load_models(cfg: dict, ckpt: dict):
    device = cfg["device"]
    meta = ckpt["meta"]
    q = cfg["models"]["qwen"]
    if (meta["qwen"]["repo"], meta["qwen"]["revision"]) != (q["repo"], q["revision"]):
        raise ProtocolError("S2 checkpoint was trained on another Qwen revision")
    model, processor, extra = S2T.build_model(cfg, {"lora": meta["lora"]}, device, qwen_static(cfg))
    if extra["sel_id"] != meta["sel_id"]:
        raise ProtocolError("<SEL> token id differs from the checkpoint")
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    LR.load_trainable(model, extra["sel_emb"], sel, ckpt)
    model.eval()
    sel = sel.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p, m = cfg["proposals"], cfg["models"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    return sam, sproc, model, processor, sel, extra["sel_id"], int(meta["layer"]), device


def run_shard(cfg, dataset, shard, num_shards):
    require_approved(cfg, ["D3", "D4", "D7", "D10", "D17"]
                     + (["D8"] if dataset in ("lvis_hier", "openimages_hier") else [])
                     + (["D9"] if dataset == "openimages_hier" else []))
    stage, s2_run, ev, tau = s2_inputs(cfg)
    ident = identity(cfg, dataset, stage, s2_run, ev)
    d = X.out_dir(cfg, dataset, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pkl"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    ckpt, ck_run = S2E.load_checkpoint(cfg, False)
    if ck_run != s2_run or read_json(ev / "metrics.json")["checkpoint_step"] != ckpt["meta"]["step"]:
        raise ProtocolError(f"checkpoint {ck_run.name} step {ckpt['meta']['step']} is not the one {ev.name} evaluated")
    run = start_run(f"x_transfer_{dataset}_{stage}", cfg, False, {
        "shard": shard, "num_shards": num_shards, "cache": str(d), "s2_run": str(s2_run), "s2_eval_run": str(ev),
        "checkpoint_step": ckpt["meta"]["step"]})
    layer = int(ckpt["meta"]["layer"])
    models = load_models(cfg, ckpt)
    del ckpt
    cap = X.node_cap()
    cov_all, nat = [], {"strict": [], "d5": [], "d6": [], "no_rejection": []}
    t0, n_img = time.time(), 0
    for key, image, hw, chains, parts in X.records(cfg, dataset, shard, num_shards):
        t1 = time.time()
        cov, items, n_nodes, n_cap = process_image(models, cfg, key, image, hw, chains, parts, cap)
        cov_all += cov
        if items:
            nat["strict"].append(V4.natural(items, tau)[0])
            nat["d5"].append(V4.natural(items, tau, inclusive=True)[0])
            nat["d6"].append(V4.natural(items, tau, box=True)[0])
            nat["no_rejection"].append(V4.natural(items, -np.inf)[0])
        n_img += 1
        print(f"[{n_img}] {key} nodes {n_nodes} capped {n_cap} chains {len(chains)} {time.time() - t1:.1f}s",
              flush=True)
    res = {"coverage": pd.DataFrame(cov_all),
           **{k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in nat.items()},
           "tau": tau, "layer": layer, "images": n_img}
    tmp = target.with_suffix(".tmp")
    pd.to_pickle(res, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": n_img, "seconds": time.time() - t0})


def merge(cfg, dataset, num_shards):
    stage, s2_run, ev, _ = s2_inputs(cfg)
    ident = identity(cfg, dataset, stage, s2_run, ev)
    # X.merge resolves its selector run and cache identity through these module functions and names the stage
    # x_transfer_<ds>_<selector_stage>_merge; pointing them at the S2 run gives x_transfer_<ds>_<s2 stage>_merge
    runs = []
    X.selector_run = lambda _: s2_run
    X.identity = lambda *_: ident
    X.start_run = lambda *a, **k: runs.append(start_run(*a, **k)) or runs[-1]
    X.merge({**cfg, "inputs": {**cfg["inputs"], "selector_stage": stage}}, dataset, num_shards)
    out = read_json(runs[0] / "metrics.json")
    out.update(deviation=["D7", "D17"], selector_stage=stage, s2_run=s2_run.name, s2_eval_run=ev.name,
               checkpoint=cfg["s2_eval"]["checkpoint"])
    write_json(runs[0] / "metrics.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=X.DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    ensure_deviation_log(cfg)
    num_shards = cfg["transfer"]["num_shards"]
    if args.merge:
        require_approved(cfg, ["D17"])
        merge(cfg, args.dataset, num_shards)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.dataset, shard=args.shard, num_shards=num_shards)


if __name__ == "__main__":
    main()
