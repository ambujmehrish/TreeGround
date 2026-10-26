import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_train  # noqa: E402,F401  (installs the D30 hooks on s2_train)
import s2_train as S2T  # noqa: E402
import s2_eval as S2E  # noqa: E402
import s3_pool as S3P  # noqa: E402
import x_transfer_s2 as X2  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.calib import calibrate_rejection  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            require_approved, start_run, write_json)
from treeground.selector import Selector  # noqa: E402

# Deviation D30 evaluation on PACO-LVIS val (+ gRefCOCO val no-target): every query is answered over its own
# augmented pool = the image's capped D3/D4 nodes + the top-k SAM 3 text instances of the query's own phrases
# (s3_pool.phrases; the pool of one query never contains another query's instances). Node features come from one
# LoRA pass per image over the union of the image's pools (selector scores are per candidate, so this equals a
# separate pass per pool). Single step: top-1 over the pool, IoU with the GT (empty = 0); chains: object = top-1
# of step 1, part flat = top-1 of step 2 over the pool, part tree = top-1 of step 2 among the strict descendants of
# the object pick in the containment tree of that pool (the object pick itself when it has none). Reported without
# rejection and at tau_rej calibrated (V2 objective, calib_lambda) on the other half of the images (2-fold).

KINDS = ("object", "part", "chain", "absent")


def num_shards(cfg: dict) -> int:
    return int(cfg["s3_eval"]["num_shards"])


def load_model(cfg: dict, static: dict, device: str):
    ckpt, run = S2E.load_checkpoint(cfg, False)
    meta = ckpt["meta"]
    model, processor, extra = S2T.build_model(cfg, {"lora": meta["lora"]}, device, static)
    if extra["sel_id"] != meta["sel_id"]:
        raise ProtocolError("<SEL> token id differs from the checkpoint")
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    LR.load_trainable(model, extra["sel_emb"], sel, ckpt)
    return model.eval(), processor, extra, sel.to(device).eval(), meta, run


def pick(sig: np.ndarray, idx: np.ndarray) -> int:
    return int(idx[int(np.argmax(sig[idx]))]) if len(idx) else -1


def image_rows(model, processor, extra, sel, layer, device, text, cfg, g, image, nodes, gts) -> list[dict]:
    k = cfg["s3_pool"]["instances_per_phrase"]
    plist = list(dict.fromkeys(p for t in g.text for p in S3P.phrases(t)))
    inst = text(image, plist, k)
    pool, owner = list(nodes), {}
    for p, lst in zip(plist, inst):
        owner[p] = []
        for m, _ in lst:
            if m.any():
                owner[p].append(len(pool))
                pool.append(M.encode(m)[0])
    n = len(nodes)
    qs = [S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [])
          for q in g.itertuples()]
    if not pool:
        sigs = [np.zeros((q.n_sel, 0)) for q in g.itertuples()]
    else:
        z, probes = S2T.recompute(model, X2.ImageSource(processor, extra["sel_id"], image, pool), None, qs, layer,
                                  device, cfg["s2_eval"]["batch_size"])
        e = sel.nodes(z.half().to(device))
        sigs = [torch.sigmoid(sel.scores(sel.queries(p.half()), e)).float().cpu().numpy() for p in probes]
    rows = []
    for q, sig in zip(g.itertuples(), sigs):
        idx = np.asarray(list(range(n)) + [i for p in S3P.phrases(q.text) for i in owner.get(p, [])], np.int64)
        base = {"qid": q.qid, "kind": q.kind, "image": q.image_key, "present": q.kind != "absent",
                "pool": int(len(idx)), "nodes": n}
        if q.kind == "absent":
            rows.append({**base, "max_sigma": float(sig[0, idx].max()) if len(idx) else float("-inf")})
            continue
        gt = [gts[a] for a in json.loads(q.gt)]
        ious = M.iou([pool[i] for i in idx], gt) if len(idx) else np.zeros((0, len(gt)))
        loc = {int(i): j for j, i in enumerate(idx)}
        if q.kind == "chain":
            a1, af = pick(sig[0], idx), pick(sig[1], idx)
            sub = [pool[i] for i in idx]
            tree = T.build_tree(sub, device)
            desc = tree.strict_descendants(loc[a1]) if a1 >= 0 else np.zeros(0, np.int64)
            a2 = pick(sig[1], idx[desc]) if len(desc) else a1
            rows.append({**base, "object_iou": float(ious[loc[a1], 0]) if a1 >= 0 else 0.0,
                         "part_flat_iou": float(ious[loc[af], 1]) if af >= 0 else 0.0,
                         "part_tree_iou": float(ious[loc[a2], 1]) if a2 >= 0 else 0.0,
                         "max_sigma": float(sig[1, idx].max()) if len(idx) else float("-inf"),
                         "part_flat_kind": "node" if 0 <= af < n else "inst",
                         "best_part_iou": float(ious[:, 1].max()) if len(idx) else 0.0})
            continue
        a = pick(sig[0], idx)
        inter = M.intersections([pool[a]], gt)[0, 0] if a >= 0 else 0
        area_p = M.areas([pool[a]])[0] if a >= 0 else 0
        area_g = M.areas(gt)[0]
        rows.append({**base, "sel_iou": float(ious[loc[a], 0]) if a >= 0 else 0.0,
                     "max_sigma": float(sig[0, a]) if a >= 0 else float("-inf"),
                     "pick_kind": "node" if 0 <= a < n else "inst",
                     "best_node_iou": float(ious[:n, 0].max()) if n else 0.0,
                     "best_pool_iou": float(ious[:, 0].max()) if len(idx) else 0.0,
                     "sel_inter": int(inter), "sel_union": int(area_p + area_g - inter), "gt_area": int(area_g)})
    return rows


def cache_dir(cfg: dict, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "s3_eval" / S2T.s2_stage(cfg) / canonical_hash(ident)[:16]


def identity(cfg: dict, ck_run: Path, step: int) -> dict:
    import hashlib
    h = hashlib.sha256()
    for rel in ("scripts/s3_eval.py", "scripts/s3_pool.py", "scripts/s2_train.py", "scripts/b_baselines.py"):
        h.update((ROOT / rel).read_bytes())
    return {"kind": "s3_eval", "run": ck_run.name, "checkpoint": cfg["s2_eval"]["checkpoint"], "step": step,
            "pool": {k: cfg["s3_pool"][k] for k in ("instances_per_phrase",)}, "sam3_text": cfg["baselines"]["sam3_text"],
            "num_shards": num_shards(cfg), "code": h.hexdigest()}


def run_shard(cfg: dict, shard: int) -> None:
    require_approved(cfg, S2T.deviations(cfg))
    device = cfg["device"]
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    static = qwen_static(cfg)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    _, queries = open_queries(cfg, False, mhash, static)
    model, processor, extra, sel, meta, ck_run = load_model(cfg, static, device)
    ident = identity(cfg, ck_run, meta["step"])
    d = cache_dir(cfg, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"{S2T.s2_stage(cfg)}_s3eval_shard", cfg, False, {"shard": shard, "cache": str(d)})
    text = S3P.TextInstances(cfg)
    files = dict(zip(wl.sha1, wl.path))
    val = queries[(queries.split == "val") & queries.kind.isin(KINDS)]
    groups = sorted(val.sha1.unique())[shard::num_shards(cfg)]
    pids = {int(k.split(":")[1]) for k in val[val.sha1.isin(groups) & (val.kind != "absent")].image_key}
    pimgs = paco.load(cfg["data"]["paco_lvis"], "val", pids)
    rows, t0, tau = [], time.time(), cfg["proposals"]["node_dedup"]
    for i, sha1 in enumerate(groups, 1):
        g = val[val.sha1 == sha1].sort_values("offset")
        image = load_rgb(assert_real_path(files[sha1]))
        nd = props.nodes(sha1, tau)
        if (nd["h"], nd["w"]) != image.shape[:2]:
            raise ProtocolError(f"{sha1}: proposal masks differ from the image size")
        present = g[g.kind != "absent"]
        gts = paco.gt_rles(pimgs[int(present.image_key.iloc[0].split(":")[1])]) if len(present) else {}
        rows += image_rows(model, processor, extra, sel, meta["layer"], device, text, cfg, g, image, nd["rles"], gts)
        if i % 25 == 0:
            print(f"[{i}/{len(groups)}] queries {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(groups), "queries": len(rows), "seconds": time.time() - t0})


def summarize(cfg: dict, df: pd.DataFrame) -> dict:
    bs = cfg["v2"]["bootstrap"]
    lam = cfg["v2"]["calib_lambda"]

    def ci(v, frame):
        return mean_ci(np.asarray(v, float), frame.image.to_numpy(), bs["n"], bs["seed"])

    pres = df[df.kind.isin(["object", "part"])].reset_index(drop=True)
    absn = df[df.kind == "absent"].reset_index(drop=True)
    chains = df[df.kind == "chain"].reset_index(drop=True)
    images = np.array(sorted(df.image.unique()))
    fold = dict(zip(images, np.random.default_rng(0).permutation(len(images)) % 2))
    accept = np.zeros(len(pres), bool)
    abst = np.zeros(len(absn), bool)
    taus = []
    for f in (0, 1):
        tp, ta = pres.image.map(fold) != f, absn.image.map(fold) != f
        tau, _ = calibrate_rejection(pres.max_sigma[tp], pres.sel_iou[tp], absn.max_sigma[ta], lam)
        taus.append(tau)
        ep, ea = ~tp, ~ta
        accept[ep.to_numpy()] = pres.max_sigma[ep] >= tau
        abst[ea.to_numpy()] = absn.max_sigma[ea] < tau
    sel = pres.sel_iou.to_numpy()
    keep = pres.sel_union.to_numpy() > 0

    def block(iou, acc):
        k = acc & keep
        union = np.where(k, pres.sel_union, pres.gt_area)
        return {"gIoU": ci(iou, pres), "success@0.5": ci(iou >= 0.5, pres),
                "cIoU": float(np.where(k, pres.sel_inter, 0).sum() / max(1, union.sum())),
                "by_kind": {kd: ci(iou[(pres.kind == kd).to_numpy()], pres[pres.kind == kd])
                            for kd in ("object", "part")}}

    out = {"present_queries": int(len(pres)), "no_target_queries": int(len(absn)), "chains": int(len(chains)),
           "no_rejection": block(sel, np.ones(len(pres), bool)),
           "held_out_tau": {**block(np.where(accept, sel, 0.0), accept), "taus": taus,
                            "accept_rate": float(accept.mean()), "no_target_accuracy": float(abst.mean())},
           "pick_from_instances": float((pres.pick_kind == "inst").mean()),
           "oracle": {"best_node": float(pres.best_node_iou.mean()), "best_pool": float(pres.best_pool_iou.mean())}}
    if len(chains):
        obj = chains.qid.map(lambda t: t.rsplit("|", 1)[-1].split(":")[0])
        objects = chains.assign(obj=obj).drop_duplicates(["image", "obj"])
        out["chain"] = {"object_gIoU": ci(objects.object_iou, objects),
                        "part_gIoU_flat": ci(chains.part_flat_iou, chains),
                        "part_gIoU_tree": ci(chains.part_tree_iou, chains),
                        "part_success_flat": ci(chains.part_flat_iou >= 0.5, chains),
                        "part_success_tree": ci(chains.part_tree_iou >= 0.5, chains),
                        "best_part_pool": float(chains.best_part_iou.mean())}
    return out


def merge(cfg: dict) -> None:
    require_approved(cfg, S2T.deviations(cfg))
    ck_run = S2E.load_checkpoint(cfg, False)[1]
    step = torch.load(ck_run / cfg["s2_eval"]["checkpoint"], map_location="cpu")["meta"]["step"]
    d = cache_dir(cfg, identity(cfg, ck_run, step))
    n = num_shards(cfg)
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    run = start_run(f"{S2T.s2_stage(cfg)}_s3eval", cfg, False, {"cache": str(d)})
    df.to_csv(run / "eval_queries.csv", index=False)
    write_json(run / "metrics.json", {"label": "dev", "deviation": "D30", "split": "PACO-LVIS val + gRefCOCO val "
                                      "no-target", "run": ck_run.name, "checkpoint_step": step,
                                      **summarize(cfg, df)})
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg)
    else:
        if not 0 <= args.shard < num_shards(cfg):
            raise ProtocolError(f"--shard must be in 0..{num_shards(cfg) - 1}")
        with torch.no_grad():
            run_shard(cfg, args.shard)


if __name__ == "__main__":
    main()
