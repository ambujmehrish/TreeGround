import argparse
import hashlib
import json
import re
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_baselines as B  # noqa: E402
import e_edit as EE  # noqa: E402
import x_transfer as X  # noqa: E402
import x_transfer_s2 as X2  # noqa: E402
from x_transfer_s2 import V4  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import rerank as RR  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, ensure_deviation_log, load_config, read_json,  # noqa: E402
                            require_approved, resolve_run, start_run, write_json)

# Edit-mask selection with the S2 model over an augmented pool (deviation D23). Per edit sample: the D3/D4 capped
# SAM 3 nodes plus the top-N SAM 3 text-prompt instances for the referent form one candidate set (one tree over
# the union); node features and the referent probe come from the LoRA model as in x_transfer_s2 (query 'Locate
# {referent}.' without <SEL>, teacher-forced single-step answer); the pick is the top-1 sigma, no rejection.
# Shards follow e_edit's partition, so e_edit's ours2_* conditions read shard i of this cache. --merge reports the
# localisation of every pick against the GT edit mask.

THR = 0.5
CHAIN_RE = re.compile(r"^the (.+?) of the (.+)$")  # 'the {part} of the {object}' (PartEdit, PACO edit set)
CODE = ["scripts/e_select.py", "scripts/x_transfer_s2.py", "scripts/s2_train.py", "treeground/s2.py",
        "treeground/lora.py", "treeground/data/edit_bench.py", "treeground/data/paco.py"]


def settings(cfg: dict) -> dict:
    return cfg["edit_select"]


def rerank_weight(cfg: dict, stage: str) -> tuple[int, float, str] | None:
    # rule logit + w log(sam_iou) on the stage's top-k capped nodes; w and k from the stage's D20 run (PACO val CV)
    if settings(cfg)["rerank"] != "rule":
        return None
    run = resolve_run(cfg, f"rerank_{stage}", cfg["inputs"].get(f"rerank_{stage}", "latest"), False)
    m = read_json(run / "metrics.json")
    k = int(m["primary_k"])
    return k, float(np.median(m["detail"][f"k{k}"]["rule_weight_per_fold"])), run.name


def identity(cfg: dict, dataset: str, ids: list[str], smoke: bool) -> dict:
    stage, s2_run, ev, _ = X2.s2_inputs(cfg)
    p, m = cfg["proposals"], cfg["models"]
    h = hashlib.sha256()
    for rel in CODE:
        h.update((ROOT / rel).read_bytes())
    rr = rerank_weight(cfg, stage)
    return {"kind": "edit_select_s2", "dataset": dataset, "smoke": smoke, "settings": cfg["edit"]["datasets"][dataset],
            "samples": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "sam3": [m["sam3"]["repo"], m["sam3"]["revision"]], "qwen": [m["qwen"]["repo"], m["qwen"]["revision"]],
            "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                            "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
            "node_cap": X.node_cap(), "s2_stage": stage, "s2_run": s2_run.name, "s2_eval_run": ev.name,
            "checkpoint": cfg["s2_eval"]["checkpoint"], "template": EE.REFERENT_TEMPLATE,
            "sam3_text": cfg["baselines"]["sam3_text"], "select": settings(cfg), "rerank": rr, "code": h.hexdigest()}


def cache_dir(cfg: dict, dataset: str, ident: dict, smoke: bool) -> Path:
    return EE.cache_dir(cfg, smoke) / "select_s2" / dataset / canonical_hash(ident)[:16]


@torch.no_grad()
def text_instances(st: B.Sam3Text, image: np.ndarray, phrase: str, n: int) -> list[tuple[np.ndarray, float]]:
    # B.Sam3Text's scoring and mask post-processing, keeping the n highest-scoring instances (first = sam3text)
    h, w = image.shape[:2]
    pix = st.proc(images=Image.fromarray(image), return_tensors="pt").pixel_values
    ve = st.model.get_vision_features(pixel_values=pix.to(st.device, st.dtype))
    t = st.proc(text=B.concept(phrase), return_tensors="pt").to(st.device)
    r = st.model(vision_embeds=ve, input_ids=t.input_ids, attention_mask=t.attention_mask)
    s = r.pred_logits[0].float().sigmoid()
    if r.presence_logits is not None:
        s = s * r.presence_logits[0].float().sigmoid()
    out = []
    for i in torch.argsort(s, descending=True, stable=True)[:n].tolist():
        m = F.interpolate(r.pred_masks[0, i][None, None].float().sigmoid(), size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0] > st.s["mask_threshold"]
        out.append((m.cpu().numpy(), float(s[i])))
    return out


def chain_parts(referent: str) -> tuple[str, str] | None:
    m = CHAIN_RE.match(referent)
    return (m[2], m[1]) if m else None


def referent_query(cfg: dict, referent: str) -> X2.S2T.Query:
    user = S2.user_text(EE.REFERENT_TEMPLATE.format(referent=referent))
    return X2.S2T.Query("edit", "object", user, settings(cfg)["answer"].format(referent=referent), [])


@torch.no_grad()
def node_logits(model, src: X2.ImageSource, sel, queries: list, layer: int, device: str) -> list[np.ndarray]:
    z, probes = X2.S2T.recompute(model, src, None, queries, layer, device, len(queries))
    e = sel.nodes(z.half().to(device))
    return [sel.scores(sel.queries(p.half()), e).float().cpu().numpy() for p in probes]


def rule_pick(logit: np.ndarray, sam_iou: np.ndarray, k: int, w: float) -> int:
    idx = RR.top_k(logit, k)
    df = pd.DataFrame({"qid": 0, "rank": np.arange(len(idx)), "logit": logit[idx],
                       "log_sam_iou": np.log(np.clip(sam_iou[idx], RR.SAM_IOU_FLOOR, None))})
    return int(idx[RR.pick(df, RR.rule_score(df, w)).iloc[0]])


def candidates(models, cfg: dict, image: np.ndarray, cap: dict):
    sam, sproc = models[:2]
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"], p["stability_offset"],
                            p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    c = X.cap_nodes(rles, out["boxes"][kept], out["stability"][kept], out["src_index"][kept], cap)
    return [rles[i] for i in c], out["sam_iou"][kept][c], len(rles)


def picks(tree: T.Tree, flat: np.ndarray, chain: list[np.ndarray] | None) -> tuple[int, int, int]:
    # flat: top-1 over the pool. Tree (V4 natural search, no rejection): object over the pool, part among the strict
    # descendants of the object pick, the object pick itself when it has none; plain referents fall back to flat
    n = len(flat)
    a = V4.decide(flat, np.arange(n), -np.inf)
    if chain is None:
        return a, -1, a
    a1 = V4.decide(chain[0], np.arange(n), -np.inf)
    pool = tree.strict_descendants(a1)
    return a, a1, V4.decide(chain[1], pool if len(pool) else np.array([a1]), -np.inf)


def select_sample(models, st, cfg: dict, s, cap: dict, rr) -> dict:
    _, _, model, processor, sel, sel_id, layer, device = models
    k = settings(cfg)["sam3_instances"]
    image, gt = s.load()
    g = M.encode(gt)[0]
    nodes, sam_iou, n_all = candidates(models, cfg, image, cap)
    ch = chain_parts(s.referent)
    ref = text_instances(st, image, s.referent, k)
    top = ref[0][0] if ref else None
    obj = text_instances(st, image, f"the {ch[0]}", k) if ch else []
    inst = [(m, t, r) for t, lst in (("ref", ref), ("obj", obj)) for r, (m, _) in enumerate(lst) if m.any()]
    pool = nodes + [M.encode(m)[0] for m, _, _ in inst]
    kind = ["node"] * len(nodes) + [f"{t}{r}" for _, t, r in inst]
    n = len(nodes)
    ious = M.iou(pool, [g])[:, 0] if pool else np.zeros(0)
    row = {"sample_id": s.sample_id, "image": s.image_key, "subset": s.subset, "chain": ch is not None,
           "nodes": n_all, "nodes_capped": n, "instances": len(inst), "gt_area": int(gt.sum()),
           "instance_scores": json.dumps({"ref": [sc for _, sc in ref], "obj": [sc for _, sc in obj]}),
           "sam3_iou": float(M.iou(M.encode(top), [g])[0, 0]) if top is not None and top.any() else 0.0,
           "best_node_iou": float(ious[:n].max()) if n else 0.0, "best_pool_iou": float(ious.max()) if pool else 0.0}
    if not pool:
        return {**row, **{f"{p}_{c}": v for p in ("ours2", "ours2tree") for c, v in
                          (("rle", json.dumps(None)), ("iou", 0.0), ("sigma", float("-inf")), ("kind", "none"))},
                "tree_object_kind": "none", "tree_step2_candidates": 0, "flat_outside_object": False,
                "flat_in_object_frac": float("nan"), "nodes_only_iou": 0.0, "nodes_rule_iou": float("nan")}
    tree = T.build_tree(pool, device)
    qs = [referent_query(cfg, s.referent)] + ([X2.chain_query("edit|chain", ch[0], ch[1])] if ch else [])
    lg = node_logits(model, X2.ImageSource(processor, sel_id, image, pool), sel, qs, layer, device)
    sig = [1.0 / (1.0 + np.exp(-x.astype(np.float64))) for x in lg]
    a, a1, a2 = picks(tree, sig[0][0], [sig[1][0], sig[1][1]] if ch else None)
    b = int(np.argmax(sig[0][0][:n])) if n else -1
    inside = M.intersections([pool[a]], [pool[a1]])[0, 0] / max(1, M.areas([pool[a]])[0]) if ch else float("nan")
    row.update({"ours2_rle": json.dumps(pool[a]), "ours2_iou": float(ious[a]), "ours2_sigma": float(sig[0][0][a]),
                "ours2_kind": kind[a], "ours2tree_rle": json.dumps(pool[a2]), "ours2tree_iou": float(ious[a2]),
                "ours2tree_sigma": float(sig[-1][-1][a2]), "ours2tree_kind": kind[a2],
                "tree_object_kind": kind[a1] if ch else "none",
                "tree_step2_candidates": int(len(tree.strict_descendants(a1))) if ch else 0,
                "flat_outside_object": bool(ch and a != a1 and not tree.is_strict_descendant(a, a1)),
                "flat_in_object_frac": float(inside),
                "nodes_only_iou": float(ious[b]) if n else 0.0,
                "nodes_rule_iou": float(ious[rule_pick(lg[0][0][:n], sam_iou, *rr[:2])]) if rr and n else float("nan")})
    return row


def run_shard(cfg: dict, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D3", "D4", "D10", "D14", "D17", "D23"] + (["D21"] if cfg.get("s2", {}).get("stage_suffix")
                                                                       else []))
    items = EE.evaluated(EE.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    ident = identity(cfg, dataset, [s.sample_id for s in items], smoke)
    d = cache_dir(cfg, dataset, ident, smoke)
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    stage, s2_run, ev, _ = X2.s2_inputs(cfg)
    ckpt, ck_run = X2.S2E.load_checkpoint(cfg, False)
    if ck_run != s2_run or read_json(ev / "metrics.json")["checkpoint_step"] != ckpt["meta"]["step"]:
        raise ProtocolError(f"checkpoint {ck_run.name} step {ckpt['meta']['step']} is not the one {ev.name} evaluated")
    run = start_run(f"e_select_{dataset}_{stage}", cfg, smoke, {"shard": shard, "num_shards": num_shards,
                                                               "cache": str(d), "s2_run": str(s2_run),
                                                               "s2_eval_run": str(ev)})
    models = X2.load_models(cfg, ckpt)
    del ckpt
    st = B.Sam3Text(cfg)
    cap, rr = X.node_cap(), rerank_weight(cfg, stage)
    mine = items[shard::num_shards]
    tau = float(read_json(ev / "metrics.json")["single_step"]["tau_rej"])
    rows, t0 = [], time.time()
    for i, s in enumerate(mine, 1):
        t1 = time.time()
        r = select_sample(models, st, cfg, s, cap, rr)
        rows.append({**r, "ours2_above_tau_rej": bool(r["ours2_sigma"] >= tau), "tau_rej": tau,
                     "select_seconds": time.time() - t1})
        print(f"[select {i}/{len(mine)}] {s.sample_id} nodes {r['nodes_capped']} inst {r['instances']} flat "
              f"{r['ours2_kind']} {r['ours2_iou']:.3f} tree {r['ours2tree_kind']} {r['ours2tree_iou']:.3f} "
              f"{time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"samples": len(mine), "seconds": time.time() - t0})


def shard_path(cfg: dict, dataset: str, items: list, shard: int, smoke: bool) -> Path:
    ident = identity(cfg, dataset, [s.sample_id for s in items], smoke)
    return cache_dir(cfg, dataset, ident, smoke) / "shards" / f"{shard:05d}.parquet"


def kind_rates(v: pd.Series) -> dict:
    return {k: float(x) for k, x in v.str.rstrip("0123456789").value_counts(normalize=True).sort_index().items()}


def localization(df: pd.DataFrame, bs: dict) -> dict:
    img = df.image.to_numpy()

    def ci(v):
        return mean_ci(np.asarray(v, np.float64), img, bs["n"], bs["seed"])
    out = {}
    for name, col in (("ours2", "ours2_iou"), ("ours2tree", "ours2tree_iou"), ("s2_nodes_only", "nodes_only_iou"),
                      ("s2_nodes_rule", "nodes_rule_iou"), ("sam3text", "sam3_iou"), ("best_node", "best_node_iou"),
                      ("best_pool", "best_pool_iou")):
        if len(df) and df[col].notna().all():
            out[f"{name}_mask_iou"] = ci(df[col])
            out[f"{name}_success@0.5"] = ci(df[col] >= THR)
    ch = df[df.chain]
    out.update({"n": int(len(df)), "chain_referents": int(len(ch)),
                "ours2_pick_kind": kind_rates(df.ours2_kind), "ours2tree_pick_kind": kind_rates(df.ours2tree_kind),
                "ours2_instance_pick_rate": float((df.ours2_kind.str.rstrip("0123456789") != "node").mean()),
                "ours2tree_instance_pick_rate": float((df.ours2tree_kind.str.rstrip("0123456789") != "node").mean()),
                "tree_object_pick_kind": kind_rates(ch.tree_object_kind) if len(ch) else None,
                "flat_outside_tree_object_rate": float(ch.flat_outside_object.mean()) if len(ch) else None,
                "flat_in_tree_object_frac_median": float(ch.flat_in_object_frac.median()) if len(ch) else None,
                "tree_step2_candidates_median": float(ch.tree_step2_candidates.median()) if len(ch) else None,
                "flat_tree_same_pick_rate": float((df.ours2_rle == df.ours2tree_rle).mean())
                if "ours2_rle" in df else None,
                "ours2_above_tau_rej_rate": float(df.ours2_above_tau_rej.mean()),
                "instances_mean": float(df.instances.mean()), "nodes_capped_mean": float(df.nodes_capped.mean())})
    return out


def merge(cfg: dict, dataset: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D23"])
    items = EE.evaluated(EE.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    ident = identity(cfg, dataset, [s.sample_id for s in items], smoke)
    d = cache_dir(cfg, dataset, ident, smoke)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} select shards of {dataset} missing under {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(num_shards)], ignore_index=True)
    if sorted(df.sample_id) != [s.sample_id for s in items]:
        raise ProtocolError(f"{dataset}: merged selections do not match the evaluation set")
    bs = cfg["edit"]["metrics"]["bootstrap"]
    stage = ident["s2_stage"]
    run = start_run(f"e_select_{dataset}_{stage}_merge", cfg, smoke, {"cache": str(d)})
    out = {"deviation": "D23", "label": "dev" if dataset == "paco_edit" else "edit benchmark", "dataset": dataset,
           "smoke": smoke, "samples": len(df), "images": int(df.image.nunique()), "s2_stage": stage,
           "s2_run": ident["s2_run"], "s2_eval_run": ident["s2_eval_run"], "checkpoint": ident["checkpoint"],
           "select": settings(cfg), "rerank": ident["rerank"], "node_cap": ident["node_cap"],
           "picks": {"ours2": "flat: top-1 sigma of the referent probe over the pool, no rejection",
                     "ours2tree": "tree: object over the pool, part among its strict descendants (itself if none); "
                                  "flat for plain referents",
                     "pool": "capped SAM 3 nodes + top-N SAM 3 text instances of the referent and, for chain "
                             "referents, of the object; one tree over the union"},
           "localization": localization(df, bs),
           "localization_chain_referents": localization(df[df.chain], bs) if df.chain.any() else None,
           "by_subset": {k: localization(g, bs) for k, g in df.groupby("subset")} if df.subset.nunique() > 1 else None,
           "seconds_per_sample": float(df.select_seconds.mean())}
    df.drop(columns=["ours2_rle", "ours2tree_rle"]).to_csv(run / "samples.csv", index=False)
    write_json(run / "metrics.json", out)
    print(json.dumps(out["localization"], indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=EE.DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    ensure_deviation_log(cfg)
    num_shards = 1 if args.smoke else cfg["edit"]["num_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.dataset, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.dataset, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
