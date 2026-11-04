import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gres_eval as GE  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            require_approved, resolve_run, start_run, write_json)

# Deviation D45: GroundingSuite-Eval (GSEval; Hu et al., ICCV 2025, arXiv 2503.10596; hustvl/GSEval on the Hub):
# 3,7xx referring expressions on COCO unlabeled2017 images in four strata (stuff, part, multi-object, single-object),
# one ground-truth mask per expression. Scored as the official evaluate_grounding.py in mask mode (which PANORAMA,
# arXiv 2609.19143, copies verbatim): per-expression IoU of the predicted mask against the ground truth, gIoU = mean
# IoU per class_id and over all expressions; an empty prediction scores 0. Published zero-shot rows (PANORAMA Table
# 10) can therefore be cited unchanged. Query = the expression ('caption'), as PANORAMA's RES prompt uses it.
# Ours (as gres_eval, D33): the expression is the query, answer '{expression} <SEL>'; per-candidate sigma over the
# checkpoint's candidates; 'top1' = the highest candidate (every expression has a target, so nothing abstains) and
# 'set' = the union of candidates with sigma >= t, t being the checkpoint's gRefCOCO val threshold (gres_eval's
# 2-fold rule, frozen there; nothing is tuned on GSEval). Baselines: the D13 / D31 predictors (one mask or abstain).

STRATA = {1: "stuff", 2: "part", 3: "multi", 4: "single"}    # class_id -> stratum (PANORAMA eval/metrics_groundingsuite)
RULES = ("top1", "set")
FILES = ("scripts/gseval_eval.py", "scripts/gres_eval.py", "scripts/s3_pool.py", "scripts/s3_eval.py",
         "scripts/s2_train.py", "scripts/b_baselines.py", "scripts/b_baselines2.py")


def expressions(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data"]["gseval"])
    rows = []
    for line in (root / "GroundingSuite-Eval.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        seg = d["segmentation"]
        if int(d["class_id"]) not in STRATA:
            raise ProtocolError(f"expression {d['idx']}: unknown class_id {d['class_id']}")
        rows.append({"qid": f"gseval|{int(d['idx'])}", "image": f"gseval:{Path(d['image_path']).stem}",
                     "path": str(root / d["image_path"]), "class_id": int(d["class_id"]), "label": d.get("label", ""),
                     "text": d["caption"].strip(), "h": int(seg["size"][0]), "w": int(seg["size"][1]),
                     "gt": json.dumps({"size": [int(s) for s in seg["size"]], "counts": seg["counts"]})})
    df = pd.DataFrame(rows)
    if df.empty or df.qid.duplicated().any():
        raise ProtocolError("GSEval: no expressions or duplicate ids")
    return df


class Ours(GE.Ours):
    # the S2 selector over the D30 pool, exactly as on gRefCOCO (image, expressions) -> per expression
    # {'nodes': [(sigma, rle)...], 'aug': [...]} in descending sigma
    pass


class Baseline:
    def __init__(self, cfg: dict, method: str):
        import b_baselines as B
        if method not in B.PREDICTORS:
            import b_baselines2  # noqa: F401
        self.pred = B.PREDICTORS[method](cfg)

    def __call__(self, image: np.ndarray, texts: list[str]) -> list[dict]:
        out = []
        for m, _, _, acc in self.pred(image, texts):
            rle = M.encode(m)[0] if (acc and m is not None and m.any()) else None
            out.append({"pred": [(1.0, rle)] if rle else []})
        return out


def stage_name(cfg: dict, method: str) -> str:
    return GE.stage_name(cfg, method)


def identity(cfg: dict, method: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "gseval_eval", "method": method, "stage": stage_name(cfg, method),
             "num_shards": cfg["gseval_eval"]["num_shards"], "code": h.hexdigest()}
    if method == "ours":
        import s2_eval as S2E
        ident.update(run=S2E.load_checkpoint(cfg, False)[1].name, checkpoint=cfg["s2_eval"]["checkpoint"],
                     pool=cfg["s3_pool"]["instances_per_phrase"], sam3_text=cfg["baselines"]["sam3_text"])
    else:
        ident.update(models=cfg["models"], baselines=cfg["baselines"])
    return ident


def out_dir(cfg: dict, method: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "gseval_eval" / stage_name(cfg, method) / canonical_hash(ident)[:16]


def approvals(cfg: dict, method: str) -> list[str]:
    if method != "ours":
        return ["D45", "D13", "D31"]
    import s2_train as S2T
    return ["D45", "D30"] + S2T.deviations(cfg)


def run_shard(cfg: dict, method: str, shard: int) -> None:
    require_approved(cfg, approvals(cfg, method))
    df = expressions(cfg)
    n = cfg["gseval_eval"]["num_shards"]
    images = sorted(df.image.unique())[shard::n]
    ident = identity(cfg, method)
    d = out_dir(cfg, method, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"gseval_{stage_name(cfg, method)}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    predict = Ours(cfg) if method == "ours" else Baseline(cfg, method)
    rows, t0 = [], time.time()
    for i, key in enumerate(images, 1):
        g = df[df.image == key]
        image = load_rgb(assert_real_path(g.path.iloc[0]))
        if image.shape[:2] != (g.h.iloc[0], g.w.iloc[0]):
            raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {(g.h.iloc[0], g.w.iloc[0])}")
        for q, res in zip(g.itertuples(), predict(image, g.text.tolist())):
            row = {"qid": q.qid, "image": key, "class_id": int(q.class_id), "gt": q.gt}
            for mode, cands in res.items():
                row[f"cand_{mode}"] = json.dumps([[s, r] for s, r in cands])
            rows.append(row)
        if i % 25 == 0:
            print(f"[{i}/{len(images)}] expressions {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "expressions": len(rows), "seconds": time.time() - t0})


def iou_rows(df: pd.DataFrame, col: str, rule: str, t: float) -> pd.DataFrame:
    # per expression: the official mask-mode IoU of the union of the chosen candidates (empty prediction -> 0)
    out = []
    for q in df.itertuples():
        cands = [(s, M._coco(r)) for s, r in json.loads(getattr(q, col))]
        chosen = [r for s, r in cands if s >= t] if rule == "set" else ([cands[0][1]] if cands else [])
        pred = (mu.merge(chosen) if len(chosen) > 1 else chosen[0]) if chosen else None
        gt = M._coco(json.loads(q.gt))
        pa, ga = GE._area(pred), GE._area(gt)
        inter = GE._area(mu.merge([pred, gt], intersect=True)) if pred is not None else 0
        union = pa + ga - inter
        out.append((q.image, int(q.class_id), inter / union if union else 0.0, inter, union, pa == 0))
    return pd.DataFrame(out, columns=["image", "class_id", "iou", "inter", "union", "pred_empty"])


def summarize(s: pd.DataFrame, bs: dict) -> dict:
    def block(g: pd.DataFrame) -> dict:
        return {"gIoU": mean_ci(g.iou.to_numpy(float), g.image.to_numpy(), bs["n"], bs["seed"]),
                "cIoU": float(g.inter.sum() / max(1, g.union.sum())), "expressions": int(len(g)),
                "empty_predictions": int(g.pred_empty.sum())}
    out = {"all": block(s)}
    for cid, name in STRATA.items():
        g = s[s.class_id == cid]
        if len(g):
            out[name] = block(g)
    return out


def gres_threshold(cfg: dict, method: str) -> float:
    # the checkpoint's gRefCOCO 'set' threshold, chosen on gRefCOCO val by gres_eval and frozen there
    run = resolve_run(cfg, f"gres_{stage_name(cfg, method)}", "latest", False)
    res = read_json(run / "metrics.json")["results"]
    key = next((k for k in res if k.endswith("/set") or k == "tuned/set"), None)
    if key is None or "t_val" not in res[key]:
        raise ProtocolError(f"{run}: no 'set' rule with a val threshold in its gRefCOCO results")
    return float(res[key]["t_val"])


def merge(cfg: dict, method: str) -> None:
    require_approved(cfg, approvals(cfg, method))
    d = out_dir(cfg, method, identity(cfg, method))
    n = cfg["gseval_eval"]["num_shards"]
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if len(df) != len(expressions(cfg)) or df.qid.duplicated().any():
        raise ProtocolError(f"{len(df)} rows do not match the benchmark's expressions")
    bs = cfg["v2"]["bootstrap"]
    run = start_run(f"gseval_{stage_name(cfg, method)}", cfg, False, {"cache": str(d)})
    df.to_csv(run / "eval_queries.csv", index=False)
    res = {"counts_by_class_id": {str(k): int(v) for k, v in df.class_id.value_counts().sort_index().items()}}
    if method == "ours":
        t = gres_threshold(cfg, method)
        cols = [c for c in df.columns if c.startswith("cand_")]
        for col in cols:
            for rule in RULES:
                res[f"{col[5:]}/{rule}"] = dict(summarize(iou_rows(df, col, rule, t if rule == "set" else -1.0), bs),
                                                threshold=t if rule == "set" else None)
    else:
        res["official"] = summarize(iou_rows(df, "cand_pred", "set", -1.0), bs)
    write_json(run / "metrics.json", {"label": "test", "deviation": "D45", "protocol": "GSEval (arXiv 2503.10596)",
                                      "method": method, "stage": stage_name(cfg, method),
                                      "images": int(df.image.nunique()), "results": res})
    print(json.dumps({k: {s: round(v[s]["gIoU"]["mean"] * 100, 2) for s in v if isinstance(v[s], dict) and "gIoU" in v[s]}
                      for k, v in res.items() if isinstance(v, dict) and "all" in v}, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method)
    else:
        if not 0 <= args.shard < cfg["gseval_eval"]["num_shards"]:
            raise ProtocolError(f"--shard must be in 0..{cfg['gseval_eval']['num_shards'] - 1}")
        with torch.no_grad():
            run_shard(cfg, args.method, args.shard)


if __name__ == "__main__":
    main()
