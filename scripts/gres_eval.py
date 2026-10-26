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

from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground.data import grefcoco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import node_cap  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# Deviation D33: the standard GRES protocol (Liu et al., CVPR 2023; ReLA / GSVA evaluation) on the official
# gRefCOCO release (FudanCVL/gRefCOCO, grefs(unc).json, identical md5 to TG_DATA): every expression of the split
# (val, testA, testB), GT = union of the target masks, no-target expressions have an empty GT. Metrics: gIoU = mean
# per-expression IoU where a no-target expression scores 1 for an empty prediction and 0 otherwise; cIoU = total
# intersection / total union over all expressions (a wrong no-target prediction adds its area to the union); N-acc
# = empty-prediction rate on no-target expressions; T-acc = non-empty rate on target expressions.
# Ours (query = the expression, answer '{expression} <SEL>' as in S2 training): per candidate sigma over the capped
# D3/D4 nodes ('nodes') or nodes + the expression's top-3 SAM 3 text instances ('aug', D30); prediction = union of
# the candidates with sigma >= t ('set', multi-target) or the top-1 if its sigma >= t ('top1'); t on a fixed grid is
# chosen by 2-fold cross-validation over val images (each fold scored with the t that maximises gIoU on the other)
# and the t of all val is frozen for testA / testB. Baselines: the D13 / D31 predictors (one mask or abstain).

SPLITS = ("val", "testA", "testB")
GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)
KEEP_MIN, KEEP_MAX = 0.05, 24
FILES = ("scripts/gres_eval.py", "scripts/s3_pool.py", "scripts/s3_eval.py", "scripts/s2_train.py",
         "scripts/b_baselines.py", "scripts/b_baselines2.py")


def expressions(cfg: dict, split: str) -> pd.DataFrame:
    root = Path(cfg["data"]["grefcoco"])
    refs, images = grefcoco.load(root)
    anns = {a["id"]: a for a in json.loads((root / "instances.json").read_text())["annotations"]}
    rows = []
    for r in refs:
        if r["split"] != split:
            continue
        im = images[r["image_id"]]
        nt = grefcoco.is_no_target(r)
        ids = [] if nt else (r["ann_id"] if isinstance(r["ann_id"], list) else [r["ann_id"]])
        for s in r["sentences"]:
            rows.append({"qid": f"grefcoco|{split}|{s['sent_id']}", "image": f"grefcoco:{r['image_id']}",
                         "file": im["file_name"], "h": im["height"], "w": im["width"], "text": s["sent"],
                         "no_target": nt, "ann_ids": json.dumps(ids)})
    df = pd.DataFrame(rows)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate sentence ids")
    return df, anns


def gt_rle(anns: dict, ids: list[int], h: int, w: int):
    if not ids:
        return None
    rles = [M._coco(M.from_annotation(anns[a]["segmentation"], h, w)) for a in ids]
    return mu.merge(rles) if len(rles) > 1 else rles[0]


class Ours:
    def __init__(self, cfg: dict):
        import s2_nodes as S2N
        import s3_eval as S3E
        import s3_pool as S3P
        from treeground import proposals as P
        from treeground.identity import qwen_static
        self.cfg, self.S2N = cfg, S2N
        dev = cfg["device"]
        self.model, self.proc, self.extra, self.sel, self.meta, self.run = S3E.load_model(cfg, qwen_static(cfg), dev)
        m, p = cfg["models"], cfg["proposals"]
        self.sam, self.sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], dev, p["dtype"], cfg.get("hf_home"))
        self.text = S3P.TextInstances(cfg)
        self.cap = node_cap(cfg, False)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, texts: list[str]) -> list[dict]:
        import s2_train as S2T
        import x_transfer_s2 as X2
        from treeground import s2 as S2
        nodes = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        pool, owner = list(nodes), []
        for lst in self.text(image, texts, self.cfg["s3_pool"]["instances_per_phrase"]):
            own = []
            for msk, _ in lst:
                if msk.any():
                    own.append(len(pool))
                    pool.append(M.encode(msk)[0])
            owner.append(own)
        if not pool:
            return [{"nodes": [], "aug": []} for _ in texts]
        qs = [S2T.Query(f"q{i}", "object", S2.user_text(f"{t} {mllm.SEL}"), f"{S2.user_text(t)} {mllm.SEL}", [])
              for i, t in enumerate(texts)]
        z, probes = S2T.recompute(self.model, X2.ImageSource(self.proc, self.extra["sel_id"], image, pool), None, qs,
                                  self.meta["layer"], self.cfg["device"], self.cfg["s2_eval"]["batch_size"])
        e = self.sel.nodes(z.half().to(self.cfg["device"]))
        n = len(nodes)
        out = []
        for pr, own in zip(probes, owner):
            sig = torch.sigmoid(self.sel.scores(self.sel.queries(pr.half()), e)[0]).float().cpu().numpy()
            res = {}
            for mode, idx in (("nodes", np.arange(n)), ("aug", np.array(list(range(n)) + own, np.int64))):
                order = idx[np.argsort(-sig[idx], kind="stable")][:KEEP_MAX] if len(idx) else idx
                res[mode] = [(float(sig[i]), pool[i]) for i in order if sig[i] >= KEEP_MIN or i == order[0]]
            out.append(res)
        return out


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
    if method != "ours":
        return method
    import s2_train as S2T
    return S2T.s2_stage(cfg)


def identity(cfg: dict, method: str, split: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "gres_eval", "split": split, "method": method, "stage": stage_name(cfg, method),
             "num_shards": cfg["gres_eval"]["num_shards"], "code": h.hexdigest()}
    if method == "ours":
        import s2_eval as S2E
        ident.update(run=S2E.load_checkpoint(cfg, False)[1].name, checkpoint=cfg["s2_eval"]["checkpoint"])
    else:
        ident.update(models=cfg["models"], baselines=cfg["baselines"])
    return ident


def out_dir(cfg: dict, method: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "gres_eval" / stage_name(cfg, method) / ident["split"] / canonical_hash(ident)[:16]


def approvals(method: str) -> list[str]:
    return ["D33"] + (["D10", "D21", "D30"] if method == "ours" else ["D13"])


def run_shard(cfg: dict, method: str, split: str, shard: int) -> None:
    require_approved(cfg, approvals(method))
    df, anns = expressions(cfg, split)
    n = cfg["gres_eval"]["num_shards"]
    images = sorted(df.image.unique())[shard::n]
    ident = identity(cfg, method, split)
    d = out_dir(cfg, method, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"gres_{stage_name(cfg, method)}_{split}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    predict = Ours(cfg) if method == "ours" else Baseline(cfg, method)
    rows, t0 = [], time.time()
    for i, key in enumerate(images, 1):
        g = df[df.image == key]
        image = load_rgb(assert_real_path(str(grefcoco.image_path(cfg["data"]["coco_images"], g.file.iloc[0]))))
        if image.shape[:2] != (g.h.iloc[0], g.w.iloc[0]):
            raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation")
        for q, res in zip(g.itertuples(), predict(image, g.text.tolist())):
            gt = gt_rle(anns, json.loads(q.ann_ids), q.h, q.w)
            row = {"qid": q.qid, "image": key, "no_target": bool(q.no_target),
                   "gt": json.dumps(None if gt is None else {"size": gt["size"], "counts": gt["counts"].decode()
                                                             if isinstance(gt["counts"], bytes) else gt["counts"]})}
            for mode, cands in res.items():
                row[f"cand_{mode}"] = json.dumps([[s, r] for s, r in cands])
            rows.append(row)
        print(f"[{i}/{len(images)}] {key} expressions {len(g)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "expressions": len(rows), "seconds": time.time() - t0})


def _area(r) -> int:
    return int(mu.area(r)) if r is not None else 0


def scores(df: pd.DataFrame, col: str, rule: str, t: float) -> pd.DataFrame:
    # per expression: IoU (GRES convention), intersection, union, predicted empty
    out = []
    for q in df.itertuples():
        cands = [(s, M._coco(r)) for s, r in json.loads(getattr(q, col))]
        if rule == "set":
            chosen = [r for s, r in cands if s >= t]
        else:
            chosen = [cands[0][1]] if cands and cands[0][0] >= t else []
        pred = (mu.merge(chosen) if len(chosen) > 1 else chosen[0]) if chosen else None
        gt = None if q.gt == "null" else M._coco(json.loads(q.gt))
        pa, ga = _area(pred), _area(gt)
        inter = _area(mu.merge([pred, gt], intersect=True)) if pred is not None and gt is not None else 0
        union = pa + ga - inter
        if q.no_target:
            iou = 1.0 if pa == 0 else 0.0
        else:
            iou = inter / union if union else 0.0
        out.append((q.image, q.no_target, iou, inter, union, pa == 0))
    return pd.DataFrame(out, columns=["image", "no_target", "iou", "inter", "union", "pred_empty"])


def metrics(s: pd.DataFrame, bs: dict) -> dict:
    nt, tg = s[s.no_target], s[~s.no_target]
    return {"gIoU": mean_ci(s.iou.to_numpy(float), s.image.to_numpy(), bs["n"], bs["seed"]),
            "cIoU": float(s.inter.sum() / max(1, s.union.sum())),
            "N-acc": float(nt.pred_empty.mean()) if len(nt) else None,
            "T-acc": float((~tg.pred_empty).mean()) if len(tg) else None,
            "expressions": int(len(s)), "no_target": int(len(nt))}


def best_t(df: pd.DataFrame, col: str, rule: str) -> tuple[float, list]:
    curve = [(float(t), float(scores(df, col, rule, t).iou.mean())) for t in GRID]
    return max(curve, key=lambda x: x[1])[0], curve


def load_split(cfg: dict, method: str, split: str) -> pd.DataFrame:
    ident = identity(cfg, method, split)
    d = out_dir(cfg, method, ident)
    n = cfg["gres_eval"]["num_shards"]
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{split}: {len(missing)} of {n} shards missing in {d}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if len(df) != len(expressions(cfg, split)[0]) or df.qid.duplicated().any():
        raise ProtocolError(f"{split}: {len(df)} rows do not match the split's expressions")
    return df


def merge(cfg: dict, method: str) -> None:
    require_approved(cfg, approvals(method))
    bs = cfg["v2"]["bootstrap"]
    splits = {sp: load_split(cfg, method, sp) for sp in SPLITS if cfg["gres_eval"]["splits"].count(sp)}
    run = start_run(f"gres_{stage_name(cfg, method)}", cfg, False, {})
    val = splits["val"]
    out = {"deviation": "D33", "method": method, "stage": stage_name(cfg, method), "protocol": "GRES (gRefCOCO)",
           "results": {}}
    for col in [c for c in val.columns if c.startswith("cand_")]:
        mode = col[5:]
        rules = ("set", "top1") if method == "ours" else ("set",)
        for rule in rules:
            name = f"{mode}/{rule}"
            if method == "ours":
                images = np.array(sorted(val.image.unique()))
                fold = dict(zip(images, np.random.default_rng(0).permutation(len(images)) % 2))
                f = val.image.map(fold).to_numpy()
                parts = []
                for k in (0, 1):
                    t, _ = best_t(val[f != k], col, rule)
                    parts.append(scores(val[f == k], col, rule, t).assign(t=t))
                res = {"val (2-fold t)": metrics(pd.concat(parts), bs),
                       "t_folds": [float(p.t.iloc[0]) for p in parts]}
                t_all, curve = best_t(val, col, rule)
                res["t_val"] = t_all
                res["val_curve"] = curve
            else:
                t_all = 0.5
                res = {"val": metrics(scores(val, col, rule, t_all), bs)}
            for sp, df in splits.items():
                if sp != "val":
                    res[sp] = metrics(scores(df, col, rule, t_all), bs)
            out["results"][name] = res
    write_json(run / "metrics.json", out)
    print(json.dumps({k: {s: (v[s]["gIoU"]["mean"], v[s]["cIoU"], v[s]["N-acc"]) for s in v
                          if isinstance(v[s], dict) and "gIoU" in v[s]} for k, v in out["results"].items()}, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method)
    else:
        with torch.no_grad():
            run_shard(cfg, args.method, args.split, args.shard)


if __name__ == "__main__":
    main()
