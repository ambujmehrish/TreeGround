import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gres_eval as G  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.run import load_config, require_approved, start_run, write_json  # noqa: E402

# D33 re-scoring from the existing GRES shards (no new inference): the ReLA / GSVA evaluation counts a predicted
# mask with fewer than 50 pixels as an empty prediction (correct on a no-target expression, a miss with I = 0 on a
# target). Reported beside the plain rule; for ours the threshold t is re-selected under the same rule (2-fold CV
# on val, the all-val t frozen for testA / testB), exactly as gres_eval.merge.

EMPTY_PX = 50


def scores(df: pd.DataFrame, col: str, rule: str, t: float, empty_px: int) -> pd.DataFrame:
    out = []
    for q in df.itertuples():
        cands = [(s, M._coco(r)) for s, r in json.loads(getattr(q, col))]
        chosen = [r for s, r in cands if s >= t] if rule == "set" else (
            [cands[0][1]] if cands and cands[0][0] >= t else [])
        pred = (mu.merge(chosen) if len(chosen) > 1 else chosen[0]) if chosen else None
        if pred is not None and int(mu.area(pred)) < empty_px:
            pred = None
        gt = None if q.gt == "null" else M._coco(json.loads(q.gt))
        pa, ga = G._area(pred), G._area(gt)
        inter = G._area(mu.merge([pred, gt], intersect=True)) if pred is not None and gt is not None else 0
        union = pa + ga - inter
        iou = (1.0 if pa == 0 else 0.0) if q.no_target else (inter / union if union else 0.0)
        out.append((q.image, q.no_target, iou, inter, union, pa == 0))
    return pd.DataFrame(out, columns=["image", "no_target", "iou", "inter", "union", "pred_empty"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--clip", action="store_true", help="SAM 3 text through gres_sam3clip (stage sam3_text_clip)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.clip:
        import gres_sam3clip  # noqa: F401  (patches gres_eval for the clipped SAM 3 text stage)
    require_approved(cfg, G.approvals(args.method))
    G.scores = lambda df, col, rule, t: scores(df, col, rule, t, EMPTY_PX)
    splits = {sp: G.load_split(cfg, args.method, sp) for sp in G.SPLITS if sp in cfg["gres_eval"]["splits"]}
    bs = cfg["v2"]["bootstrap"]
    val = splits["val"]
    out = {"deviation": "D33", "rule": f"prediction < {EMPTY_PX} px counts as empty (ReLA / GSVA)",
           "method": args.method, "stage": G.stage_name(cfg, args.method), "results": {}}
    for col in [c for c in val.columns if c.startswith("cand_")]:
        for rule in (("set", "top1") if args.method == "ours" else ("set",)):
            name, res = f"{col[5:]}/{rule}", {}
            if args.method == "ours":
                images = np.array(sorted(val.image.unique()))
                fold = dict(zip(images, np.random.default_rng(0).permutation(len(images)) % 2))
                f = val.image.map(fold).to_numpy()
                parts = []
                for k in (0, 1):
                    t, _ = G.best_t(val[f != k], col, rule)
                    parts.append(G.scores(val[f == k], col, rule, t).assign(t=t))
                res["val (2-fold t)"] = G.metrics(pd.concat(parts), bs)
                t_all = G.best_t(val, col, rule)[0]
                res["t_val"] = t_all
            else:
                t_all = 0.5
                res["val"] = G.metrics(G.scores(val, col, rule, t_all), bs)
            for sp, df in splits.items():
                if sp != "val":
                    res[sp] = G.metrics(G.scores(df, col, rule, t_all), bs)
            out["results"][name] = res
    run = start_run(f"gres_rescore50_{G.stage_name(cfg, args.method)}", cfg, False, {})
    write_json(run / "metrics.json", out)
    print(json.dumps({k: {s: (round(100 * v[s]["gIoU"]["mean"], 1), round(100 * v[s]["cIoU"], 1),
                              None if v[s]["N-acc"] is None else round(100 * v[s]["N-acc"], 1))
                          for s in v if isinstance(v[s], dict)} for k, v in out["results"].items()}, indent=1))
    print(run)


if __name__ == "__main__":
    main()
