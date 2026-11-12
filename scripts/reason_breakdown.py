import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.run import ProtocolError, load_config, resolve_run, start_run, write_json  # noqa: E402

# D50 breakdown (read-only, CPU; nothing re-predicted or tuned): the reasoning runs' per-query results split by
# target size (GT area / image area) and query form (sentence vs phrase), plus what each method's misses look like:
# 'inside' = the prediction lies mostly within the target (a sub-region, e.g. a part of the asked object),
# 'covers' = it contains most of the target and more, 'elsewhere' = neither, 'empty' = no mask.

BENCHES = ("instructpart_human", "instructpart_gpt", "reasonseg")
METHODS = ("ours", "qwen_box_sam", "sa2va", "sam3_text_na")
SIZE_BINS = (0.0, 0.05, 0.2, 0.5, 1.0001)
MISS_IOU, MOSTLY = 0.5, 0.8


def size_bin(frac: np.ndarray) -> np.ndarray:
    labels = [f"{lo:.0%}-{hi:.0%}" for lo, hi in zip(SIZE_BINS, SIZE_BINS[1:])]
    return np.array(labels, dtype=object)[np.digitize(frac, SIZE_BINS[1:-1])]


def miss_kind(inter: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.full(len(inter), "elsewhere", dtype=object)
    inside = (pred > 0) & (inter >= MOSTLY * pred) & (pred < gt)
    covers = (pred > 0) & (inter >= MOSTLY * gt) & (pred >= gt)
    out[inside], out[covers], out[pred == 0] = "inside", "covers", "empty"
    return out


def query_table(rows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # rows: one method's eval_queries (qid, inter, union); meta: qid, gt_area, image_area, sentence
    df = rows.merge(meta, on="qid", how="inner", validate="one_to_one")
    if len(df) != len(meta) or len(df) != len(rows):
        raise ProtocolError(f"{len(rows)} result rows for {len(meta)} queries")
    pred = df.union - df.gt_area + df.inter
    if (df.inter > df.gt_area).any() or (pred < 0).any():
        raise ProtocolError("stored intersection / union do not fit the benchmark's ground truth")
    return df.assign(iou=np.where(df.union > 0, df.inter / df.union.clip(lower=1), 0.0), pred_area=pred,
                     size=size_bin(df.gt_area / df.image_area))


def summarize(tables: dict[str, pd.DataFrame]) -> dict:
    out = {"queries": int(len(next(iter(tables.values())))), "by_size": {}, "by_form": {}, "misses": {}}
    ref = next(iter(tables.values()))
    for b in sorted(ref["size"].unique()):
        out["by_size"][b] = {"n": int((ref["size"] == b).sum()),
                             **{m: float(t.iou[t["size"] == b].mean()) for m, t in tables.items()}}
    for form, sel in (("sentence", True), ("phrase", False)):
        n = int((ref.sentence == sel).sum())
        if n:
            out["by_form"][form] = {"n": n, **{m: float(t.iou[t.sentence == sel].mean()) for m, t in tables.items()}}
    for m, t in tables.items():
        miss = t[t.iou < MISS_IOU]
        kinds = pd.Series(miss_kind(miss.inter.to_numpy(), miss.pred_area.to_numpy(), miss.gt_area.to_numpy()))
        out["misses"][m] = {"n": int(len(miss)), **{k: float((kinds == k).mean()) if len(miss) else 0.0
                                                    for k in ("inside", "covers", "elsewhere", "empty")}}
    return out


def bench_meta(cfg: dict, bench: str) -> pd.DataFrame:
    import reason_eval as RE
    rows = []
    for qid, _, text, _, load_gt in RE.records(cfg, bench):
        gt = load_gt()
        rows.append({"qid": qid, "gt_area": int(gt.sum()), "image_area": int(gt.size), "sentence": RE.is_sentence(text)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reason_ours.yaml")
    args = ap.parse_args()
    import reason_eval as RE
    cfg = load_config(ROOT / args.config)
    stage = RE.stage_name(cfg, "ours")
    res = {}
    for bench in BENCHES:
        meta = bench_meta(cfg, bench)
        tables = {m: query_table(pd.read_csv(resolve_run(cfg, f"reason_{bench}_{stage if m == 'ours' else m}",
                                                         "latest", False) / "eval_queries.csv"), meta)
                  for m in METHODS}
        res[bench] = summarize(tables)
        for b, v in res[bench]["by_size"].items():
            print(bench, b, v, flush=True)
        print(bench, "misses", res[bench]["misses"], flush=True)
    run = start_run("reason_breakdown", cfg, False, {"stage": stage})
    write_json(run / "metrics.json", {"label": "test", "deviation": "D50", "stage": stage, "methods": list(METHODS),
                                      "size_bins": list(SIZE_BINS), "miss_iou": MISS_IOU, "mostly": MOSTLY,
                                      "results": res})
    print(run)


if __name__ == "__main__":
    main()
