import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import leak_audit as LA  # noqa: E402
import s2_train as S2T  # noqa: E402
import ta_paco_nodes as TPN  # noqa: E402
import ta_train as TAT  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# Read-only checks of D41 test results (nothing is trained, cached or changed).
#   leak     scripts/leak_audit.py with the train side replaced by every image a D41 run trains on: the worklist
#            training roles (PACO-LVIS 10k, gRefCOCO no-target), the PACO-LVIS train rest cache, and the S2 node
#            caches of the other sources; test side unchanged (all test sets). Exact pixel sha1 + 64-bit dHash.
#   partonomy  per (image, part) IoU of two partonomy_eval caches of 'ours' (e.g. S2 long vs ta_tree) on the same
#            parts with the same GT rule, to see where a difference comes from.


def ta_train_images(cfg: dict) -> dict[str, str]:
    _, _, _, wl = manifest_inputs(cfg, False)
    out = {r.path: r.role for r in wl[wl.role.str.endswith("_train") | wl.role.str.endswith("_train_nt")]
           .itertuples()}
    for source in S2T.source_weights(cfg["ta"]):
        if source == "paco_train":
            continue
        _, files = S2T.node_cache(cfg, source, False)
        for f in files:
            for r in torch.load(f, map_location="cpu", weights_only=False)["records"]:
                out[r["path"]] = source
    if cfg["ta"]["paco_rest"]:
        d = TPN.out_dir(cfg, False)
        files = sorted((d / "shards").glob("*.pt"))
        if len(files) != TPN.num_shards(cfg, False):
            raise ProtocolError(f"PACO rest cache incomplete in {d}")
        for f in files:
            for r in torch.load(f, map_location="cpu", weights_only=False)["records"]:
                out[r["path"]] = "paco_train_rest"
    return out


def partonomy_cache(cfg: dict, stage: str) -> pd.DataFrame:
    # the complete cache of the stage (every shard of its identity present); several complete ones: the newest
    base = Path(cfg["cache_root"]) / "full" / "partonomy_eval" / stage
    done = []
    for d in base.glob("*"):
        if not (d / "shards").is_dir() or not (d / "identity.json").exists():
            continue
        n = json.loads((d / "identity.json").read_text())["identity"]["num_shards"]
        files = sorted((d / "shards").glob("*.parquet"))
        if len(files) == n:
            done.append((max(f.stat().st_mtime for f in files), d, files))
    if not done:
        raise ProtocolError(f"no complete partonomy cache under {base}")
    _, d, files = max(done)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def part_ious(qs: list[dict], df: pd.DataFrame, col: str, gt_mask, iou) -> pd.DataFrame:
    # identification questions only (the reported setting); one row per (question, part)
    rows = {(r.image, r.part): r for r in df.itertuples()}
    out = []
    for i, q in enumerate(qs):
        if q["question_type"] != "identification_with_label":
            continue
        for p in q["parts"]:
            row = rows[(q["image_path"], p)]
            rle = json.loads(getattr(row, col))
            pred = M.decode(rle).astype(bool) if rle is not None else None
            gt = gt_mask(q["segmentations"][p], (row.h, row.w))
            out.append({"q": i, "image": q["image_path"], "label": q["image_label"], "part": p,
                        "iou": iou(pred, gt), "pred_area": int(pred.sum()) if pred is not None else 0,
                        "gt_area": int(gt.sum())})
    return pd.DataFrame(out)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on=["q", "image", "label", "part"], suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ProtocolError("the two caches cover different parts")
    d = m.iou_b - m.iou_a
    small = m.gt_area_a < np.percentile(m.gt_area_a, 25)
    return {"parts": int(len(m)), "micro_a": float(m.iou_a.mean()), "micro_b": float(m.iou_b.mean()),
            "b_better_by_0.1": float((d > 0.1).mean()), "a_better_by_0.1": float((d < -0.1).mean()),
            "same_within_0.1": float((d.abs() <= 0.1).mean()),
            "success@0.5_a": float((m.iou_a >= 0.5).mean()), "success@0.5_b": float((m.iou_b >= 0.5).mean()),
            "empty_pred_a": float((m.pred_area_a == 0).mean()), "empty_pred_b": float((m.pred_area_b == 0).mean()),
            "pred_to_gt_area_median_a": float((m.pred_area_a / m.gt_area_a.clip(lower=1)).median()),
            "pred_to_gt_area_median_b": float((m.pred_area_b / m.gt_area_b.clip(lower=1)).median()),
            "smallest_quarter_micro": {"a": float(m.iou_a[small].mean()), "b": float(m.iou_b[small].mean())},
            "largest_gains": m.assign(d=d).sort_values("d", ascending=False)
            [["image", "label", "part", "iou_a", "iou_b"]].head(15).to_dict("records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a ta_test_<variant>.yaml")
    ap.add_argument("--check", required=True, choices=("leak", "partonomy"))
    ap.add_argument("--against", default="s2_from_s1_long", help="partonomy: the other 'ours' stage")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_hamming", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.check == "leak":
        LA.train_images = ta_train_images
        sys.argv = [sys.argv[0], "--config", args.config, "--workers", str(args.workers),
                    "--max_hamming", str(args.max_hamming)]
        LA.main()
        return
    import partonomy_eval as PE
    qs = PE.questions(Path(cfg["data"]["partonomy"]))
    stage = TAT.stage(cfg)
    a = part_ious(qs, partonomy_cache(cfg, args.against), "pred_nodes", PE.gt_mask, PE.iou)
    b = part_ious(qs, partonomy_cache(cfg, stage), "pred_nodes", PE.gt_mask, PE.iou)
    res = {"label": "test (read-only check)", "a": args.against, "b": stage, **compare(a, b)}
    run = start_run(f"ta_check_partonomy_{stage}", cfg, False, {"against": args.against})
    a.merge(b, on=["q", "image", "label", "part"], suffixes=("_a", "_b")).to_csv(run / "parts.csv", index=False)
    write_json(run / "metrics.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "largest_gains"}, indent=1))
    print(run)


if __name__ == "__main__":
    main()
