import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2long_diag as SD  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.cache import load_tensors  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json, resolve_run, start_run, write_json  # noqa: E402

# Paired diagnosis of D41 evaluations on PACO-LVIS VAL (development only). For every listed evaluation stage (latest
# completed run) the stored node logits give the pick of every present object / part query; the pick is compared
# with the best node (S2 long diagnosis categories: correct, near duplicate, too big, too small, partial overlap,
# elsewhere). Every stage is then compared with a reference stage on the SAME queries: paired difference of
# selector success (covered queries) and of the wrong-level rate (too big + too small), with 95% intervals from an
# image-level bootstrap. Reads only stored outputs; no model is run.

WRONG_LEVEL = ("too_big", "too_small")


def pick_rows(stage: str, logits: np.ndarray, ious: np.ndarray, areas: np.ndarray, rles: list, qid: str, kind: str,
              image: str) -> dict:
    a, b = int(np.argmax(logits)), int(np.argmax(ious))
    inter = int(M.intersections([rles[a]], [rles[b]])[0, 0])
    return {"stage": stage, "qid": qid, "kind": kind, "image": image, "covered": bool(ious[b] >= 0.5),
            "success": bool(ious[a] >= 0.5), "pick_iou": float(ious[a]), "best_iou": float(ious[b]),
            "relation": SD.relation({"i": a, "area": areas[a]}, {"i": b, "area": areas[b]}, inter)}


def paired(ref: pd.DataFrame, other: pd.DataFrame, n_boot: int, seed: int) -> dict:
    # covered queries both stages answered; differences ref - other, resampled by image
    m = ref.merge(other, on=["qid", "image"], suffixes=("_ref", "_oth"))
    m = m[m.covered_ref]
    if m.empty:
        raise ProtocolError("no shared covered queries")
    m = m.assign(d_success=m.success_ref.astype(float) - m.success_oth.astype(float),
                 d_level=(m.relation_oth.isin(WRONG_LEVEL).astype(float)
                          - m.relation_ref.isin(WRONG_LEVEL).astype(float)))
    by_img = m.groupby("image")[["d_success", "d_level"]].agg(["sum", "count"])
    sums = by_img.xs("sum", axis=1, level=1).to_numpy()
    counts = by_img[("d_success", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    est = sums.sum(0) / counts.sum()
    return {"queries": int(len(m)), "images": int(len(counts)),
            "success_ref_minus_other": {"mean": float(est[0]), "ci95": [float(lo[0]), float(hi[0])]},
            "wrong_level_other_minus_ref": {"mean": float(est[1]), "ci95": [float(lo[1]), float(hi[1])]}}


def summary(df: pd.DataFrame) -> dict:
    cov = df[df.covered]
    wrong = cov[~cov.success]
    return {"covered": int(len(cov)), "success_if_covered": float(cov.success.mean()),
            "gIoU_no_rejection": float(df.pick_iou.mean()),
            "wrong_level_rate": float(cov.relation.isin(WRONG_LEVEL).mean()),
            "wrong_pick_relation": {k: round(float(v), 4)
                                    for k, v in wrong.relation.value_counts(normalize=True).items()},
            "by_kind": {k: float(g.success.mean()) for k, g in cov.groupby("kind")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", required=True, help="evaluation stages, e.g. ta_tree_eval ta_full_eval")
    ap.add_argument("--reference", default="ta_tree_eval")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if args.reference not in args.stages:
        raise ProtocolError("the reference stage must be one of --stages")
    cfg = load_config(args.config)
    runs = {s: resolve_run(cfg, s, "latest", False) for s in args.stages}
    layers = {s: int(read_json(r / "metrics.json")["layer"]) for s, r in runs.items()}
    _, manifest, mhash, _ = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    _, queries = open_queries(cfg, False, mhash, qwen_static(cfg))
    tau = cfg["proposals"]["node_dedup"]
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    qval = queries[(queries.split == "val") & queries.kind.isin(["object", "part"])]
    run = start_run("ta_diag", cfg, False, {"stages": {s: str(r) for s, r in runs.items()}, "layers": layers,
                                            "reference": args.reference, "split": "PACO-LVIS val"})
    rows = []
    for n_img, (sha1, g) in enumerate(qval.groupby("sha1"), 1):
        key = g.image_key.iloc[0]
        nodes = props.nodes(sha1, tau)
        if not nodes["rles"]:
            continue
        gts = paco.gt_rles(images[int(key.split(":")[1])])
        areas = M.areas(nodes["rles"]).astype(np.float64)
        stored = {s: load_tensors(r / "logits" / "probes" / f"{sha1}.safetensors", ["probes"])[0]["probes"][:, 0]
                  .numpy() for s, r in runs.items()}
        for s, x in stored.items():
            if x.shape[1] != len(nodes["rles"]):
                raise ProtocolError(f"{s}: {x.shape[1]} logits for {len(nodes['rles'])} nodes in {sha1}")
        for q in g.itertuples():
            ious = M.iou(nodes["rles"], [gts[json.loads(q.gt)[0]]])[:, 0]
            for s, x in stored.items():
                rows.append(pick_rows(s, x[q.offset], ious, areas, nodes["rles"], q.qid, q.kind, key))
        if n_img % 500 == 0:
            print(f"{n_img} images", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "diag_queries.csv", index=False)
    per = {s: df[df.stage == s].drop(columns="stage") for s in args.stages}
    out = {"label": "dev", "split": "PACO-LVIS val, present object / part queries", "layers": layers,
           "reference": args.reference, "stages": {s: summary(d) for s, d in per.items()},
           "paired_vs_reference": {s: paired(per[args.reference], d, args.bootstrap, 0)
                                   for s, d in per.items() if s != args.reference}}
    write_json(run / "metrics.json", out)
    print(json.dumps(out, indent=1))
    print(run)


if __name__ == "__main__":
    main()
