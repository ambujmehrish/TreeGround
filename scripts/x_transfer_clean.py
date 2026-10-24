import argparse
import hashlib
import io
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_transfer as BT  # noqa: E402
import v4_tree_value as V4  # noqa: E402
import x_giou as G  # noqa: E402
from treeground.data import lvis_hier, pascal_part  # noqa: E402
from treeground.data.images import pixel_sha1  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, load_config, read_json, require_approved,  # noqa: E402
                            resolve_run, start_run, write_json)

# Deviation D26: training-image leakage audit of the D7/D8 transfer sets and the LVIS-hier numbers recomputed on the
# clean image subset, from the published merge CSVs only (no model is run). Training images = every manifest image of
# split class train or dev (PACO-LVIS train/val, gRefCOCO train/val: the V2, S1 (D11), S2 (D10) and S2-long (D21)
# training and calibration/checkpoint images, a superset of what each model saw) plus Pascal-Part train (D11/D21).
# An evaluation image overlaps when its COCO id is a training COCO id (LVIS v1 val and PACO/gRefCOCO share COCO ids;
# train2014 ids are kept in train2017) or its pixel sha1 equals a training sha1.

DATASETS = ("lvis_hier", "pascal_part", "ade20k", "openimages_hier")
TRAIN_CLASSES = ("train", "dev")
SELECTORS = ("", "_s1", "_s2", "_s2_from_s1", "_s2_from_s1_long")
GIOU = ("v2", "s1", "s2", "s2_from_s1")
METHODS = ("sam3_text", "qwen_box_sam", "sa2va")
MODES = (("strict", False), ("d5", False), ("d6", True), ("no_rejection", False))
TOL = 1e-9


def array_sha1(arr: np.ndarray) -> str:
    h, w = arr.shape[:2]
    d = hashlib.sha1()
    d.update(f"{h}x{w}".encode())
    d.update(np.ascontiguousarray(arr).tobytes())
    return d.hexdigest()


def hash_paths(paths: list[str], workers: int) -> list[str]:
    for p in paths:
        assert_real_path(p)
    with Pool(workers) as pool:
        return [r[0] for r in pool.map(pixel_sha1, paths, chunksize=32)]


def coco_id(key: str) -> int | None:
    src, _, rest = key.partition(":")
    return int(rest) if src in ("lvis_val", "paco_lvis", "grefcoco") else None


def overlap(images: list[str], sha1: dict[str, str], train: pd.DataFrame) -> pd.DataFrame:
    # one row per evaluation image: matched by COCO id, by pixel sha1, and the training sources it matches
    ids = train.dropna(subset=["coco_id"])
    by_id = ids.groupby(ids.coco_id.astype(int))["tag"].agg(set)
    by_sha = train.groupby("sha1")["tag"].agg(set)
    rows = []
    for k in sorted(images):
        cid = coco_id(k)
        a = by_id.get(cid, set()) if cid is not None else set()
        b = by_sha.get(sha1.get(k), set())
        rows.append({"image": k, "coco_id_match": bool(a), "sha1_match": bool(b), "overlap": bool(a or b),
                     "sources": ",".join(sorted(a | b))})
    return pd.DataFrame(rows, columns=["image", "coco_id_match", "sha1_match", "overlap", "sources"])


def overlap_summary(ov: pd.DataFrame) -> dict:
    tags = pd.Series([t for s in ov.sources for t in s.split(",") if t])
    return {"eval_images": int(len(ov)), "overlap": int(ov.overlap.sum()),
            "overlap_coco_id": int(ov.coco_id_match.sum()), "overlap_sha1": int(ov.sha1_match.sum()),
            "by_training_source": {k: int(v) for k, v in tags.value_counts().sort_index().items()}}


def coverage_block(cov: pd.DataFrame, thr: float = 0.5) -> dict:
    # x_transfer.merge's coverage_uncapped block
    pc, ch = cov[cov.kind == "part"], cov[cov.kind == "chain"]
    return {"part_recall": float((pc.best_iou >= thr).mean()) if len(pc) else None, "parts": len(pc),
            "part_recall_by_area_bin": {b: float((g.best_iou >= thr).mean()) for b, g in pc.groupby("area_bin")},
            "chains": len(ch),
            "chain_object_found": float((ch.object_iou >= thr).mean()) if len(ch) else None,
            "chain_part_found": float((ch.best_iou >= thr).mean()) if len(ch) else None,
            "chain_feasible_mask_tree": float(ch.feasible_uncapped.astype(bool).mean()) if len(ch) else None,
            "single_level_chains": int((ch.part_ratio >= 0.5).sum())}


def natural_block(df: pd.DataFrame, n_rand: int, bs: dict, box: bool) -> dict | None:
    if not len(df):
        return None
    df = df.assign(single_level=df.single_level.astype(bool))
    nr = 0 if box else n_rand
    return {"all": V4.natural_summary(df, nr, bs, box=box),
            "two_level": V4.natural_summary(df[~df.single_level], nr, bs, box=box),
            "single_level": V4.natural_summary(df[df.single_level], nr, bs, box=box) if df.single_level.any() else None}


def x_transfer_block(run: Path, drop: set, n_rand: int, bs: dict) -> tuple[dict, dict]:
    m = read_json(run / "metrics.json")
    cov = pd.read_csv(run / "coverage.csv")
    out = {k: m[k] for k in ("label", "deviation", "tau_rej_chain", "layer") if k in m}
    out["images_dropped"] = len(drop & set(cov.image))
    out["images"] = int(m["images"]) - out["images_dropped"]
    out["coverage_uncapped"] = coverage_block(cov[~cov.image.isin(drop)])
    full = {"coverage_uncapped": coverage_block(cov)}
    frames = {}
    for mode, box in MODES:
        path = run / f"chains_{mode}.csv"
        if not path.exists():
            out[f"natural_{mode}"] = m.get(f"natural_{mode}")
            continue
        df = pd.read_csv(path)
        frames[mode] = df[~df.image.isin(drop)]
        out[f"natural_{mode}"] = natural_block(frames[mode], n_rand, bs, box)
        full[f"natural_{mode}"] = natural_block(df, n_rand, bs, box)
    return out, {"reproduces_published": same({k: m.get(k) for k in full}, full), "frames": frames}


def baseline_block(run: Path, drop: set, strict: pd.DataFrame | None, bs: dict, scratch: Path) -> tuple[dict, bool]:
    m = read_json(run / "metrics.json")
    df, ch = pd.read_csv(run / "queries.csv"), pd.read_csv(run / "chains.csv")
    for c in ("step1", "step2", "chain", "single_level"):
        ch[c] = ch[c].astype(bool)
    ok = same({k: m.get(k) for k in ("by_kind", "chains")}, BT.summarize(df, ch, bs))
    df, ch = df[~df.image.isin(drop)], ch[~ch.image.isin(drop)]
    out = {k: m[k] for k in ("method", "dataset", "label", "deviation", "box_matched", "models", "settings") if k in m}
    out.update(images=int(df.image.nunique()), queries={k: int((df.kind == k).sum()) for k in ("object", "part")},
               **BT.summarize(df, ch, bs))
    if strict is None:
        out["common_eligible_subset"] = "not available: no clean x_transfer_lvis_hier_merge chains_strict.csv"
        return out, ok
    path = scratch / "chains_strict_clean.csv"
    strict.to_csv(path, index=False)
    common, listed = BT.common_chains(path)
    ours = set(ch.qid)
    blk = {"x_transfer_chains": len(listed), "baseline_chains": len(ours), "shared_chains": len(ours & listed),
           "chains_match_x_transfer": ours == listed, "size": len(common & ours),
           "fraction": len(common & ours) / max(1, len(ours & listed))}
    blk.update(BT.summarize(BT.restrict(df, common & ours), ch[ch.qid.isin(common)], bs))
    out["common_eligible_subset"] = blk
    return out, ok


def giou_block(run: Path, drop: set, bs: dict) -> tuple[dict, bool]:
    m = read_json(run / "metrics.json")
    df = pd.read_csv(run / "chains.csv")
    df["single_level"] = df.single_level.astype(bool)

    def blocks(d):
        return {"all": G.summarize(d, bs),
                "two_level": G.summarize(d[~d.single_level], bs) if (~d.single_level).any() else None,
                "single_level": G.summarize(d[d.single_level], bs) if d.single_level.any() else None}
    ok = same({k: m.get(k) for k in ("all", "two_level", "single_level")}, blocks(df))
    return {k: m[k] for k in ("label", "deviation", "selector") if k in m} | blocks(df[~df.image.isin(drop)]), ok


def same(a, b) -> bool:
    a, b = json.loads(json.dumps(a, default=float)), json.loads(json.dumps(b, default=float))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOL * max(1.0, abs(a))
    return a == b


def latest(cfg: dict, stage: str) -> Path | None:
    try:
        return resolve_run(cfg, stage, cfg.get("inputs", {}).get(stage, "latest"), False)
    except ProtocolError:
        return None


def eval_images(cfg: dict, dataset: str) -> dict[str, tuple[str, set]]:
    runs = {f"x_transfer_{dataset}{t}_merge": "coverage.csv" for t in SELECTORS}
    runs.update({f"b_transfer_{m}_{dataset}_merge": "queries.csv" for m in METHODS})
    runs.update({f"x_giou_{g}_{dataset}_merge": "chains.csv" for g in GIOU})
    found = {}
    for stage, name in runs.items():
        run = latest(cfg, stage)
        if run is not None and (run / name).exists():
            found[stage] = (run.name, set(pd.read_csv(run / name, usecols=["image"]).image))
    return found


def eval_paths(cfg: dict, dataset: str, images: set) -> dict[str, str]:
    d = cfg["data"]
    if dataset == "lvis_hier":
        imgs = lvis_hier.load(assert_real_path(d["lvis_val_json"]))
        return {k: str(Path(d["coco_images"]) / imgs[coco_id(k)].file_name) for k in images}
    if dataset == "pascal_part":
        return {k: str(pascal_part.image_path(d["pascal_voc"], k.split(":", 1)[1])) for k in images}
    if dataset == "openimages_hier":
        return {k: str(Path(d["openimages_images"]) / f"{k.split(':', 1)[1]}.jpg") for k in images}
    return {}


def ade_sha1(path: str, images: set) -> dict[str, str]:
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(assert_real_path(path))
    out = {}
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["image", "filename"])
        for img, fn in zip(t.column("image").to_pylist(), t.column("filename").to_pylist()):
            key = f"ade20k:{fn.rsplit('.', 1)[0]}"
            if key in images:
                with Image.open(io.BytesIO(img["bytes"])) as im:
                    out[key] = array_sha1(np.asarray(im.convert("RGB")))
    return out


def training_images(cfg: dict, manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    tr = manifest[manifest.split_class.isin(TRAIN_CLASSES)].copy()
    tr["tag"] = tr.source + "/" + tr.split
    tr["coco_id"] = pd.to_numeric(tr.coco_id, errors="coerce")
    voc, ann = cfg["data"]["pascal_voc"], cfg["data"]["pascal_part_ann"]
    ids = [i for i in pascal_part.split_ids(voc, "train") if (Path(ann) / f"{i}.mat").exists()]
    paths = [str(pascal_part.image_path(voc, i)) for i in ids]
    pp = pd.DataFrame({"image_key": [f"pascal_part:{i}" for i in ids], "source": "pascal_part", "split": "train",
                       "split_class": "train", "coco_id": np.nan, "sha1": hash_paths(paths, workers),
                       "tag": "pascal_part/train"})
    return pd.concat([tr[["image_key", "source", "split", "split_class", "coco_id", "sha1", "tag"]], pp],
                     ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D7", "D8", "D26"])
    mrun, manifest, mhash, _ = manifest_inputs(cfg, False)
    run = start_run("x_transfer_clean_lvis_hier", cfg, False, {"manifest_run": str(mrun)})
    train = training_images(cfg, manifest, args.workers)
    out = {"label": "transfer:lvis_hier (clean)", "deviation": "D26", "manifest_sha256": mhash,
           "training_images": {"rows": int(len(train)), "coco_ids": int(train.coco_id.nunique()),
                               "sha1": int(train.sha1.nunique()),
                               "by_source": train.groupby("tag").size().to_dict()},
           "overlap": {}, "clean": {}, "reproduces_published": {}}
    drops = {}
    for ds in DATASETS:
        found = eval_images(cfg, ds)
        images = set().union(*(v[1] for v in found.values()))
        if not images:
            out["overlap"][ds] = "not available: no transfer merge run"
            continue
        if ds == "ade20k":
            sha = ade_sha1(cfg["data"]["ade20k_parquet"], images)
        else:
            paths = eval_paths(cfg, ds, images)
            keys = sorted(paths)
            sha = dict(zip(keys, hash_paths([paths[k] for k in keys], args.workers)))
        if set(sha) != images:
            raise ProtocolError(f"{ds}: {len(images - set(sha))} evaluation images could not be hashed")
        ov = overlap(sorted(images), sha, train)
        ov.to_csv(run / f"overlap_{ds}.csv", index=False)
        drops[ds] = set(ov.image[ov.overlap])
        out["overlap"][ds] = {**overlap_summary(ov), "runs": {
            s: {"run": r, "images": len(imgs), "overlap": len(drops[ds] & imgs)} for s, (r, imgs) in found.items()}}
        if ds == "pascal_part":
            val = set(pascal_part.split_ids(cfg["data"]["pascal_voc"], cfg["transfer"]["pascal_part"]["split"]))
            out["overlap"][ds]["voc_split_ids_disjoint"] = not (val & set(pascal_part.split_ids(
                cfg["data"]["pascal_voc"], "train")))
    drop = drops.get("lvis_hier", set())
    bs, n_rand = cfg["v4"]["bootstrap"], len(cfg["v4"]["seeds"])
    strict = None
    for tag in SELECTORS:
        stage = f"x_transfer_lvis_hier{tag}_merge"
        r = latest(cfg, stage)
        if r is None:
            continue
        blk, extra = x_transfer_block(r, drop, n_rand, bs)
        out["clean"][stage] = {"source_run": r.name, **blk}
        out["reproduces_published"][stage] = extra["reproduces_published"]
        if tag == "":
            strict = extra["frames"].get("strict")
            if strict is not None:
                strict.to_csv(run / "chains_strict.csv", index=False)
    for meth in METHODS:
        stage = f"b_transfer_{meth}_lvis_hier_merge"
        r = latest(cfg, stage)
        if r is not None:
            blk, ok = baseline_block(r, drop, strict, bs, run)
            out["clean"][stage] = {"source_run": r.name, **blk}
            out["reproduces_published"][stage] = ok
    for g in GIOU:
        stage = f"x_giou_{g}_lvis_hier_merge"
        r = latest(cfg, stage)
        if r is not None:
            blk, ok = giou_block(r, drop, bs)
            out["clean"][stage] = {"source_run": r.name, **blk}
            out["reproduces_published"][stage] = ok
    write_json(run / "metrics.json", out)
    print(json.dumps({"overlap": {k: v if isinstance(v, str) else {x: v[x] for x in ("eval_images", "overlap")}
                                  for k, v in out["overlap"].items()},
                      "reproduces_published": out["reproduces_published"]}, indent=1))
    print(run)


if __name__ == "__main__":
    main()
