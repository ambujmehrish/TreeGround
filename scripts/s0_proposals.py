import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import proposals as P  # noqa: E402
from treeground.cache import cache_dir, keep_column, mark_complete, write_identity  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, proposal_identity  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, ensure_deviation_log, load_config,  # noqa: E402
                            require_approved, start_run, update_meta, write_json)


def shard_items(wl: pd.DataFrame, shard: int, num_shards: int) -> pd.DataFrame:
    unique = wl.drop_duplicates("sha1").sort_values("sha1").reset_index(drop=True)
    return unique.iloc[shard::num_shards]


def process_image(model, processor, row, pcfg: P.ProposalConfig, taus: list[float]) -> pd.DataFrame:
    image = load_rgb(assert_real_path(row.path))
    if image.shape[:2] != (row.h, row.w):
        raise ProtocolError(f"{row.path}: decoded size {image.shape[:2]} differs from manifest {(row.h, row.w)}")
    out = P.generate(model, processor, image, pcfg)
    df = pd.DataFrame({
        "sha1": row.sha1, "src_index": out["src_index"], "rle": [r["counts"] for r in out["rles"]],
        "h": row.h, "w": row.w, "area": out["areas"], "box": list(out["boxes"]),
        "sam_iou": out["sam_iou"], "stability": out["stability"],
    })
    for t in taus:
        df[keep_column(t)] = P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], t)
    return df


def run_shard(cfg: dict, smoke: bool, shard: int, num_shards: int) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D1", "D3"])
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    ident = proposal_identity(cfg, smoke, mhash)
    out_dir = cache_dir(cfg, "proposals", ident)
    write_identity(out_dir, ident)
    target = out_dir / "shards" / f"{shard:05d}.parquet"
    run = start_run("s0_proposals", cfg, smoke, {"shard": shard, "num_shards": num_shards, "manifest_run": str(mrun),
                                                  "manifest_sha256": mhash, "cache": str(out_dir),
                                                  "route": P.ROUTE})
    items = shard_items(wl, shard, num_shards)
    if target.exists():
        write_json(run / "metrics.json", {"skipped": "shard already cached", "images": len(items)})
        return
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                            p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"],
                            p["dtype"])
    sam = cfg["models"]["sam3"]
    print(f"shard {shard}/{num_shards}: {len(items)} images -> {target}", flush=True)
    model, processor = P.load_sam3(sam["repo"], sam["revision"], cfg["device"], p["dtype"], cfg.get("hf_home"))
    print(f"loaded {type(model).__name__} on {cfg['device']}", flush=True)
    update_meta(run, sam3_class=type(model).__name__, sam3_processor=type(processor).__name__)
    frames, t0 = [], time.time()
    for i, row in enumerate(items.itertuples(), 1):
        t1 = time.time()
        df = process_image(model, processor, row, pcfg, p["dedup_thresholds"])
        frames.append(df)
        kept = " ".join(f"{keep_column(t)}={int(df[keep_column(t)].sum())}" for t in p["dedup_thresholds"])
        print(f"[{i}/{len(items)}] {row.image_key} "
              f"{row.w}x{row.h} raw={len(df)} {kept} {time.time() - t1:.1f}s", flush=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)
    per_image = df.groupby("sha1").size() if len(df) else pd.Series(dtype=int)
    write_json(run / "metrics.json", {
        "images": len(items), "seconds": time.time() - t0,
        "raw_proposals_mean": float(per_image.mean()) if len(per_image) else 0.0,
        **{f"nodes_mean_{keep_column(t)}": float(df.groupby("sha1")[keep_column(t)].sum().mean()) if len(df) else 0.0
           for t in p["dedup_thresholds"]},
    })


def merge(cfg: dict, smoke: bool, num_shards: int) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    ident = proposal_identity(cfg, smoke, mhash)
    out_dir = cache_dir(cfg, "proposals", ident)
    run = start_run("s0_proposals_merge", cfg, smoke, {"manifest_run": str(mrun), "cache": str(out_dir)})
    print(f"merging {num_shards} proposal shards in {out_dir}", flush=True)
    missing = [k for k in range(num_shards) if not (out_dir / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} proposal shards missing, e.g. {missing[:5]}")
    rows = []
    taus = cfg["proposals"]["dedup_thresholds"]
    for k in range(num_shards):
        name = f"{k:05d}.parquet"
        df = pd.read_parquet(out_dir / "shards" / name)
        counts = {c: df.groupby("sha1")[c].sum() if len(df) else pd.Series(dtype=int)
                  for c in [keep_column(t) for t in taus]}
        raw = df.groupby("sha1").size() if len(df) else pd.Series(dtype=int)
        for item in shard_items(wl, k, num_shards).itertuples():
            rows.append({"sha1": item.sha1, "shard": name, "h": item.h, "w": item.w,
                         "n_raw": int(raw.get(item.sha1, 0)),
                         **{c: int(v.get(item.sha1, 0)) for c, v in counts.items()}})
    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / "index.parquet", index=False)
    summary = {"images": len(index), "images_without_proposals": int((index["n_raw"] == 0).sum()),
               "n_raw": index["n_raw"].describe().to_dict(),
               **{c: index[c].describe().to_dict() for c in [keep_column(t) for t in taus]}}
    mark_complete(out_dir, summary)
    write_json(run / "metrics.json", summary)
    print(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = args.num_shards or (1 if args.smoke else cfg["proposals"]["num_shards"])
    if args.merge:
        merge(cfg, args.smoke, num_shards)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.smoke, args.shard, num_shards)


if __name__ == "__main__":
    main()
