import argparse
import hashlib
import itertools
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s1_cache as S1C  # noqa: E402
import s2_nodes as S2N  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, node_cap  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, start_run, write_json)

# Node cache for the PACO-LVIS train images outside the 10k-image worklist (D41: the full train split). Same
# proposals, dedup, node budget, tree and step targets as the S2 node caches (s2_nodes.node_set / targets), plus the
# PACO part labels of every node for the text-only items. The manifest already keeps only train images that are
# in no val or test split (pixel-sha1 precedence), and worklist images are skipped here because the proposal
# store serves them.

SOURCE = "paco_train_rest"
FILES = ("scripts/ta_paco_nodes.py", "treeground/labels.py", "treeground/data/paco.py")


def num_shards(cfg: dict, smoke: bool) -> int:
    return 1 if smoke else int(cfg["ta"]["paco_rest_shards"])


def identity(cfg: dict, smoke: bool) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    base = S2N.identity(cfg, SOURCE, smoke)
    return {**base, "kind": "ta_paco_nodes", "code": base["code"] + [h.hexdigest()],
            "smoke_images": cfg["ta"]["paco_rest_smoke_images"] if smoke else None}


def out_dir(cfg: dict, smoke: bool) -> Path:
    return S2N.out_dir(cfg, identity(cfg, smoke))


def rest_items(cfg: dict, shard: int, n: int):
    _, _, _, wl = manifest_inputs(cfg, False)
    worklist = set(wl[wl.role == "paco_train"].image_key)
    for key, path, queries, img in S1C.items(cfg, "paco_train", shard, n):
        if key not in worklist:
            yield key, path, queries, img


def run_shard(cfg: dict, shard: int, smoke: bool) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D3", "D4", "D21", "D41"])
    n = num_shards(cfg, smoke)
    if not 0 <= shard < n:
        raise ProtocolError(f"shard {shard} outside 0..{n - 1}")
    ident = identity(cfg, smoke)
    d = out_dir(cfg, smoke)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    (d / "nodes").mkdir(exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pt"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"ta_nodes_{SOURCE}", cfg, smoke, {"shard": shard, "num_shards": n, "cache": str(d)})
    device = cfg["device"]
    m, p = cfg["models"], cfg["proposals"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    cap = node_cap(cfg, smoke)
    rows = rest_items(cfg, shard, n)
    if smoke:
        rows = itertools.islice(rows, cfg["ta"]["paco_rest_smoke_images"])
    recs, totals, t0 = [], Counter(), time.time()
    for i, (key, path, queries, img) in enumerate(rows, 1):
        t1 = time.time()
        image = load_rgb(assert_real_path(path))
        if image.shape[:2] != (img.h, img.w):
            raise ProtocolError(f"{key}: image size differs from the PACO annotation")
        rles = S2N.node_set(sam, sproc, cfg, image, cap)
        rec, counts = S2N.record(d, key, path, image, rles, queries, device)
        totals.update(counts)
        if rec is not None:
            lab = paco_part_labels(rles, img)
            rec.update(labels=lab.label.tolist(), parts=lab.part.tolist(), high=lab.high_overlap.astype(bool))
            recs.append(rec)
        print(f"[{i}] {key} queries {counts['queries']} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"records": recs, "counts": dict(totals)}, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(recs), "counts": dict(totals), "seconds": time.time() - t0,
                                      "deviations": ["D21", "D41"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_shard(load_config(args.config), args.shard, args.smoke)


if __name__ == "__main__":
    main()
