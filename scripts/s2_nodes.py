import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s1_cache as S1C  # noqa: E402
import x_transfer as X  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import code_version  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, node_cap  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, start_run, write_json)

# S2 node cache for the extra training sources (deviation D21): per image the D3 proposals after tau_d dedup and
# the D4 node budget (as RLEs, read by the S2 data loader for the coverage maps), the containment tree and the
# step targets of every query by the S1 rules (s1_cache.items: same images, queries and GT masks). Features are
# not cached: S2 recomputes them with the LoRA model at every step.

SOURCES = ("pascal_part_train", "grefcoco_train")
FILES = ("scripts/s2_nodes.py", "scripts/s1_cache.py", "scripts/x_transfer.py", "treeground/data/pascal_part.py",
         "treeground/data/grefcoco.py", "treeground/sampling.py", "treeground/tree.py")


def num_shards(cfg: dict, source: str, smoke: bool) -> int:
    return 1 if smoke else int(cfg["s2_nodes"]["num_shards"][source])


def identity(cfg: dict, source: str, smoke: bool) -> dict:
    p, m = cfg["proposals"], cfg["models"]
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    cap = node_cap(cfg, smoke)
    return {"kind": "s2_nodes", "source": source, "smoke": smoke, "manifest": manifest_inputs(cfg, False)[2],
            "smoke_images": cfg["s2_nodes"]["smoke_images"] if smoke else None,
            "sam3": [m["sam3"]["repo"], m["sam3"]["revision"]],
            "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                            "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
            "node_cap": cap and {"k": cap["k"], "order": cap["order"]},
            "code": [code_version("proposals"), h.hexdigest()]}


def out_dir(cfg: dict, ident: dict) -> Path:
    sub = "smoke" if ident["smoke"] else "full"
    return Path(cfg["cache_root"]) / sub / "s2_nodes" / ident["source"] / canonical_hash(ident)[:16]


def targets(rles: list, tree: T.Tree, queries: list) -> tuple[list[dict], list[dict], Counter]:
    # s1_cache.build_record rules: chains by the max-IoU tree path, referring (multi-target) by the best node of
    # every GT, object/part by the single best node; queries without GT are no-target examples
    present, absent, counts = [], [], Counter()
    for qid, kind, text, gts in queries:
        if gts is None:
            absent.append({"qid": qid, "text": text})
            counts["absent"] += 1
            continue
        ious = M.iou(rles, gts).T
        path = None
        if kind == "chain":
            path = T.max_iou_path(tree, ious)
            if path is None:
                counts["unreachable_chains"] += 1
                continue
            steps = S.chain_step_targets(path, ious)
        else:
            tgt = S1C.multi_target(ious) if kind == "referring" else S.single_step_target(ious[0])
            if tgt is None:
                counts["proposal_failures"] += 1
                continue
            steps = [tgt]
        present.append({"qid": qid, "kind": kind, "text": text,
                        "steps": [(s.pos.astype(np.int32), s.ignore.astype(np.int32)) for s in steps],
                        "path": None if path is None else tuple(int(v) for v in path)})
        counts["queries"] += 1
    return present, absent, counts


def node_set(sam, sproc, cfg: dict, image: np.ndarray, cap: dict | None) -> list:
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                            p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    if cap is None:
        return rles
    c = X.cap_nodes(rles, out["boxes"][kept], out["stability"][kept], out["src_index"][kept], cap)
    return [rles[i] for i in c]


def record(d: Path, key: str, path: str, image: np.ndarray, rles: list, queries: list, device: str
           ) -> tuple[dict | None, Counter]:
    if not rles:
        return None, Counter(proposal_failures=sum(1 for q in queries if q[3] is not None))
    tree = T.build_tree(rles, device)
    present, absent, counts = targets(rles, tree, queries)
    h, w = image.shape[:2]
    rel = f"nodes/{key.replace(':', '_')}.json"
    (d / rel).write_text(json.dumps({"h": h, "w": w, "counts": [r["counts"] for r in rles]}))
    return {"key": key, "path": path, "h": h, "w": w, "nodes": rel, "parent": tree.parent.astype(np.int32),
            "queries": present, "absent": absent}, counts


def run_shard(cfg: dict, source: str, shard: int, smoke: bool) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D3", "D4", "D21"])
    n = num_shards(cfg, source, smoke)
    if not 0 <= shard < n:
        raise ProtocolError(f"shard {shard} outside 0..{n - 1}")
    ident = identity(cfg, source, smoke)
    d = out_dir(cfg, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    (d / "nodes").mkdir(exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pt"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"s2_nodes_{source}", cfg, smoke, {"shard": shard, "num_shards": n, "cache": str(d)})
    device = cfg["device"]
    m, p = cfg["models"], cfg["proposals"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    cap = node_cap(cfg, smoke)
    rows = S1C.items(cfg, source, shard, n)
    if smoke:
        rows = itertools.islice(rows, cfg["s2_nodes"]["smoke_images"])
    recs, totals, t0 = [], Counter(), time.time()
    for i, (key, path, queries, _) in enumerate(rows, 1):
        t1 = time.time()
        image = load_rgb(assert_real_path(path))
        rec, counts = record(d, key, path, image, node_set(sam, sproc, cfg, image, cap), queries, device)
        totals.update(counts)
        if rec is not None:
            recs.append(rec)
        print(f"[{i}] {key} queries {counts['queries']} absent {counts['absent']} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"records": recs, "counts": dict(totals)}, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(recs), "counts": dict(totals), "seconds": time.time() - t0,
                                      "deviations": ["D21"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_shard(load_config(args.config), args.source, args.shard, args.smoke)


if __name__ == "__main__":
    main()
