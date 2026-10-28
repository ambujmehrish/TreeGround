import argparse
import hashlib
import itertools
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
import s2_nodes as S2N  # noqa: E402
import s3_pool as S3P  # noqa: E402
import s5_common as S5C  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data import partimagenet  # noqa: E402
from treeground.data.images import load_rgb, pixel_sha1  # noqa: E402
from treeground.data.paco import CHAIN_TEMPLATE, OBJECT_TEMPLATE, PART_TEMPLATE  # noqa: E402
from treeground.identity import manifest_inputs, node_cap  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, start_run, write_json)

# Deviation D36 training pools: per training image the D36 pool (s5_common: capped nodes + top-3 SAM 3 text
# instances + the Qwen3-VL-4B box + SAM 3 mask of every phrase of the image's queries; s3_pool.augment order, cap and
# dedup), the containment tree and step targets (s2_nodes.record), PACO part labels, and per query and step the IoU
# of every pool entry with the step's GT (the max over GTs for multi-target expressions), used by the S5 listwise
# term. Sources: the S3 sources plus PartImageNet_Seg train (Partonomy-PartImageNet, the PARTONOMY training source
# we did not use): object / part / chain queries with the PACO templates; an image whose pixels equal any eval or
# dev image of the manifest is skipped and counted.

SOURCES = S3P.SOURCES + ("partimagenet_train",)
FILES = ("scripts/s5_pool.py", "scripts/s5_common.py", "scripts/s3_pool.py", "scripts/b_baselines.py",
         "scripts/s2_nodes.py", "treeground/data/partimagenet.py")


def settings(cfg: dict) -> dict:
    return cfg["s5_pool"]


def num_shards(cfg: dict, source: str, smoke: bool) -> int:
    return 1 if smoke else int(settings(cfg)["num_shards"][source])


def identity(cfg: dict, source: str, smoke: bool) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    base = S2N.identity(cfg, source, smoke)
    base.pop("smoke_images", None)
    s = cfg["s3_pool"]
    return {"kind": "s5_pool", "source": source, "smoke": smoke, "nodes": base,
            "smoke_images": settings(cfg)["smoke_images"] if smoke else None, "sam3_text": cfg["baselines"]["sam3_text"],
            "qwen_box_sam": cfg["baselines"]["qwen_box_sam"], "qwen": cfg["models"]["qwen"],
            "pool": {k: s[k] for k in ("instances_per_phrase", "max_instances", "dup_iou")},
            "num_shards": num_shards(cfg, source, smoke), "code": h.hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    sub = "smoke" if ident["smoke"] else "full"
    return Path(cfg["cache_root"]) / sub / "s5_pool" / ident["source"] / canonical_hash(ident)[:16]


def partimagenet_items(cfg: dict, shard: int, n: int, counts: Counter):
    _, manifest, _, _ = manifest_inputs(cfg, False)
    held = set(manifest[manifest.split_class.isin(["eval", "dev", "heldout"])].sha1)
    for r in partimagenet.load(cfg["data"]["partimagenet"], "train")[shard::n]:
        if pixel_sha1(r["path"])[0] in held:
            counts["skipped_equal_to_eval_image"] += 1
            continue
        o, parts = r["object"], r["parts"]
        obj = partimagenet.union(list(parts.values())) if len(parts) > 1 else next(iter(parts.values()))
        qs = [(f"{r['key']}|object|{o}", "object", OBJECT_TEMPLATE.format(object=o), [obj])]
        for p, rle in sorted(parts.items()):
            fmt = {"object": o, "part": p}
            qs.append((f"{r['key']}|part|{o}:{p}", "part", PART_TEMPLATE.format(**fmt), [rle]))
            qs.append((f"{r['key']}|chain|{o}:{p}", "chain", CHAIN_TEMPLATE.format(**fmt), [obj, rle]))
        yield r["key"], r["path"], qs, None


def items(cfg: dict, source: str, shard: int, n: int, counts: Counter):
    if source == "partimagenet_train":
        return partimagenet_items(cfg, shard, n, counts)
    rows = S1C.items(cfg, source, shard, n)
    if source == "paco_train":
        wl = manifest_inputs(cfg, False)[3]
        keep = set(wl[wl.role == "paco_train"].image_key)
        rows = (r for r in rows if r[0] in keep)
    return rows


def step_ious(pool: list, q: dict, gts: list) -> list[np.ndarray]:
    ious = M.iou(pool, gts)                       # (pool, gts)
    if q["kind"] == "chain":
        return [ious[:, t].astype(np.float16) for t in range(len(gts))]
    if q["kind"] == "referring":
        return [ious.max(1).astype(np.float16)]
    return [ious[:, 0].astype(np.float16)]


def run_shard(cfg: dict, source: str, shard: int, smoke: bool) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D3", "D4", "D21", "D30", "D36"])
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
    run = start_run(f"s5_pool_{source}", cfg, smoke, {"shard": shard, "num_shards": n, "cache": str(d)})
    device = cfg["device"]
    m, p, s = cfg["models"], cfg["proposals"], cfg["s3_pool"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    text = S5C.TextPlusBox(cfg)
    cap = node_cap(cfg, smoke)
    recs, totals, t0 = [], Counter(), time.time()
    rows = items(cfg, source, shard, n, totals)
    if smoke:
        rows = itertools.islice(rows, settings(cfg)["smoke_images"])
    for i, (key, path, queries, img) in enumerate(rows, 1):
        t1 = time.time()
        image = load_rgb(assert_real_path(path))
        nodes = S2N.node_set(sam, sproc, cfg, image, cap)
        plist = list(dict.fromkeys(ph for _, _, t, _ in queries for ph in S3P.phrases(t)))
        pool, imeta = S3P.augment(nodes, text(image, plist, s["instances_per_phrase"]), s["max_instances"], s["dup_iou"])
        rec, counts = S2N.record(d, key, path, image, pool, queries, device)
        totals.update(counts)
        totals["instances"] += len(imeta)
        totals["box_instances"] += sum(1 for x in imeta if x["score"] == S5C.BOX_SCORE)
        if rec is not None:
            gts = {qid: g for qid, _, _, g in queries}
            for q in rec["queries"]:
                q["ious"] = step_ious(pool, q, gts[q["qid"]])
            rec.update(n_nodes=len(nodes), phrases=plist, instances=imeta)
            if img is not None:
                lab = paco_part_labels(pool, img)
                rec.update(labels=list(lab.label), parts=list(lab.part), high=lab.high_overlap.tolist())
            recs.append(rec)
        print(f"[{i}] {key} nodes {len(nodes)} instances {len(imeta)} queries {counts['queries']} "
              f"absent {counts['absent']} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"records": recs, "counts": dict(totals)}, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(recs), "counts": dict(totals), "seconds": time.time() - t0,
                                      "deviations": ["D21", "D30", "D36"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with torch.no_grad():
        run_shard(load_config(args.config), args.source, args.shard, args.smoke)


if __name__ == "__main__":
    main()
