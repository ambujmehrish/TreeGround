import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.data import grefcoco  # noqa: E402
from treeground.data.images import pixel_sha1  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# Read-only train/test leakage audit (D40 audit of 29 Oct; changes no cache and no training input).
# Train = every image of the S5 training pools (the superset of S2-long / S3 / S4 / S5 / specialists / Qwen-box SFT
# images: PACO-LVIS train, Pascal-Part train, gRefCOCO train, PartImageNet_Seg train), read from the pool records.
# Test = PACO-LVIS test, gRefCOCO testA / testB, PARTONOMY-Core test, InstructPart test, PascalPart-116,
# PartImageNet_OOD test. Checks: (1) exact pixel sha1 overlap; (2) COCO id overlap between manifest train and eval
# rows; (3) near duplicates by a 64-bit difference hash (Hamming distance <= max_hamming), listed for inspection.


def dhash(path: str) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def signature(path: str) -> tuple[str, str, int]:
    return path, pixel_sha1(path)[0], dhash(path)


def train_images(cfg: dict) -> dict[str, str]:
    import s5_train as S5T  # noqa: F401  (S5 sources)
    import s2_train as S2T
    out = {}
    for source in S2T.source_weights(cfg["s2"]):
        _, files = S2T.node_cache(cfg, source, False)
        for f in files:
            for r in torch.load(f, map_location="cpu", weights_only=False)["records"]:
                out[r["path"]] = source
    return out


def test_images(cfg: dict) -> dict[str, str]:
    import gres_eval as G
    import ophrg_eval as O
    import paco_test_eval as PT
    import partonomy_eval as PE
    out = {}
    for r in PT.test_images(cfg).itertuples():
        out[r.path] = "paco_test"
    for sp in ("testA", "testB"):
        df, _ = G.expressions(cfg, sp)
        for f in df.file.unique():
            out[str(grefcoco.image_path(cfg["data"]["coco_images"], f))] = f"grefcoco_{sp}"
    root = Path(cfg["data"]["partonomy"])
    for r in PE.questions(root):
        out[str(root / r["image_path"])] = "partonomy_test"
    for bench in ("instructpart", "pascalpart", "partimagenet"):
        for _, path, _ in O.LOADERS[bench](cfg):
            if Path(path).exists():
                out[path] = f"ophrg_{bench}"
    return out


def coco_overlap(cfg: dict) -> dict:
    _, manifest, _, _ = manifest_inputs(cfg, False)
    if "coco_id" not in manifest:
        return {"coco_id": "column absent"}
    m = manifest[manifest.coco_id.notna()]
    tr = set(m[m.split_class == "train"].coco_id)
    ev = set(m[m.split_class.isin(["eval", "dev", "heldout"])].coco_id)
    return {"train_coco_ids": len(tr), "eval_coco_ids": len(ev), "overlap": len(tr & ev),
            "examples": sorted(map(str, tr & ev))[:20]}


def near(train: list, test: list, max_h: int) -> list:
    a = np.array([h for _, _, h in train], dtype=np.uint64)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = []
    for path, _, h in test:
        x = np.bitwise_xor(a, np.uint64(h))
        d = sum(pop[((x >> np.uint64(8 * k)) & np.uint64(255)).astype(np.int64)].astype(np.int64) for k in range(8))
        for j in np.flatnonzero(d <= max_h):
            out.append({"test": path, "train": train[j][0], "hamming": int(d[j])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_hamming", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run("leak_audit", cfg, False, {})
    tr, te = train_images(cfg), test_images(cfg)
    if not tr or not te:
        raise ProtocolError(f"empty image sets: train {len(tr)} test {len(te)}")
    with ProcessPoolExecutor(args.workers) as ex:
        trs = list(ex.map(signature, sorted(tr), chunksize=64))
        tes = list(ex.map(signature, sorted(te), chunksize=64))
    by_sha = defaultdict(list)
    for p, s, _ in trs:
        by_sha[s].append(p)
    exact = [{"test": p, "set": te[p], "train": by_sha[s], "train_sources": sorted({tr[t] for t in by_sha[s]})}
             for p, s, _ in tes if s in by_sha]
    nd = near(trs, tes, args.max_hamming)
    for r in nd:
        r["set"], r["train_source"] = te[r["test"]], tr[r["train"]]
    per_set = defaultdict(lambda: {"images": 0, "exact": 0, "near": 0})
    for p in te:
        per_set[te[p]]["images"] += 1
    for r in exact:
        per_set[r["set"]]["exact"] += 1
    for t in {r["test"] for r in nd}:
        per_set[te[t]]["near"] += 1
    res = {"train_images": len(tr), "test_images": len(te), "per_test_set": dict(per_set),
           "train_by_source": {s: sum(1 for v in tr.values() if v == s) for s in set(tr.values())},
           "exact_overlap": exact, "near_duplicates": nd, "max_hamming": args.max_hamming,
           "coco_id": coco_overlap(cfg)}
    write_json(run / "metrics.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("exact_overlap", "near_duplicates")}, indent=1))
    print(run)


if __name__ == "__main__":
    main()
