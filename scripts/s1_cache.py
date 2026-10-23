import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import code_version  # noqa: E402
from treeground.data import grefcoco, paco, pascal_part  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.data.paco import CHAIN_TEMPLATE, OBJECT_TEMPLATE, PART_TEMPLATE  # noqa: E402
from treeground.features import image_features  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.run import (REPO, ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# S1 training cache (spec 2.4, S1 at full scale; deviation D11). One shard file per array task holds, per image:
# the selector's node features z_v at layer l (D3 proposals, tau_d dedup, D4 node budget), the containment tree,
# node labels for text-only samples (PACO), and every training query with its <SEL> probe states and step targets
# built by the V2 rules (single-step best-IoU positive + high-overlap ignore; chains: DP path positives).
# Sources: PACO-LVIS train (all manifest images), Pascal-Part train (VOC 2010 train), gRefCOCO train (present
# targets and no-target queries). The manifest's precedence keeps every train image out of dev/eval splits.

SOURCES = ("paco_train", "pascal_part_train", "grefcoco_train")


def load_transfer_helpers():
    spec = importlib.util.spec_from_file_location("x_transfer", REPO / "scripts" / "x_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def items(cfg: dict, source: str, shard: int, num_shards: int):
    # yields (key, image path, list of queries); a query is (qid, kind, text, list of GT RLEs or None if absent)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    if source == "paco_train":
        rows = manifest[(manifest.source == "paco_lvis") & (manifest.split == "train")].drop_duplicates("image_key")
        rows = rows.sort_values("image_key").iloc[shard::num_shards]
        images = paco.load(cfg["data"]["paco_lvis"], "train", {int(k.split(":")[1]) for k in rows.image_key})
        for r in rows.itertuples():
            img = images[int(r.image_key.split(":")[1])]
            gts = paco.gt_rles(img)
            qs = [(q.qid, q.kind, q.text, [gts[a] for a in q.gt]) for q in paco.i1_queries(img)[0]]
            yield r.image_key, r.path, qs, img
    elif source == "pascal_part_train":
        voc = cfg["data"]["pascal_voc"]
        ids = [i for i in pascal_part.split_ids(voc, "train")
               if (Path(cfg["data"]["pascal_part_ann"]) / f"{i}.mat").exists()]
        for i in ids[shard::num_shards]:
            img = pascal_part.load_image(cfg["data"]["pascal_part_ann"], i)
            if img is None:
                continue
            counts: dict[str, int] = {}
            for o in img.objects:
                counts[o.name] = counts.get(o.name, 0) + 1
            qs = []
            for o in img.objects:
                if counts[o.name] != 1:
                    continue
                qs.append((f"{img.key}|object|{o.name}", "object", OBJECT_TEMPLATE.format(object=o.name), [o.rle]))
                for part, rle in o.parts.items():
                    fmt = {"object": o.name, "part": part}
                    qs.append((f"{img.key}|part|{o.name}:{part}", "part", PART_TEMPLATE.format(**fmt), [rle]))
                    qs.append((f"{img.key}|chain|{o.name}:{part}", "chain", CHAIN_TEMPLATE.format(**fmt), [o.rle, rle]))
            yield img.key, str(pascal_part.image_path(voc, i)), qs, None
    elif source == "grefcoco_train":
        refs, gimages = grefcoco.load(cfg["data"]["grefcoco"])
        anns = json.loads((Path(cfg["data"]["grefcoco"]) / "instances.json").read_text())["annotations"]
        by_ann = {a["id"]: a for a in anns}
        keep = set(manifest[(manifest.source == "grefcoco") & (manifest.split == "train")].image_key)
        per_image: dict[int, list] = {}
        for ref in refs:
            if ref["split"] != "train" or f"grefcoco:{ref['image_id']}" not in keep:
                continue
            per_image.setdefault(ref["image_id"], []).append(ref)
        ids = sorted(per_image)[shard::num_shards]
        for iid in ids:
            im = gimages[iid]
            qs = []
            for ref in per_image[iid]:
                absent = grefcoco.is_no_target(ref)
                ann_ids = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
                gts = None if absent else [M.from_annotation(by_ann[a]["segmentation"], im["height"], im["width"])
                                           for a in ann_ids]
                for s in ref["sentences"]:
                    qs.append((f"grefcoco|train|{s['sent_id']}", "absent" if absent else "referring",
                               f"{s['sent']} {mllm.SEL}", gts))
            yield f"grefcoco:{iid}", str(grefcoco.image_path(cfg["data"]["coco_images"], im["file_name"])), qs, None
    else:
        raise ProtocolError(f"unknown S1 source {source}")


def multi_target(ious: np.ndarray) -> S.StepTarget | None:
    # several GT masks (gRefCOCO multi-target): the best node of each GT with IoU >= 0.5 is a positive
    if ious.shape[1] == 0:
        return None
    pos = sorted({int(np.argmax(r)) for r in ious if r.max() >= S.IOU_THR})
    if not pos:
        return None
    high = np.flatnonzero((ious >= S.IOU_THR).any(0))
    return S.StepTarget(np.array(pos), np.setdiff1d(high, pos))


def identity(cfg: dict, source: str, layer: int) -> dict:
    p, m = cfg["proposals"], cfg["models"]
    h = hashlib.sha256()
    for rel in ("scripts/s1_cache.py", "scripts/x_transfer.py", "treeground/data/pascal_part.py",
                "treeground/sampling.py", "treeground/labels.py"):
        h.update((REPO / rel).read_bytes())
    x = load_transfer_helpers()
    return {"kind": "s1_cache", "source": source, "layer": layer, "node_cap": x.node_cap(),
            "sam3": [m["sam3"]["repo"], m["sam3"]["revision"]], "qwen": [m["qwen"]["repo"], m["qwen"]["revision"]],
            "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                            "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
            "templates": [OBJECT_TEMPLATE, PART_TEMPLATE, CHAIN_TEMPLATE],
            "code": [code_version("proposals"), code_version("features"), code_version("queries"), h.hexdigest()]}


def out_dir(cfg: dict, source: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "s1" / source / canonical_hash(ident)[:16]


def chosen_layer() -> int:
    import yaml
    return int(yaml.safe_load((REPO / "configs" / "chosen_layer.yaml").read_text())["layer"])


@torch.no_grad()
def build_record(models, cfg, key, path, queries, paco_img, cap, x, layer):
    sam, sproc, qwen, qproc, sel_id, device = models
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                            p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    image = load_rgb(assert_real_path(path))
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    c = x.cap_nodes(rles, out["boxes"][kept], out["stability"][kept], out["src_index"][kept], cap)
    rc, bc = [rles[i] for i in c], out["boxes"][kept][c]
    counts = {"proposal_failures": 0, "unreachable_chains": 0, "queries": 0, "absent": 0}
    if not rc:
        counts["proposal_failures"] += sum(1 for q in queries if q[3] is not None)
        return None, counts
    tree = T.build_tree(rc, device)
    z = image_features(qwen, qproc, image, rc, bc, [layer], ["hidden"])[f"hidden_L{layer}"].half()
    probes = mllm.probe_states(qwen, qproc, Image.fromarray(image), [q[2] for q in queries], [layer], sel_id,
                               cfg["s1"]["probe_batch"]) if queries else []
    recq, absent = [], []
    for (qid, kind, text, gts), pr in zip(queries, probes):
        pv = torch.from_numpy(np.ascontiguousarray(pr[:, 0])).half()
        if gts is None:
            absent.append({"qid": qid, "probes": pv})
            counts["absent"] += 1
            continue
        ious = M.iou(rc, gts).T
        if kind == "chain":
            path_ = T.max_iou_path(tree, ious)
            if path_ is None:
                counts["unreachable_chains"] += 1
                continue
            steps = S.chain_step_targets(path_, ious)
        elif kind == "referring":
            tgt = multi_target(ious)
            if tgt is None:
                counts["proposal_failures"] += 1
                continue
            steps, path_ = [tgt], None
        else:
            tgt = S.single_step_target(ious[0])
            if tgt is None:
                counts["proposal_failures"] += 1
                continue
            steps, path_ = [tgt], None
        recq.append({"qid": qid, "kind": "object" if kind == "referring" else kind, "probes": pv,
                     "steps": [(s.pos.astype(np.int32), s.ignore.astype(np.int32)) for s in steps],
                     "path": None if path_ is None else tuple(int(v) for v in path_)})
        counts["queries"] += 1
    n = len(rc)
    labels = parts = np.full(n, "", dtype=object)
    high = np.zeros(n, dtype=bool)
    if paco_img is not None:
        lab = paco_part_labels(rc, paco_img)
        labels, parts, high = lab.label, lab.part, lab.high_overlap
    rec = {"key": key, "z": z, "parent": tree.parent.astype(np.int32), "labels": labels, "parts": parts,
           "high": high, "queries": recq, "absent": absent}
    return rec, counts


def run_shard(cfg, source, shard, num_shards):
    require_approved(cfg, ["D3", "D4", "D11"])
    layer = chosen_layer()
    ident = identity(cfg, source, layer)
    d = out_dir(cfg, source, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pt"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"s1_cache_{source}", cfg, False, {"shard": shard, "num_shards": num_shards, "cache": str(d)})
    x = load_transfer_helpers()
    device = cfg["device"]
    m, p = cfg["models"], cfg["proposals"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    qwen, qproc = mllm.load_qwen(m["qwen"]["repo"], m["qwen"]["revision"], device, cfg["features"]["dtype"],
                                 cfg.get("hf_home"))
    sel_id, sel_hash = mllm.add_sel_token(qwen, qproc)
    models = (sam, sproc, qwen, qproc, sel_id, device)
    cap = x.node_cap()
    recs, totals, t0 = [], {}, time.time()
    for i, (key, path, queries, pimg) in enumerate(items(cfg, source, shard, num_shards), 1):
        t1 = time.time()
        rec, counts = build_record(models, cfg, key, path, queries, pimg, cap, x, layer)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        if rec is not None:
            recs.append(rec)
        print(f"[{i}] {key} queries {counts['queries']} absent {counts['absent']} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"records": recs, "counts": totals, "sel_init": sel_hash, "layer": layer}, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(recs), "counts": totals, "seconds": time.time() - t0})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = cfg["s1"]["num_shards"][args.source]
    if not 0 <= args.shard < n:
        raise ProtocolError(f"shard {args.shard} outside 0..{n - 1}")
    run_shard(cfg, args.source, args.shard, n)


if __name__ == "__main__":
    main()
