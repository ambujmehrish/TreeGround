import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import code_version  # noqa: E402
from treeground.data import ade20k, lvis_hier, openimages_hier, pascal_part  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.data.paco import CHAIN_TEMPLATE  # noqa: E402
from treeground.features import image_features  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import (REPO, ProtocolError, assert_real_path, canonical_hash, load_config, read_deviations, read_json,  # noqa: E402
                            require_approved, resolve_run, start_run, write_json)
from treeground.selector import Selector  # noqa: E402

# Transfer evaluation (deviation D7): the PACO-trained V2 selector on other part datasets, no retraining.
# One shard runs, per image: SAM 3 proposals (same settings as S0), tau_d dedup, the D4 node budget, Qwen node
# features at the chosen layer, Qwen <SEL> probe states for the dataset's object->part chains, and the V4 chain
# search (spec, D5 descendant-or-self, D6 box tree). Coverage is also recorded on the uncapped node set.
# --merge pools the shards and writes the summaries. Every number is a transfer result, labelled per dataset.

THR = 0.5
OPENIMAGES_MAX_SIDE = 1024  # the download bucket serves originals (up to several thousand px); resized like the
#                             common 1024-px OpenImages release. GT boxes are normalised, so they scale exactly.
DATASETS = ("pascal_part", "ade20k", "lvis_hier", "openimages_hier")


def load_v4():
    spec = importlib.util.spec_from_file_location("v4_tree_value", REPO / "scripts" / "v4_tree_value.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records(cfg: dict, dataset: str, shard: int, num_shards: int):
    t = cfg["transfer"][dataset]
    if dataset == "pascal_part":
        voc = cfg["data"]["pascal_voc"]
        ids = [i for i in pascal_part.split_ids(voc, t["split"])
               if (Path(cfg["data"]["pascal_part_ann"]) / f"{i}.mat").exists()]
        for i in ids[shard::num_shards]:
            img = pascal_part.load_image(cfg["data"]["pascal_part_ann"], i)
            if img is None:
                continue
            image = load_rgb(assert_real_path(pascal_part.image_path(voc, i)))
            parts = [(f"{o.name}:{p}", r) for o in img.objects for p, r in o.parts.items()]
            yield img.key, image, (img.h, img.w), pascal_part.chains(img), parts
    elif dataset == "ade20k":
        path = assert_real_path(cfg["data"]["ade20k_parquet"])
        n = sum(k for _, k in ade20k.row_groups(path))
        for _, image, img in ade20k.iter_rows(path, set(range(shard, n, num_shards))):
            yield img.key, image, (img.h, img.w), ade20k.chains(img), ade20k.all_parts(img)
    elif dataset == "lvis_hier":
        imgs = lvis_hier.load(assert_real_path(cfg["data"]["lvis_val_json"]))
        pairs = lvis_hier.class_pairs(imgs)
        _, _, _, wl = manifest_inputs(cfg, False)
        used = {int(k.split(":")[1]) for k in wl.image_key if k.split(":")[0] in ("paco_lvis", "grefcoco")}
        ids = [i for i in sorted(imgs) if i not in used and lvis_hier.chains(imgs[i], pairs)]
        for i in ids[shard::num_shards]:
            img = imgs[i]
            image = load_rgb(assert_real_path(Path(cfg["data"]["coco_images"]) / img.file_name))
            ch = lvis_hier.chains(img, pairs)
            yield img.key, image, (img.h, img.w), ch, [(c[0].split("|")[-1], c[4]) for c in ch]
    elif dataset == "openimages_hier":
        names = openimages_hier.class_names(assert_real_path(cfg["data"]["openimages_classes"]))
        boxes = openimages_hier.load_boxes(assert_real_path(cfg["data"]["openimages_boxes"]), names)
        pairs = openimages_hier.class_pairs(boxes)
        ids = [i for i in sorted(boxes) if openimages_hier.chains(i, boxes[i], pairs, 1, 1)]
        for i in ids[shard::num_shards]:
            image = load_rgb(assert_real_path(Path(cfg["data"]["openimages_images"]) / f"{i}.jpg"))
            h, w = image.shape[:2]
            if max(h, w) > OPENIMAGES_MAX_SIDE:
                s = OPENIMAGES_MAX_SIDE / max(h, w)
                image = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]
            ch = openimages_hier.chains(i, boxes[i], pairs, w, h)
            yield f"openimages:{i}", image, (h, w), ch, [(c[0].split("|")[-1], c[4]) for c in ch]
    else:
        raise ProtocolError(f"unknown transfer dataset {dataset}")


def selector_run(cfg: dict) -> Path:
    stage = cfg.get("inputs", {}).get("selector_stage", "v2")
    return resolve_run(cfg, stage, cfg.get("inputs", {}).get(stage, "latest"), False)


def node_cap() -> dict:
    doc = yaml.safe_load((REPO / "configs" / "node_cap.yaml").read_text())
    return {"k": int(doc["k"]), "order": str(doc["order"])}


def identity(cfg: dict, dataset: str, v2: Path) -> dict:
    p, m = cfg["proposals"], cfg["models"]
    files = ["scripts/x_transfer.py", "treeground/data/pascal_part.py", "treeground/data/ade20k.py",
             "treeground/data/lvis_hier.py", "treeground/data/openimages_hier.py"]
    h = hashlib.sha256()
    for rel in files:
        h.update((REPO / rel).read_bytes())
    return {"kind": "transfer", "dataset": dataset, "settings": cfg["transfer"][dataset],
            "sam3": [m["sam3"]["repo"], m["sam3"]["revision"]], "qwen": [m["qwen"]["repo"], m["qwen"]["revision"]],
            "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                            "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
            "node_cap": node_cap(), "v2_run": v2.name, "selector_stage": cfg.get("inputs", {}).get("selector_stage", "v2"),
            "templates": CHAIN_TEMPLATE,
            "code": [code_version("proposals"), code_version("features"), h.hexdigest()]}


def out_dir(cfg: dict, dataset: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "transfer" / dataset / canonical_hash(ident)[:16]


def cap_nodes(rles, boxes, stability, src, cap):
    n = len(rles)
    if n <= cap["k"]:
        return np.arange(n)
    if cap["order"] != "stability":
        raise ProtocolError(f"transfer supports the stability node budget only, got {cap['order']}")
    keep = np.lexsort((src, -stability))[:cap["k"]]
    return np.sort(keep)


def gt_ious(gts, rles, boxes) -> np.ndarray:
    # IoU of every GT (rows) with every node (columns); boxes (x0, y0, x1, y1 arrays) are box-matched (I10)
    if not rles:
        return np.zeros((len(gts), 0))
    if isinstance(gts[0], np.ndarray):
        return M.box_iou(np.stack(gts), boxes)
    return M.iou(rles, gts).T


def gt_area(g) -> float:
    if isinstance(g, np.ndarray):
        return float(max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1]))
    return float(M.areas([g])[0])


@torch.no_grad()
def process_image(models, cfg, key, image, hw, chains, parts, cap, V4):
    sam, sproc, qwen, qproc, sel, sel_id, layer, device = models
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                            p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    if image.shape[:2] != hw:
        raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {hw}")
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    boxes, stab, src = out["boxes"][kept], out["stability"][kept], out["src_index"][kept]
    cov = []
    if parts:
        pi = gt_ious([r for _, r in parts], rles, boxes)
        best = pi.max(1) if pi.shape[1] else np.zeros(len(parts))
        cov += [{"image": key, "kind": "part", "label": lab, "area_bin": area_bin(gt_area(r)), "best_iou": float(b)}
                for (lab, r), b in zip(parts, best)]
    full_tree = T.build_tree(rles, device) if rles else None
    c_idx = cap_nodes(rles, boxes, stab, src, cap)
    rc = [rles[i] for i in c_idx]
    bc = boxes[c_idx]
    rows = []
    if not chains:
        return cov, rows, len(rles), len(rc)
    tree = T.build_tree(rc, device) if rc else T.from_parents(np.array([-1]))
    rand = [T.randomised_tree(tree, V4.tree_seed(s, key)) for s in cfg["v4"]["seeds"]]
    box_tree = T.build_box_tree(bc) if len(rc) else tree
    if rc:
        feats = image_features(qwen, qproc, image, rc, bc, [layer], ["hidden"])
        e = sel.nodes(feats[f"hidden_L{layer}"].to(device))
    texts = [CHAIN_TEMPLATE.format(object=obj, part=part) for _, obj, part, _, _ in chains]
    probes = mllm.probe_states(qwen, qproc, Image.fromarray(image), texts, [layer], sel_id,
                               cfg["transfer"]["probe_batch"])
    items = []
    for (qid, obj, part, orle, prle), pr in zip(chains, probes):
        ious = gt_ious([orle, prle], rc, bc)
        full = gt_ious([orle, prle], rles, boxes)
        ga = np.array([gt_area(orle), gt_area(prle)])
        if rc:
            qv = sel.queries(torch.from_numpy(np.ascontiguousarray(pr[:, 0])).to(device))
            sig = torch.sigmoid(sel.scores(qv, e)).cpu().numpy()
        else:
            sig = np.zeros((2, 0))
        cov.append({"image": key, "kind": "chain", "label": qid.split("|")[-1], "area_bin": area_bin(ga[1]),
                    "best_iou": float(full[1].max()) if full.shape[1] else 0.0,
                    "object_iou": float(full[0].max()) if full.shape[1] else 0.0,
                    "feasible_uncapped": bool(full.shape[1] and T.max_iou_path(full_tree, full, THR) is not None),
                    "part_ratio": float(ga[1] / max(1, ga[0]))})
        items.append({"qid": qid, "image": key, "sha1": key, "tree": tree, "rand": rand, "box": box_tree,
                      "ious": ious, "sig": sig, "part_ratio": float(ga[1] / max(1, ga[0]))})
    return cov, items, len(rles), len(rc)


def load_models(cfg: dict, v2: Path):
    device = cfg["device"]
    ckpt = torch.load(v2 / "selector.pt", map_location="cpu")
    if ckpt["smoke"]:
        raise ProtocolError("transfer needs a full (non-smoke) V2 selector")
    sel = Selector(ckpt["d_node"], ckpt["d_query"], ckpt["temperature"])
    sel.load_state_dict(ckpt["state_dict"])
    sel = sel.to(device).eval()
    layer = int(ckpt["layer"])
    tau = float(read_json(v2 / "metrics.json")["tau_rej"])
    p, m = cfg["proposals"], cfg["models"]
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    q = m["qwen"]
    qwen, qproc = mllm.load_qwen(q["repo"], q["revision"], device, cfg["features"]["dtype"], cfg.get("hf_home"))
    sel_id, _ = mllm.add_sel_token(qwen, qproc)
    return (sam, sproc, qwen, qproc, sel, sel_id, layer, device), tau, layer


def run_shard(cfg, dataset, shard, num_shards):
    require_approved(cfg, ["D3", "D4", "D7"] + (["D8"] if dataset in ("lvis_hier", "openimages_hier") else [])
                     + (["D9"] if dataset == "openimages_hier" else []))
    v2 = selector_run(cfg)
    ident = identity(cfg, dataset, v2)
    d = out_dir(cfg, dataset, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pkl"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"x_transfer_{dataset}", cfg, False, {"shard": shard, "num_shards": num_shards, "cache": str(d),
                                                          "v2_run": str(v2)})
    V4 = load_v4()
    models, tau, layer = load_models(cfg, v2)
    cap = node_cap()
    cov_all, nat = [], {"strict": [], "d5": [], "d6": [], "no_rejection": []}
    t0, n_img = time.time(), 0
    for key, image, hw, chains, parts in records(cfg, dataset, shard, num_shards):
        t1 = time.time()
        cov, items, n_nodes, n_cap = process_image(models, cfg, key, image, hw, chains, parts, cap, V4)
        cov_all += cov
        if items:
            nat["strict"].append(V4.natural(items, tau)[0])
            nat["d5"].append(V4.natural(items, tau, inclusive=True)[0])
            nat["d6"].append(V4.natural(items, tau, box=True)[0])
            nat["no_rejection"].append(V4.natural(items, -np.inf)[0])
        n_img += 1
        print(f"[{n_img}] {key} nodes {n_nodes} capped {n_cap} chains {len(chains)} {time.time() - t1:.1f}s",
              flush=True)
    res = {"coverage": pd.DataFrame(cov_all),
           **{k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in nat.items()},
           "tau": tau, "layer": layer, "images": n_img}
    tmp = target.with_suffix(".tmp")
    pd.to_pickle(res, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": n_img, "seconds": time.time() - t0})


def merge(cfg, dataset, num_shards):
    v2 = selector_run(cfg)
    ident = identity(cfg, dataset, v2)
    d = out_dir(cfg, dataset, ident)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.pkl").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} transfer shards missing for {dataset}, e.g. {missing[:5]}")
    parts = [pd.read_pickle(d / "shards" / f"{k:05d}.pkl") for k in range(num_shards)]
    V4 = load_v4()
    tag = "" if cfg.get("inputs", {}).get("selector_stage", "v2") == "v2" else "_" + cfg["inputs"]["selector_stage"]
    run = start_run(f"x_transfer_{dataset}{tag}_merge", cfg, False, {"cache": str(d), "v2_run": str(v2)})
    cov = pd.concat([p["coverage"] for p in parts], ignore_index=True)
    bs, n_rand = cfg["v4"]["bootstrap"], len(cfg["v4"]["seeds"])
    out = {"label": f"transfer:{dataset}", "deviation": "D7", "v2_run": v2.name, "tau_rej_chain": parts[0]["tau"],
           "layer": parts[0]["layer"], "images": int(sum(p["images"] for p in parts))}
    pc = cov[cov.kind == "part"]
    ch = cov[cov.kind == "chain"]
    out["coverage_uncapped"] = {
        "part_recall": float((pc.best_iou >= THR).mean()) if len(pc) else None, "parts": len(pc),
        "part_recall_by_area_bin": {b: float((g.best_iou >= THR).mean()) for b, g in pc.groupby("area_bin")},
        "chains": len(ch),
        "chain_object_found": float((ch.object_iou >= THR).mean()) if len(ch) else None,
        "chain_part_found": float((ch.best_iou >= THR).mean()) if len(ch) else None,
        "chain_feasible_mask_tree": float(ch.feasible_uncapped.mean()) if len(ch) else None,
        "single_level_chains": int((ch.part_ratio >= 0.5).sum())}
    for mode, box in (("strict", False), ("d5", False), ("d6", True), ("no_rejection", False)):
        df = pd.concat([p[mode] for p in parts if len(p.get(mode, ()))], ignore_index=True) \
            if any(len(p.get(mode, ())) for p in parts) else pd.DataFrame()
        if mode == "no_rejection" and read_deviations(cfg).get("D12", {}).get("status") != "approved":
            out[f"natural_{mode}"] = "not reported: deviation D12 is not approved in deviations.jsonl"
            continue
        if not len(df):
            out[f"natural_{mode}"] = None
            continue
        df.to_csv(run / f"chains_{mode}.csv", index=False)
        nr = 0 if box else n_rand
        out[f"natural_{mode}"] = {"all": V4.natural_summary(df, nr, bs, box=box),
                                  "two_level": V4.natural_summary(df[~df.single_level], nr, bs, box=box),
                                  "single_level": (V4.natural_summary(df[df.single_level], nr, bs, box=box)
                                                   if df.single_level.any() else None)}
    cov.to_csv(run / "coverage.csv", index=False)
    write_json(run / "metrics.json", out)
    print(json.dumps({k: out[k] for k in ("images", "coverage_uncapped")}, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = cfg["transfer"]["num_shards"]
    if args.merge:
        merge(cfg, args.dataset, num_shards)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.dataset, args.shard, num_shards)


if __name__ == "__main__":
    main()
