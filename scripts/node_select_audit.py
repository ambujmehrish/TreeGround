import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import keep_column  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# Read-only diagnostic (development, PACO-LVIS val): which node set to keep from the stored SAM 3 proposals.
# Starting from the deduplicated stored masks (tau_d), each option keeps a subset; the tree is rebuilt over the
# subset with the D3 rule (80% containment). Per option: part recall (all val part annotations, and small parts),
# I1 chain coverage (object found, part found, tree-feasible path), nodes per image, and noise near the target:
# per chain part, candidates with 0.3 <= IoU < 0.5 ("confusers") and with IoU >= 0.5 ("near-duplicate goods").
# Nothing is cached or trained; no test split is read.

THR = 0.5
SMALL, LARGE = 0.005, 0.05          # relative area bins for the size-balanced budget
SPILL_MIN = 0.2                      # a spilling mask overlaps >= 2 larger nodes by >= 20% of itself, inside none

OPTIONS = {
    "cap256_stability": {"cap": 256},                     # the current S2-long tree (D4)
    "cap384_stability": {"cap": 384},
    "cap512_stability": {"cap": 512},
    "uncapped": {},
    "q80_cap256": {"sam_iou": 0.80, "cap": 256},
    "q80_cap512": {"sam_iou": 0.80, "cap": 512},
    "amg_default": {"sam_iou": 0.88, "stability": 0.95},  # SAM automatic mask generator defaults, no cap
    "size_cap256": {"cap": 256, "balance": "size"},
    "size_cap512": {"cap": 512, "balance": "size"},
    "level_cap512": {"cap": 512, "balance": "level"},
    "q80_size_cap512": {"sam_iou": 0.80, "cap": 512, "balance": "size"},
    "q80_size_cap512_nospill": {"sam_iou": 0.80, "cap": 512, "balance": "size", "nospill": True},
}


def by_stability(idx: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    # D4 ranking: higher stability first, ties to the lower proposal index
    return idx[np.lexsort((df.src_index.to_numpy()[idx], -df.stability.to_numpy()[idx]))]


def balanced(idx: np.ndarray, df: pd.DataFrame, groups: np.ndarray, cap: int) -> np.ndarray:
    # equal budget per group; a group with fewer masks gives its unused budget to the others
    ranked = {g: list(by_stability(idx[groups[idx] == g], df)) for g in np.unique(groups[idx])}
    keep, share = [], {g: cap // len(ranked) for g in ranked}
    spare = cap - sum(share.values())
    for g in ranked:
        take = ranked[g][:share[g]]
        spare += share[g] - len(take)
        keep += take
        ranked[g] = ranked[g][share[g]:]
    rest = by_stability(np.asarray([i for g in ranked for i in ranked[g]], np.int64), df) if spare > 0 else []
    keep += list(rest[:spare])
    return np.sort(np.asarray(keep, np.int64))


def spill(sub: np.ndarray, areas: np.ndarray, inter: np.ndarray) -> np.ndarray:
    out = np.zeros(len(sub), bool)
    for k, i in enumerate(sub):
        larger = sub[areas[sub] > areas[i]]
        frac = inter[i, larger] / max(1, areas[i])
        out[k] = (frac >= 0.8).sum() == 0 and (frac >= SPILL_MIN).sum() >= 2
    return out


def select(opt: dict, df: pd.DataFrame, hw: int, areas: np.ndarray, inter: np.ndarray) -> np.ndarray:
    idx = np.arange(len(df))
    if "sam_iou" in opt:
        idx = idx[df.sam_iou.to_numpy()[idx] >= opt["sam_iou"]]
    if "stability" in opt:
        idx = idx[df.stability.to_numpy()[idx] >= opt["stability"]]
    cap = opt.get("cap")
    if cap and len(idx) > cap:
        if opt.get("balance") == "size":
            rel = areas / hw
            groups = np.where(rel < SMALL, 0, np.where(rel < LARGE, 1, 2))
            idx = balanced(idx, df, groups, cap)
        elif opt.get("balance") == "level":
            idx = balanced(idx, df, df.src_index.to_numpy() % 3, cap)
        else:
            idx = np.sort(by_stability(idx, df)[:cap])
    if opt.get("nospill") and len(idx):
        idx = idx[~spill(idx, areas, inter)]
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tau = cfg["proposals"]["node_dedup"]
    _, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash)          # uncapped store; options apply their own cap
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key")
    if args.max_images:
        rows = rows.iloc[:args.max_images]
    run = start_run("node_select_audit", cfg, False, {"tau": tau, "split": "PACO-LVIS val"})
    part_rows, chain_rows, size_rows = [], [], []
    t0 = time.time()
    for n_img, row in enumerate(rows.itertuples(), 1):
        img = images[int(row.image_key.split(":")[1])]
        df = props.rows(row.sha1)
        df = df[df[keep_column(tau)].astype(bool)].reset_index(drop=True)
        h, w = int(props.index.loc[row.sha1, "h"]), int(props.index.loc[row.sha1, "w"])
        if (h, w) != (img.h, img.w):
            raise ProtocolError(f"{row.image_key}: annotation size differs from proposal masks")
        rles = [{"size": [h, w], "counts": c} for c in df.rle]
        if not rles:
            continue
        areas = M.areas(rles)
        inter = T.intersections_dense(rles, cfg.get("device")) if len(rles) > 64 else M.intersections(rles, rles)
        gts = paco.gt_rles(img)
        parts = img.parts
        part_iou = M.iou(rles, [p.rle for p in parts]) if parts else np.zeros((len(rles), 0))
        part_bins = [area_bin(a) for a in M.areas([p.rle for p in parts])] if parts else []
        chains = [q for q in paco.i1_queries(img)[0] if q.kind == "chain"]
        chain_iou = [M.iou(rles, [gts[a] for a in q.gt]).T for q in chains]
        for name, opt in OPTIONS.items():
            sub = select(opt, df, h * w, areas, inter)
            size_rows.append({"option": name, "image": row.image_key, "nodes": len(sub), "stored": len(rles)})
            for k in range(len(parts)):
                best = float(part_iou[sub, k].max()) if len(sub) else 0.0
                part_rows.append({"option": name, "hit": best >= THR, "area_bin": part_bins[k]})
            if not chains:
                continue
            tree = T.from_parents(T.parents_from_overlaps(areas[sub], inter[np.ix_(sub, sub)])) if len(sub) else None
            for q, ci in zip(chains, chain_iou):
                step = ci[:, sub]
                o_hit = bool(len(sub) and step[0].max() >= THR)
                p_hit = bool(len(sub) and step[1].max() >= THR)
                chain_rows.append({
                    "option": name, "object_hit": o_hit, "part_hit": p_hit,
                    "feasible": bool(o_hit and p_hit and T.max_iou_path(tree, step, THR) is not None),
                    "confusers": int(((step[1] >= 0.3) & (step[1] < THR)).sum()) if len(sub) else 0,
                    "goods": int((step[1] >= THR).sum()) if len(sub) else 0})
        if n_img % 100 == 0:
            print(f"{n_img}/{len(rows)} images {time.time() - t0:.0f}s", flush=True)
    P, C, S = pd.DataFrame(part_rows), pd.DataFrame(chain_rows), pd.DataFrame(size_rows)
    out = {"label": "dev", "diagnostic": True, "split": "PACO-LVIS val", "tau_d": tau, "images": int(S.image.nunique()),
           "stored_masks_per_image_mean": float(S[S.option == "uncapped"].stored.mean()), "options": {}}
    for name in OPTIONS:
        p, c, s = P[P.option == name], C[C.option == name], S[S.option == name]
        out["options"][name] = {
            "setting": OPTIONS[name],
            "nodes_per_image_mean": float(s.nodes.mean()), "nodes_per_image_max": int(s.nodes.max()),
            "part_recall": float(p.hit.mean()),
            "part_recall_by_size": {b: float(g.hit.mean()) for b, g in p.groupby("area_bin")},
            "chains": int(len(c)), "chain_object_found": float(c.object_hit.mean()),
            "chain_part_found": float(c.part_hit.mean()), "chain_tree_feasible": float(c.feasible.mean()),
            "confusers_per_chain_part_mean": float(c.confusers.mean()),
            "good_candidates_per_found_part_mean": float(c[c.part_hit].goods.mean()) if c.part_hit.any() else 0.0}
    write_json(run / "metrics.json", out)
    for name, o in out["options"].items():
        print(f"{name:26s} nodes {o['nodes_per_image_mean']:7.1f} part_recall {o['part_recall']:.3f} "
              f"small {o['part_recall_by_size'].get('small', float('nan')):.3f} chain_part {o['chain_part_found']:.3f} "
              f"tree {o['chain_tree_feasible']:.3f} confusers {o['confusers_per_chain_part_mean']:.2f} "
              f"goods {o['good_candidates_per_found_part_mean']:.2f}", flush=True)
    print(json.dumps({"stored_masks_per_image_mean": out["stored_masks_per_image_mean"]}))
    print(run)


if __name__ == "__main__":
    main()
