import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# Diagnostic: splits the gap between "both masks exist" and "the tree links them" for PACO val I1 chains.
# Nothing here is a gate quantity. For chains whose object and part both have a proposal with IoU >= 0.5 but no
# feasible tree path, the best-IoU object and part nodes are inspected:
#   spill:        the part node is < 80% inside the object node, so containment never makes it a descendant
#   not_smaller:  the part node is not smaller than the object node
#   wrong_branch: the part node is >= 80% inside the object node, but its smallest container lies outside the
#                 object's subtree (an overlapping mask that is not contained in the object node)
THR = 0.5
# containment thresholds to try, as exact fractions num/den; (4, 5) is the spec's 0.8
CONTAINMENT = {"0.8": (4, 5), "0.7": (7, 10), "0.6": (3, 5), "0.5": (1, 2)}


def classify(tree: T.Tree, areas: np.ndarray, inter_po: int, o: int, p: int) -> str:
    if areas[p] >= areas[o]:
        return "not_smaller"
    if T.CONTAIN_DEN * inter_po < T.CONTAIN_NUM * areas[p]:
        return "spill"
    return "linked" if tree.is_strict_descendant(p, o) else "wrong_branch"


def split(cfg, props, wl, tau, device, max_images):
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key")
    if max_images:
        rows = rows.iloc[:max_images]
    recs, t0 = [], time.time()
    for i, row in enumerate(rows.itertuples(), 1):
        if i % 200 == 0 or i == len(rows):
            print(f"[tau {tau}] {i}/{len(rows)} images {time.time() - t0:.0f}s", flush=True)
        img = images[int(row.image_key.split(":")[1])]
        chains = [q for q in paco.i1_queries(img)[0] if q.kind == "chain"]
        if not chains:
            continue
        nodes = props.nodes(row.sha1, tau)
        if (nodes["h"], nodes["w"]) != (img.h, img.w):
            raise ProtocolError(f"{row.image_key}: annotation size differs from proposal masks")
        rles = nodes["rles"]
        trees = {}
        if rles:
            areas = M.areas(rles)
            inter_all = (T.intersections_dense(rles, device) if device and len(rles) > 64
                         else M.intersections(rles, rles))
            trees = {c: T.from_parents(T.parents_from_overlaps(areas, inter_all, n, d))
                     for c, (n, d) in CONTAINMENT.items()}
        tree = trees.get("0.8")
        gts = paco.gt_rles(img)
        for q in chains:
            obj, part = gts[q.gt[0]], gts[q.gt[1]]
            step = M.iou(rles, [obj, part]).T if rles else np.zeros((2, 0))
            o_hit = bool(step.shape[1] and step[0].max() >= THR)
            p_hit = bool(step.shape[1] and step[1].max() >= THR)
            rec = {"qid": q.qid, "label": q.qid.split("|")[-1], "part_area_bin": area_bin(M.areas([part])[0]),
                   "object_hit": o_hit, "part_hit": p_hit, "both_hit": o_hit and p_hit,
                   "tree_feasible": bool(o_hit and p_hit and T.max_iou_path(tree, step, THR) is not None),
                   "reason": None, "containment": np.nan,
                   **{f"tree_feasible_c{c}": bool(o_hit and p_hit and T.max_iou_path(t, step, THR) is not None)
                      for c, t in trees.items()}}
            if rec["both_hit"] and not rec["tree_feasible"]:
                o, p = int(np.argmax(step[0])), int(np.argmax(step[1]))
                inter = int(M.intersections([rles[p]], [rles[o]])[0, 0])
                rec["containment"] = inter / max(1, int(nodes["areas"][p]))
                rec["reason"] = classify(tree, nodes["areas"], inter, o, p)
            recs.append(rec)
    return pd.DataFrame(recs)


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    both = df[df.both_hit]
    lost = both[~both.tree_feasible]
    out = {
        "chains": n,
        "object_found": float(df.object_hit.mean()),
        "part_found": float(df.part_hit.mean()),
        "both_found": float(df.both_hit.mean()),
        "tree_feasible": float(df.tree_feasible.mean()),
        "tree_feasible_given_both_found": float(both.tree_feasible.mean()) if len(both) else float("nan"),
        "tree_feasible_by_containment": {c: float(df[f"tree_feasible_c{c}"].mean()) if f"tree_feasible_c{c}" in df
                                         else 0.0 for c in CONTAINMENT},
        "structural_loss_chains": int(len(lost)),
        "structural_loss_reasons": lost.reason.value_counts().to_dict(),
        "lost_containment_quantiles": (lost.containment.quantile([0.25, 0.5, 0.75]).round(3).to_dict()
                                       if len(lost) else {}),
        "missing": {"object_only_missing": int((~df.object_hit & df.part_hit).sum()),
                    "part_only_missing": int((df.object_hit & ~df.part_hit).sum()),
                    "both_missing": int((~df.object_hit & ~df.part_hit).sum())},
        "by_part_area_bin": {b: {"chains": len(g), "both_found": float(g.both_hit.mean()),
                                 "tree_feasible": float(g.tree_feasible.mean())}
                             for b, g in df.groupby("part_area_bin")},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stability", type=float, help="open the proposal cache built with this stability threshold "
                                                     "(0.9 = spec cache; default: the current config, D3)")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stability is not None:
        cfg["proposals"]["stability_thresh"] = args.stability
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash)
    tag = f"stability{cfg['proposals']['stability_thresh']}"
    run = start_run("v1_chain_split", cfg, False, {"manifest_run": str(mrun), "proposals": str(props.path),
                                                    "proposal_setting": tag})
    out = {"split": "PACO-LVIS val I1 chains", "label": "dev", "diagnostic": True, "proposal_setting": tag,
           "by_tau": {}}
    for tau in cfg["v1"]["taus"]:
        df = split(cfg, props, wl, tau, cfg.get("device"), args.max_images)
        df.to_csv(run / f"chains_tau{tau}.csv", index=False)
        out["by_tau"][str(tau)] = s = summary(df)
        print(f"tau {tau}: chains {s['chains']} object {s['object_found']:.3f} part {s['part_found']:.3f} "
              f"both {s['both_found']:.3f} tree {s['tree_feasible']:.3f} "
              f"tree|both {s['tree_feasible_given_both_found']:.3f} lost {s['structural_loss_reasons']} "
              f"by containment {s['tree_feasible_by_containment']}", flush=True)
        write_json(run / "metrics.json", out)
    print(run)


if __name__ == "__main__":
    main()
