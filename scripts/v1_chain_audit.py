import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# Audit of the chain coverage numbers with code that shares nothing with the tree module except the masks:
#   brute:  some proposal pair (o, p) with IoU >= 0.5 to the GT object and GT part, p strictly smaller than o and
#           |p ∩ o| >= 0.8 |p|, counted directly from dense pixel arrays (direct containment, no tree)
#   same:   some single proposal has IoU >= 0.5 with both the GT object and the GT part
#   tree:   the tree-based feasibility used by V1 (T.max_iou_path on T.build_tree)
# Disagreements between brute and tree are listed and drawn, together with a sample of chains lost by both.
THR = 0.5


def dense(rles):
    return np.stack([M.decode(r) for r in rles]).astype(bool) if rles else np.zeros((0, 1, 1), bool)


def iou_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa, fb = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None] - inter
    return inter / np.maximum(union, 1)


def draw(image, obj_gt, part_gt, o_mask, p_mask, title):
    def tint(m, color):
        out = image.copy()
        out[m] = (0.45 * np.array(color) + 0.55 * out[m]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, 2)
        return out
    panels = [tint(obj_gt, (40, 160, 255)), tint(part_gt, (255, 140, 0)),
              tint(o_mask, (40, 160, 255)) if o_mask is not None else image.copy(),
              tint(p_mask, (255, 140, 0)) if p_mask is not None else image.copy()]
    names = ["GT object", "GT part", "best proposal for object", "best proposal for part"]
    for pnl, name in zip(panels, names):
        cv2.putText(pnl, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pnl, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    row = np.concatenate(panels, 1)
    bar = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title[:160], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, row], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stability", type=float)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--draw", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stability is not None:
        cfg["proposals"]["stability_thresh"] = args.stability
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash)
    run = start_run("v1_chain_audit", cfg, False, {"proposals": str(props.path), "tau": args.tau,
                                                    "stability": cfg["proposals"]["stability_thresh"]})
    (run / "img").mkdir()
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key").iloc[:args.max_images]
    recs, drawn = [], []
    for i, row in enumerate(rows.itertuples(), 1):
        img = images[int(row.image_key.split(":")[1])]
        chains = [q for q in paco.i1_queries(img)[0] if q.kind == "chain"]
        if not chains:
            continue
        nodes = props.nodes(row.sha1, args.tau)
        if (nodes["h"], nodes["w"]) != (img.h, img.w):
            raise ProtocolError(f"{row.image_key}: size mismatch")
        rles = nodes["rles"]
        D = dense(rles)
        area = D.reshape(len(D), -1).sum(1)
        tree = T.build_tree(rles) if rles else None
        gts = paco.gt_rles(img)
        picture = None
        for q in chains:
            og, pg = M.decode(gts[q.gt[0]]).astype(bool), M.decode(gts[q.gt[1]]).astype(bool)
            if len(D):
                ious = iou_dense(np.stack([og, pg]), D)
                ref = M.iou(rles, [gts[q.gt[0]], gts[q.gt[1]]]).T
                if np.abs(ious - ref).max() > 1e-6:
                    raise ProtocolError(f"{q.qid}: dense IoU differs from pycocotools IoU")
            else:
                ious = np.zeros((2, 0))
            O, P = np.flatnonzero(ious[0] >= THR), np.flatnonzero(ious[1] >= THR)
            brute = False
            if len(O) and len(P):
                inter = D[P].reshape(len(P), -1).astype(np.float32) @ D[O].reshape(len(O), -1).astype(np.float32).T
                ok = (area[P][:, None] < area[O][None]) & (5 * inter >= 4 * area[P][:, None])
                brute = bool(ok.any())
            same = bool(len(np.intersect1d(O, P)))
            feas = bool(len(O) and len(P) and T.max_iou_path(tree, ious, THR) is not None)
            o_best = int(np.argmax(ious[0])) if ious.shape[1] else -1
            p_best = int(np.argmax(ious[1])) if ious.shape[1] else -1
            rec = {"qid": q.qid, "object_found": bool(len(O)), "part_found": bool(len(P)), "brute": brute,
                   "tree": feas, "same_mask_hits_both": same, "n_nodes": len(rles),
                   "best_object_iou": float(ious[0].max()) if ious.shape[1] else 0.0,
                   "best_part_iou": float(ious[1].max()) if ious.shape[1] else 0.0,
                   "best_nodes_identical": o_best == p_best and o_best >= 0,
                   "gt_part_inside_gt_object": float((og & pg).sum() / max(1, pg.sum())),
                   "gt_part_over_object_area": float(pg.sum() / max(1, og.sum()))}
            recs.append(rec)
            mismatch = brute != feas
            if (mismatch or (rec["object_found"] and rec["part_found"] and not feas)) and len(drawn) < args.draw:
                if picture is None:
                    picture = load_rgb(row.path)
                title = (f"{q.qid} | tree={feas} brute={brute} same_mask={same} "
                         f"objIoU={rec['best_object_iou']:.2f} partIoU={rec['best_part_iou']:.2f} "
                         f"GTpart/GTobj area={rec['gt_part_over_object_area']:.2f}")
                pan = draw(picture, og, pg, D[o_best] if o_best >= 0 else None, D[p_best] if p_best >= 0 else None,
                           title)
                name = f"{len(drawn):03d}.jpg"
                cv2.imwrite(str(run / "img" / name), cv2.cvtColor(pan, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                drawn.append((name, title))
        if i % 100 == 0:
            print(f"{i}/{len(rows)} images", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(run / "chains.csv", index=False)
    both = df[df.object_found & df.part_found]
    lost = both[~both.tree]
    out = {
        "tau": args.tau, "images": len(rows), "chains": len(df),
        "tree_feasible": float(df.tree.mean()), "brute_feasible": float(df.brute.mean()),
        "tree_vs_brute_disagreements": int((df.tree != df.brute).sum()),
        "both_found": float(len(both) / max(1, len(df))),
        "lost_given_both_found": len(lost),
        "lost_where_one_mask_hits_both": int(lost.same_mask_hits_both.sum()),
        "lost_where_best_nodes_identical": int(lost.best_nodes_identical.sum()),
        "lost_gt_part_over_object_area_median": float(lost.gt_part_over_object_area.median()) if len(lost) else None,
        "all_gt_part_over_object_area_median": float(df.gt_part_over_object_area.median()),
    }
    write_json(run / "metrics.json", out)
    page = ["<html><body style='font-family:sans-serif'>", f"<h2>Chain audit tau {args.tau}</h2>",
            f"<pre>{html.escape(str(out))}</pre>"]
    page += [f"<p>{html.escape(t)}</p><img src='img/{n}' style='max-width:100%'>" for n, t in drawn]
    (run / "index.html").write_text("\n".join(page + ["</body></html>"]))
    print(out)
    print(run)


if __name__ == "__main__":
    main()
