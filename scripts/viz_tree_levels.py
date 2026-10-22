import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.run import load_config, start_run, write_json  # noqa: E402
from viz_proposals import overlay, palette  # noqa: E402

# Diagnostic picture of the containment tree SAM 3's own masks give, one panel per tree depth, plus an outline.
# Uses every stored SAM 3 mask kept by dedup at --tau (no zoom, no other model). Nodes are named with the PACO
# object or part they match (IoU >= 0.5) where one exists; unnamed nodes are regions PACO does not annotate.
THR = 0.5


def names_for(rles, img: paco.PacoImage | None) -> list[str]:
    if img is None or not rles:
        return [""] * len(rles)
    gts = [(o.name, o.rle) for o in img.objects] + [(p.label, p.rle) for p in img.parts]
    if not gts:
        return [""] * len(rles)
    iou = M.iou(rles, [g[1] for g in gts])
    best = iou.argmax(1)
    return [f"{gts[b][0]} ({iou[i, b]:.2f})" if iou[i, b] >= THR else "" for i, b in enumerate(best)]


def level_panels(image, dense, tree, max_levels):
    depth = tree.depth[:tree.n]
    panels = []
    for d in range(1, min(int(depth.max(initial=0)), max_levels) + 1):
        idx = np.flatnonzero(depth == d)
        dim = (image * 0.35).astype(np.uint8)
        covered = dense[idx].any(0) if len(idx) else np.zeros(image.shape[:2], bool)
        dim[covered] = image[covered]
        pnl = overlay(dim, [dense[i] for i in idx], palette(len(idx), seed=d), alpha=0.4, contour=1)
        txt = f"depth {d}: {len(idx)} nodes"
        cv2.putText(pnl, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pnl, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(pnl)
    return panels


def grid(panels, cols):
    h, w = panels[0].shape[:2]
    rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    rows[-1] = rows[-1] + [np.full((h, w, 3), 255, np.uint8)] * (cols - len(rows[-1]))
    return np.concatenate([np.concatenate(r, 1) for r in rows], 0)


def outline(tree, areas, names, hw, min_frac, max_lines):
    lines, hidden = [], 0
    stack = [(int(c), 0) for c in tree.children[tree.root][::-1]]
    while stack:
        v, ind = stack.pop()
        frac = areas[v] / hw
        if frac < min_frac:
            hidden += 1 + len(tree.strict_descendants(v))
            continue
        if len(lines) < max_lines:
            lines.append(f"{'    ' * ind}└ node {v}  {100 * frac:.1f}% of image  {names[v]}")
        stack.extend((int(c), ind + 1) for c in tree.children[v][::-1])
    return lines, hidden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image keys, e.g. paco_lvis:110573")
    ap.add_argument("--tau", type=float, default=0.95, help="dedup level of the nodes (0.95 keeps the most)")
    ap.add_argument("--stability", type=float, help="proposal cache built with this stability threshold")
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--min-frac", type=float, default=0.001, help="outline hides nodes below this image fraction")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stability is not None:
        cfg["proposals"]["stability_thresh"] = args.stability
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash)
    run = start_run("viz_tree_levels", cfg, False, {"proposals": str(props.path), "tau": args.tau,
                                                     "images": args.images, "diagnostic": True})
    (run / "img").mkdir()
    rows = wl.drop_duplicates("image_key").set_index("image_key")
    missing = [k for k in args.images if k not in rows.index]
    if missing:
        raise SystemExit(f"not in the worklist: {missing}")
    pacos = {}
    for split in ("train", "val"):
        ids = {int(k.split(":")[1]) for k in args.images if k.startswith("paco_lvis:") and
               rows.loc[k, "role"] == f"paco_{split}"}
        if ids:
            print(f"loading PACO {split} annotations...", flush=True)
            pacos.update({f"paco_lvis:{i}": im for i, im in paco.load(cfg["data"]["paco_lvis"], split, ids).items()})
    sections, summary = [], []
    for k, key in enumerate(args.images):
        row = rows.loc[key]
        image = load_rgb(row.path)
        nodes = props.nodes(row.sha1, args.tau)
        rles = nodes["rles"]
        tree = T.build_tree(rles, cfg.get("device"))
        dense = np.stack([M.decode(r).astype(bool) for r in rles]) if rles else np.zeros((0,) + image.shape[:2], bool)
        names = names_for(rles, pacos.get(key))
        panels = [image.copy()] + level_panels(image, dense, tree, args.max_levels)
        cv2.putText(panels[0], "image", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        fname = f"{k:02d}_{key.replace(':', '_')}.jpg"
        cv2.imwrite(str(run / "img" / fname), cv2.cvtColor(grid(panels, 3), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        lines, hidden = outline(tree, nodes["areas"], names, image.shape[0] * image.shape[1], args.min_frac, 400)
        named = sum(1 for n in names if n)
        depth = tree.depth[:tree.n]
        rec = {"image": key, "nodes": len(rles), "max_depth": int(depth.max(initial=0)),
               "nodes_per_depth": {int(d): int((depth == d).sum()) for d in np.unique(depth)},
               "nodes_matching_paco": named}
        summary.append(rec)
        sections.append(f"<h2>{html.escape(key)}</h2><p>{len(rles)} nodes, depth up to {rec['max_depth']}, "
                        f"{named} match a PACO object or part; per depth {rec['nodes_per_depth']}</p>"
                        f"<img src='img/{fname}' style='max-width:100%'><details><summary>tree outline "
                        f"({hidden} nodes under {100 * args.min_frac:.1f}% of the image hidden)</summary>"
                        f"<pre>{html.escape(chr(10).join(lines))}</pre></details>")
        print(f"{key}: nodes {len(rles)} max depth {rec['max_depth']} per depth {rec['nodes_per_depth']}", flush=True)
    page = ("<!doctype html><meta charset='utf-8'><title>TreeGround tree levels</title>"
            "<body style='font:14px sans-serif;margin:16px'>"
            f"<h1>SAM 3 containment tree, tau_d {args.tau}</h1><p>{html.escape(str(props.path))}</p>"
            "<p>Each panel shows the nodes at one depth of the tree; the rest of the image is dimmed. "
            "Depth 1 hangs directly under the whole image; deeper levels lie inside the level above.</p>"
            + "".join(sections) + "</body>")
    (run / "index.html").write_text(page)
    write_json(run / "metrics.json", {"images": summary, "tau": args.tau, "diagnostic": True})
    print(run / "index.html")


if __name__ == "__main__":
    main()
