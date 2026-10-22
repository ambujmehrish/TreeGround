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
from viz_proposals import DEPTH_COLS, DEPTH_NAMES, anchor, fit, label, legend, overlay, title  # noqa: E402
from viz_tree_levels import names_for  # noqa: E402

# Diagnostic: the same SAM 3 nodes arranged by mask containment (spec) and by bounding-box containment, side by
# side, one row per depth. Every node has one colour everywhere (both trees, both outlines), so a node can be
# followed between the trees. Small nodes are left out of the drawing and the outline (--min-frac).


def node_colors(n: int) -> np.ndarray:
    # golden-angle hues: neighbouring node ids get clearly different colours
    if n == 0:
        return np.zeros((0, 3), np.int32)
    hue = ((np.arange(n) * 137.508) % 360 / 2).astype(np.uint8)
    sat = np.where(np.arange(n) % 2, 170, 235).astype(np.uint8)
    hsv = np.stack([hue, sat, np.full(n, 255, np.uint8)], 1)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0].astype(np.int32)


def depth_panel(image, dense, idx, colors, title):
    dim = (image * 0.3).astype(np.uint8)
    if len(idx):
        covered = dense[idx].any(0)
        dim[covered] = image[covered]
    pnl = overlay(dim, [dense[i] for i in idx], colors[idx], alpha=0.5, contour=2)
    for t, c in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(pnl, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t, c, cv2.LINE_AA)
    return pnl


def depth_view(image, dense, tree, keep):
    base = (image * 0.3).astype(np.uint8)
    out = base
    dmax = int(tree.depth[:tree.n][keep].max(initial=0))
    for d in range(1, dmax + 1):
        idx = np.flatnonzero((tree.depth[:tree.n] == d) & keep)
        out = overlay(out, [dense[i] for i in idx], [DEPTH_COLS[(d - 1) % len(DEPTH_COLS)]] * len(idx),
                      alpha=0.55, contour=1)
    key = " ".join(f"d{d}={DEPTH_NAMES[(d - 1) % len(DEPTH_NAMES)]}" for d in range(1, dmax + 1))
    return out, key


def overview(image, dense, trees, keep, colors, names):
    drawn = np.flatnonzero(keep)
    base = (image * 0.3).astype(np.uint8)
    if len(drawn):
        cov = dense[drawn].any(0)
        base[cov] = image[cov]
    order = drawn[np.argsort(-dense[drawn].reshape(len(drawn), -1).sum(1))] if len(drawn) else drawn
    nodes_panel = overlay(base, [dense[i] for i in order], colors[order], alpha=0.45, contour=1)
    scale = max(0.35, image.shape[1] / 1400)
    for i in (i for i in order if names[i]):
        label(nodes_panel, str(i), anchor(dense[i]), colors[i], scale)
    panels = [title(fit(image), "image"),
              title(fit(nodes_panel), f"{len(drawn)} nodes, own colour each (numbers: PACO match)")]
    for name, t in trees.items():
        view, key = depth_view(image, dense, t, keep)
        panels.append(title(fit(view), f"{name}: {key}"))
    hmax = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((hmax - p.shape[0], p.shape[1], 3), np.uint8)]) for p in panels]
    strip = np.hstack(panels)
    entries = [(f"{i}: {names[i]}  d{trees['mask tree (spec)'].depth[i]} mask / d{trees['box tree'].depth[i]} box",
                tuple(int(c) for c in colors[i])) for i in order if names[i]]
    return np.vstack([strip, legend(entries, strip.shape[1])])


def chip(rgb) -> str:
    r, g, b = (int(x) for x in rgb)
    return (f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});"
            f"border:1px solid #333;vertical-align:middle;margin-right:4px'></span>")


def outline_html(tree, keep, areas, names, colors, hw, max_lines=300):
    lines, shown = [], 0
    stack = [(int(c), 0) for c in tree.children[tree.root][::-1]]
    while stack and shown < max_lines:
        v, ind = stack.pop()
        if not keep[v]:
            continue
        shown += 1
        lines.append(f"<div style='margin-left:{18 * ind}px;white-space:nowrap'>{chip(colors[v])}node {v} "
                     f"<span style='color:#666'>{100 * areas[v] / hw:.1f}%</span> {html.escape(names[v])}</div>")
        stack.extend((int(c), ind + 1) for c in tree.children[v][::-1])
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--stability", type=float)
    ap.add_argument("--max-levels", type=int, default=6)
    ap.add_argument("--min-frac", type=float, default=0.005, help="hide nodes below this fraction of the image")
    ap.add_argument("--capped", action="store_true",
                    help="use the selector's node set (D4: top nodes by stability), as V2/V4 see it")
    ap.add_argument("--max-nodes", type=int, help="draw at most this many nodes, most stable first")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stability is not None:
        cfg["proposals"]["stability_thresh"] = args.stability
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash, capped=args.capped)
    run = start_run("viz_tree_compare", cfg, False, {"proposals": str(props.path), "tau": args.tau,
                                                      "capped": args.capped, "max_nodes": args.max_nodes,
                                                      "images": args.images, "diagnostic": True})
    (run / "img").mkdir()
    rows = wl.drop_duplicates("image_key").set_index("image_key")
    missing = [k for k in args.images if k not in rows.index]
    if missing:
        raise SystemExit(f"not in the worklist: {missing}")
    pacos = {}
    for split in ("train", "val"):
        ids = {int(k.split(":")[1]) for k in args.images
               if k.startswith("paco_lvis:") and rows.loc[k, "role"] == f"paco_{split}"}
        if ids:
            print(f"loading PACO {split} annotations...", flush=True)
            pacos.update({f"paco_lvis:{i}": im for i, im in paco.load(cfg["data"]["paco_lvis"], split, ids).items()})
    sections, summary = [], []
    for k, key in enumerate(args.images):
        row = rows.loc[key]
        image = load_rgb(row.path)
        hw = image.shape[0] * image.shape[1]
        nodes = props.nodes(row.sha1, args.tau)
        rles = nodes["rles"]
        areas = nodes["areas"]
        trees = {"mask tree (spec)": T.build_tree(rles, cfg.get("device")),
                 "box tree": T.build_box_tree(nodes["boxes"])}
        keep = areas >= args.min_frac * hw
        if args.max_nodes and keep.sum() > args.max_nodes:
            ranked = np.flatnonzero(keep)[np.lexsort((np.flatnonzero(keep), -nodes["stability"][keep]))]
            keep = np.zeros_like(keep)
            keep[ranked[:args.max_nodes]] = True
        dense = np.stack([M.decode(r).astype(bool) for r in rles]) if rles else np.zeros((0,) + image.shape[:2], bool)
        colors = node_colors(len(rles))
        names = names_for(rles, pacos.get(key))
        levels = max(int(t.depth[:t.n][keep].max(initial=0)) for t in trees.values())
        grid_rows = []
        for d in range(1, min(levels, args.max_levels) + 1):
            row_panels = []
            for name, t in trees.items():
                idx = np.flatnonzero((t.depth[:t.n] == d) & keep)
                row_panels.append(depth_panel(image, dense, idx, colors, f"{name}  depth {d}: {len(idx)} nodes"))
            grid_rows.append(np.concatenate(row_panels, 1))
        head = np.concatenate([image, image], 1).copy()
        for x, txt in ((6, "image"), (image.shape[1] + 6, "image")):
            cv2.putText(head, txt, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = np.concatenate([head] + grid_rows, 0)
        fname = f"{k:02d}_{key.replace(':', '_')}.jpg"
        cv2.imwrite(str(run / "img" / fname), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        oname = f"{k:02d}_{key.replace(':', '_')}_overview.jpg"
        cv2.imwrite(str(run / "img" / oname), cv2.cvtColor(overview(image, dense, trees, keep, colors, names),
                                                           cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        stats = {}
        for name, t in trees.items():
            dep = t.depth[:t.n][keep]
            stats[name] = {"max_depth": int(dep.max(initial=0)),
                           "nodes_per_depth": {int(d): int((dep == d).sum()) for d in np.unique(dep)}}
        changed = int(sum(1 for v in np.flatnonzero(keep)
                          if trees["mask tree (spec)"].parent[v] != trees["box tree"].parent[v]))
        rec = {"image": key, "nodes": len(rles), "nodes_drawn": int(keep.sum()),
               "nodes_with_different_parent": changed, "trees": stats}
        summary.append(rec)
        cols = "".join(f"<div style='flex:1;min-width:300px'><h3>{html.escape(name)}</h3>"
                       f"{outline_html(t, keep, areas, names, colors, hw)}</div>" for name, t in trees.items())
        sections.append(f"<h2>{html.escape(key)}</h2><p>{len(rles)} nodes, {int(keep.sum())} drawn "
                        f"(at least {100 * args.min_frac:.1f}% of the image); {changed} drawn nodes have a different "
                        f"parent in the two trees. {html.escape(str(stats))}</p>"
                        f"<img src='img/{oname}' style='max-width:100%'>"
                        f"<details open><summary>per-depth comparison (left mask tree, right box tree)</summary>"
                        f"<img src='img/{fname}' style='max-width:100%'></details>"
                        f"<details open><summary>tree outlines (same colours as the picture)</summary>"
                        f"<div style='display:flex;gap:24px;flex-wrap:wrap;font:12px monospace'>{cols}</div></details>")
        print(f"{key}: nodes {len(rles)} drawn {int(keep.sum())} different parent {changed} {stats}", flush=True)
    page = ("<!doctype html><meta charset='utf-8'><title>TreeGround tree comparison</title>"
            "<body style='font:14px sans-serif;margin:16px'>"
            f"<h1>Mask tree vs box tree, tau_d {args.tau}{' (selector node set, D4)' if args.capped else ''}</h1>"
            f"<p>{html.escape(str(props.path))}</p>"
            "<p>Left: containment of SAM 3 masks (spec). Right: containment of the same nodes' bounding boxes. "
            "Rows are tree depths; each node has the same colour in both trees and in the outlines. "
            "Names are PACO objects or parts the node matches at IoU &ge; 0.5.</p>" + "".join(sections) + "</body>")
    (run / "index.html").write_text(page)
    write_json(run / "metrics.json", {"images": summary, "tau": args.tau, "diagnostic": True})
    print(run / "index.html")


if __name__ == "__main__":
    main()
