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
from treeground.data import paco, partonomy  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.run import load_config, start_run, write_json  # noqa: E402

THR = 0.5
PANEL_W = 640
GREEN, RED, WHITE = (40, 200, 40), (230, 40, 40), (255, 255, 255)


def palette(n: int, seed: int = 0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 3), np.int32)
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.integers(0, 180, n), np.full(n, 200), np.full(n, 255)], 1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0].astype(np.int32)


def overlay(img: np.ndarray, masks: list[np.ndarray], colors, alpha: float = 0.45, contour: int = 2) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for m, c in zip(masks, colors):
        out[m] = (1 - alpha) * out[m] + alpha * np.asarray(c, np.float32)
    out = out.astype(np.uint8)
    for m, c in zip(masks, colors):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, tuple(int(x) for x in c), contour)
    return out


def label(img: np.ndarray, text: str, xy: tuple[int, int], color, scale: float) -> None:
    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 2)))
    (w, h), base = cv2.getTextSize(text, font, scale, th)
    x = int(np.clip(xy[0], 0, max(0, img.shape[1] - w - 2)))
    y = int(np.clip(xy[1], h + 2, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - h - 2), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 1, y), font, scale, tuple(int(c) for c in color), th, cv2.LINE_AA)


def anchor(m: np.ndarray) -> tuple[int, int]:
    # deepest interior point, so the tag sits on the mask even for thin or concave parts
    if not m.any():
        return 0, 12
    dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x) - 6, int(y) + 6


def legend(entries: list[tuple[str, tuple]], width: int, scale: float = 0.5) -> np.ndarray:
    if not entries:
        return np.zeros((0, width, 3), np.uint8)
    col_w, line_h = 420, 20
    cols = max(1, width // col_w)
    rows = -(-len(entries) // cols)
    strip = np.zeros((rows * line_h + 8, width, 3), np.uint8)
    for i, (text, color) in enumerate(entries):
        x, y = (i // rows) * col_w + 6, (i % rows) * line_h + 18
        cv2.putText(strip, text[:58], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, tuple(int(c) for c in color), 1,
                    cv2.LINE_AA)
    return strip


def title(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img: np.ndarray) -> np.ndarray:
    s = PANEL_W / img.shape[1]
    return cv2.resize(img, (PANEL_W, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def gt_for(key: str, sources: dict) -> tuple[list[str], list[M.Rle], list[tuple], str]:
    src, ident = key.split(":", 1)
    if src == "paco_lvis":
        img = sources["paco"][int(ident)]
        names = [o.name for o in img.objects] + [p.label for p in img.parts]
        rles = [o.rle for o in img.objects] + [p.rle for p in img.parts]
        gts = paco.gt_rles(img)
        chains = [(q.qid.split("|")[-1], gts[q.gt[0]], gts[q.gt[1]]) for q in paco.i1_queries(img)[0]
                  if q.kind == "chain"]
        return names, rles, chains, ""
    if src == "partonomy":
        img = sources["partonomy"][key]
        names = [f"{img.label}:{n}" for n, rs in img.parts.items() for _ in rs] + ["(object)"] * len(img.roots)
        rles = [r for rs in img.parts.values() for r in rs] + img.roots
        return names, rles, [(n, o, p) for n, o, p in partonomy.chains(img)], ""
    if src == "grefcoco":
        texts = sources["gref"].get(key, [])
        return [], [], [], "no-target queries: " + " | ".join(f'"{t}"' for t in texts)
    return [], [], [], f"no GT loader for {src}"


def render(image: np.ndarray, names, gt_rles, nodes: dict, tree: T.Tree, note: str) -> tuple[np.ndarray, list[dict]]:
    h, w = image.shape[:2]
    scale = max(0.4, min(0.9, w / 900))
    gt_masks = [M.decode(r) for r in gt_rles]
    node_masks = [M.decode(r) for r in nodes["rles"]]
    ious = M.iou(nodes["rles"], gt_rles) if node_masks and gt_rles else np.zeros((len(node_masks), len(gt_rles)))

    gt_panel = overlay(image, gt_masks, palette(len(gt_masks), 1))
    for k, m in enumerate(gt_masks):
        label(gt_panel, str(k), anchor(m), WHITE, scale)
    if note:
        label(gt_panel, note[:120], (4, h - 6), (255, 220, 120), scale)

    rows, best_masks, best_cols = [], [], []
    for k, (n, m) in enumerate(zip(names, gt_masks)):
        b = int(np.argmax(ious[:, k])) if len(node_masks) else -1
        v = float(ious[b, k]) if b >= 0 else 0.0
        rows.append({"id": k, "gt": n, "gt_area": int(m.sum()), "best_node": b, "best_iou": round(v, 3),
                     "hit": v >= THR})
        if b >= 0:
            best_masks.append(node_masks[b])
            best_cols.append(GREEN if v >= THR else RED)
    best_panel = overlay(image, best_masks, best_cols)
    for r, m in zip((r for r in rows if r["best_node"] >= 0), best_masks):
        label(best_panel, str(r["id"]), anchor(m), GREEN if r["hit"] else RED, scale)

    order = np.argsort(-nodes["areas"]) if len(node_masks) else []
    depth_cols = palette(int(tree.depth.max()) + 1, 7) if len(node_masks) else []
    all_panel = overlay(image, [node_masks[i] for i in order], [depth_cols[tree.depth[i]] for i in order],
                        alpha=0.25, contour=1)

    panels = [title(fit(image), f"image {w}x{h}"),
              title(fit(gt_panel), f"GT: {len(gt_masks)} masks"),
              title(fit(best_panel), f"best proposal per GT (green IoU>={THR}, red <{THR})"),
              title(fit(all_panel), f"all {len(node_masks)} proposals, colour = tree depth (max {int(tree.depth.max())})")]
    hmax = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((hmax - p.shape[0], PANEL_W, 3), np.uint8)]) for p in panels]
    canvas = np.hstack(panels)
    entries = [(f"{r['id']}: {r['gt']}  IoU {r['best_iou']:.2f}", GREEN if r["hit"] else RED) for r in rows]
    return np.vstack([canvas, legend(entries, canvas.shape[1])]), rows


def chain_rows(chains, nodes, tree) -> list[dict]:
    out = []
    for name, obj, part in chains:
        step = M.iou(nodes["rles"], [obj, part]).T if nodes["rles"] else np.zeros((2, 0))
        feas = bool(step.shape[1]) and T.max_iou_path(tree, step, THR) is not None
        best = bool(step.shape[1]) and T.best_match_path(tree, step, THR) is not None
        out.append({"chain": name, "object_best_iou": round(float(step[0].max()) if step.shape[1] else 0.0, 3),
                    "part_best_iou": round(float(step[1].max()) if step.shape[1] else 0.0, 3),
                    "feasible_path": feas, "best_match_path": best})
    return out


def table(rows: list[dict]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(k)}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tau", type=float, default=None, help="dedup threshold of the nodes to draw (default node_dedup)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tau = args.tau if args.tau is not None else cfg["proposals"]["node_dedup"]
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    props = open_proposals(cfg, args.smoke, manifest, mhash)
    run = start_run("viz_proposals", cfg, args.smoke, {"manifest_run": str(mrun), "proposals": str(props.path),
                                                        "tau": tau, "diagnostic": True})
    (run / "images").mkdir()
    keys = wl.drop_duplicates("image_key").sort_values(["role", "image_key"])
    if args.limit:
        keys = keys.head(args.limit)
    sources = {"paco": {}, "partonomy": {}, "gref": {}}
    for split in ("train", "val"):
        if (keys.image_key.str.startswith("paco_lvis:") & keys.role.eq(f"paco_{split}")).any():
            print(f"loading PACO {split} annotations (can take a few minutes)...", flush=True)
            ids = {int(k.split(":")[1]) for k in keys[keys.role == f"paco_{split}"].image_key}
            sources["paco"].update(paco.load(cfg["data"]["paco_lvis"], split, ids))
    if keys.image_key.str.startswith("partonomy:").any():
        print("loading PARTONOMY annotations...", flush=True)
        sources["partonomy"] = {i.key: i for i in partonomy.load(cfg["data"]["partonomy"]).values()}
    gq = pd.read_parquet(mrun / "grefcoco_queries.parquet")
    for k, g in gq.groupby("image_key"):
        sources["gref"][k] = g.text.tolist()

    print(f"drawing {len(keys)} images -> {run}", flush=True)
    sections, summary = [], []
    for i, row in enumerate(keys.itertuples(), 1):
        image = load_rgb(row.path)
        nodes = props.nodes(row.sha1, tau)
        tree = T.build_tree(nodes["rles"])
        names, gt_rles, chains, note = gt_for(row.image_key, sources)
        canvas, rows = render(image, names, gt_rles, nodes, tree, note)
        fname = f"{i:03d}_{row.image_key.replace(':', '_')}.jpg"
        cv2.imwrite(str(run / "images" / fname), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        crow = chain_rows(chains, nodes, tree)
        rec = {"#": i, "image": row.image_key, "role": row.role, "size": f"{row.w}x{row.h}", "nodes": len(nodes["rles"]),
               "max_depth": int(tree.depth.max()), "gt_masks": len(rows),
               "gt_recall@0.5": round(float(np.mean([r["hit"] for r in rows])), 3) if rows else "",
               "chains": len(crow), "feasible": sum(c["feasible_path"] for c in crow),
               "best_match": sum(c["best_match_path"] for c in crow)}
        summary.append(rec)
        sections.append(f'<h2 id="i{i}">{i}. {html.escape(row.image_key)} ({html.escape(row.role)})</h2>'
                        f'<p>{html.escape(note)}</p><img src="images/{fname}" loading="lazy">'
                        f"<details><summary>GT masks and best IoU</summary>{table(rows)}</details>"
                        f"<details open><summary>object-part chains</summary>{table(crow)}</details>")
        print(f"[{i}/{len(keys)}] {row.image_key} nodes={len(nodes['rles'])} gt={len(rows)} "
              f"recall={rec['gt_recall@0.5']} chains={len(crow)} feasible={rec['feasible']}", flush=True)

    links = [dict(r, image=f'<a href="#i{r["#"]}">{html.escape(r["image"])}</a>') for r in summary]
    head = "".join(f"<th>{k}</th>" for k in summary[0]) if summary else ""
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>" for r in links)
    page = ("<!doctype html><meta charset='utf-8'><title>TreeGround proposals</title><style>"
            "body{font:13px sans-serif;margin:16px;background:#111;color:#ddd}img{max-width:100%}"
            "table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #444;padding:2px 6px}"
            "a{color:#8cf}h2{margin-top:28px}</style>"
            f"<h1>Proposals (tau_d={tau}) vs ground truth</h1><p>{html.escape(str(props.path))}</p>"
            f"<table><tr>{head}</tr>{body}</table>" + "".join(sections))
    (run / "index.html").write_text(page)
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    write_json(run / "metrics.json", {"images": len(summary), "tau": tau, "diagnostic": True})
    print(run / "index.html")


if __name__ == "__main__":
    main()
