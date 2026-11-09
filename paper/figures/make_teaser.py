"""Teaser figure (two-column): (a) candidate masks enter the MLLM as tokens and <SEL> points at one; (b) one model
answers part queries, multi-target expressions and no-target expressions.

Every pixel shown is a real model output: results/figures/teaser_assets.json (scripts/teaser_assets.py, pool_soft_fam
on test queries). Scores are the model's own: (a) softmax of s_v over the whole pool (the part decision is argmax);
(b) sigmoid(s_v) against the gRefCOCO set threshold t chosen on val.
usage: python paper/figures/make_teaser.py [--hero 0 --part 1 --multi 0 --none 0] [--png teaser.png]
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_overview import INTER, bbox, dec, png_b64, text_width  # noqa: E402

W, H = 1400, 440
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1f2633", "#5b6474", "#c9ced6"
PICK = (176, 48, 111)                                  # the selected mask (as in the overview figure)
GRID, TEXT = (59, 111, 182), (217, 130, 43)            # candidate provenance: point grid / SAM 3 text
SET_COLORS = [PICK, (24, 128, 120), (203, 140, 20), (96, 76, 170)]
# test-split gains of the 4B model over the best <=4B baseline, Tab. 1 (part example: PACO; others: gRefCOCO)
STATS = {"part": "PACO part: +14.5 gIoU", "multi": "gRefCOCO: +20.8 / +17.2 gIoU", "none": "N-acc: 89.6 / 86.0"}


def hexc(c) -> str:
    return "#%02x%02x%02x" % tuple(c)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, weight=400, fill=INK, anchor="start", italic=False) -> str:
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{st} xml:space="preserve">{esc(s)}</text>')


def wrap(s: str, size: float, width: float, weight=400) -> list[str]:
    lines, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if cur and text_width(t, size, weight) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur] if cur else lines


def image_tag(x, y, w, h, b64, clip_r=0) -> str:
    cid = f"c{abs(hash((x, y, w, h))) % 10**8}"
    clip = (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{clip_r}"/></clipPath>'
            if clip_r else "")
    ref = f' clip-path="url(#{cid})"' if clip_r else ""
    return (f'{clip}<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none"{ref} '
            f'href="data:image/png;base64,{b64}"/>')


def arrow(x0, y0, x1, y1, color=INK, width=1.8, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{width}"{d} '
            f'marker-end="url(#ah)"/>')


# ---------------------------------------------------------------- image rendering
def load(ex: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(ex["image_jpeg_b64"]))).convert("RGB"))


def crop_box(h, w, focus, aspect, margin=0.35, min_frac=0.45):
    # the smallest window of the given aspect (w / h) that holds the focus box with a margin, inside the image
    x0, y0, x1, y1 = focus
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    cw = max(bw, bh * aspect, min_frac * min(w, h * aspect))
    cw = min(cw, w, h * aspect)
    ch = cw / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = min(max(0, cx - cw / 2), w - cw), min(max(0, cy - ch / 2), h - ch)
    return int(ax), int(ay), int(ax + cw), int(ay + ch)


def edge(m: np.ndarray, r: int = 2) -> np.ndarray:
    p = np.pad(m, r, mode="edge")
    e = np.zeros_like(m)
    for dy, dx in ((0, r), (2 * r, r), (r, 0), (r, 2 * r)):
        e |= m ^ p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return e


def paint(img, masks, colors, fill=0.55, dim=0.0, line=2, dashed=None) -> np.ndarray:
    x = img.astype(float) * (1 - dim) + 255 * dim
    for m, c in zip(masks, colors):
        x = np.where(m[..., None], (1 - fill) * x + fill * np.array(c), x)
    for m, c in zip(masks, colors):
        e = edge(m, line)
        x[e] = c
    if dashed is not None:
        m, c = dashed
        e = edge(m, line)
        yy, xx = np.nonzero(e)
        on = ((yy + xx) // 9) % 2 == 0
        x[yy[on], xx[on]] = c
    return x.clip(0, 255).astype(np.uint8)


def to_png(arr: np.ndarray, box, size) -> str:
    return png_b64(Image.fromarray(arr).crop(box).resize(size, Image.LANCZOS))


def cutout(img: np.ndarray, m: np.ndarray, side: int) -> str:
    # the mask's pixels on a light checkerboard, square crop around the mask
    h, w = m.shape
    x0, y0, x1, y1 = bbox(m)
    s = min(max(x1 - x0, y1 - y0, 24) * 1.3, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = int(min(max(0, cx - s / 2), w - s)), int(min(max(0, cy - s / 2), h - s))
    box = (ax, ay, int(ax + s), int(ay + s))
    yy, xx = np.mgrid[:h, :w]
    board = np.where((((yy - ay) * 8 // max(1, int(s))) + ((xx - ax) * 8 // max(1, int(s)))) % 2 == 0, 238, 226)
    out = np.where(m[..., None], img, board[..., None]).astype(np.uint8)
    return to_png(out, box, (side, side))


# ---------------------------------------------------------------- (a) hero: masks become tokens
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def shown_tokens(ex: dict, k: int) -> list[int]:
    # the pick, the next-highest scores, and the rest spread over the pool (grid first, then SAM 3 text), pool order
    lg = np.asarray(ex["logits"])
    n, ng = len(lg), ex["n_grid"]
    top = [int(i) for i in np.argsort(-lg)[:2]]
    spread = [int(i) for i in np.linspace(0, ng - 1, k)] + list(range(ng, n))[:2]
    return sorted(list(dict.fromkeys(top + spread))[:k])


def query_words(q: str) -> list[str]:
    return q.replace("<SEL>", "").split()


def pool_view(img: np.ndarray, pool: list, ng: int) -> np.ndarray:
    # point-grid masks as coloured segments (large first, small on top), SAM 3 text masks outlined
    rng = np.random.default_rng(3)
    x = img.astype(float) * 0.55 + 255 * 0.45
    for i in sorted(range(ng), key=lambda i: -int(pool[i].sum())):
        c = np.array(Image.new("HSV", (1, 1), (int(rng.integers(256)), 150, 235)).convert("RGB").getpixel((0, 0)))
        x = np.where(pool[i][..., None], 0.45 * x + 0.55 * c, x)
        x[edge(pool[i], 1)] = 255
    for i in range(ng, len(pool)):
        x[edge(pool[i], 2)] = TEXT
    return x.clip(0, 255).astype(np.uint8)


def token_row(ex, img, box, pool, k, ch, width):
    # chips (kind, payload, width); returns None if they do not fit in `width`
    words = query_words(ex["query"])
    chips = [("word", wd, text_width(wd, 12, 500) + 10) for wd in words] + [("gap", None, 6)]
    chips += [("tile", t, ch - 6) for t in range(2)] + [("dots", None, 16), ("gap", None, 6)]
    shown = shown_tokens(ex, k)
    for j, i in enumerate(shown):
        if j and i != shown[j - 1] + 1:
            chips.append(("dots", None, 16))
        chips.append(("mask", i, ch))
    if shown[-1] != len(pool) - 1:
        chips.append(("dots", None, 16))
    chips += [("gap", None, 6), ("sel", None, 52)]
    total = sum(c[2] for c in chips) + 4 * len(chips)
    return chips if total <= width else None


def hero(ex: dict, x0: float, y0: float, wdt: float, h_avail: float) -> str:
    img = load(ex)
    h, w = img.shape[:2]
    pool = [dec(r) for r in ex["pool"]]
    n, ng, pick = len(pool), ex["n_grid"], ex["pick"]
    obj = dec(ex["gt_object"]) if "gt_object" in ex else pool[pick]
    box = crop_box(h, w, bbox(obj | pool[pick]), 1.0, margin=0.25)
    out = []
    # the token row decides where <SEL> sits; the selected-mask image goes right above it
    ch = 42
    chips = next(c for k in (7, 6, 5, 4, 3) if (c := token_row(ex, img, box, pool, k, ch, wdt - 28)))
    S = 150
    yi = y0 + 30
    by = yi + S + 50
    cy = by + 92
    x = x0 + wdt - 14 - (sum(c[2] for c in chips) + 4 * (len(chips) - 1))
    p = softmax(np.asarray(ex["logits"]))
    pmax = max(p[c[1]] for c in chips if c[0] == "mask")
    groups, start, sx, sel_x = [], None, {}, None
    row = []
    for kind, val, cw in chips:
        if kind == "word":
            start = x if start is None else start
            row.append(f'<rect x="{x:.1f}" y="{cy + 8}" width="{cw:.1f}" height="30" rx="5" fill="#fff" stroke="{LINE}"/>')
            row.append(text(x + cw / 2, cy + 28, val, 12, 500, INK, "middle"))
        elif kind == "gap":
            if start is not None:
                groups.append((start, x - 4))
                start = None
        elif kind == "tile":
            start = x if start is None else start
            t = (box[2] - box[0]) // 3
            tx, ty = box[0] + t * (0, 2)[val], box[1] + t * (0, 1)[val]
            row.append(image_tag(round(x, 1), cy + 3, cw, cw, to_png(img, (tx, ty, tx + t, ty + t), (2 * int(cw),) * 2), 4))
        elif kind == "dots":
            start = x if start is None else start
            row.append(text(x + cw / 2, cy + 28, "⋯", 15, 700, MUTED, "middle"))
        elif kind == "mask":
            start = x if start is None else start
            col = PICK if val == pick else GRID if val < ng else TEXT
            row.append(f'<rect x="{x - 1:.1f}" y="{cy - 1}" width="{cw + 2}" height="{cw + 2}" rx="5" fill="#fff" '
                       f'stroke="{hexc(col)}" stroke-width="{3 if val == pick else 1.8}"/>')
            row.append(image_tag(round(x + 1, 1), cy + 1, cw - 2, cw - 2, cutout(img, pool[val], 2 * cw), 4))
            bar = 34 * p[val] / pmax
            row.append(f'<rect x="{x + 9:.1f}" y="{cy - 6 - bar:.1f}" width="{cw - 18}" height="{max(bar, 1.5):.1f}" '
                       f'rx="1.5" fill="{hexc(PICK) if val == pick else "#aab1bd"}"/>')
            if val == pick:
                row.append(text(x + cw / 2, cy - 11 - bar, f"{p[val]:.2f}", 11, 700, hexc(PICK), "middle"))
            sx[val] = x
        elif kind == "sel":
            row.append(f'<rect x="{x:.1f}" y="{cy + 4}" width="{cw}" height="38" rx="6" fill="{INK}"/>')
            row.append(text(x + cw / 2, cy + 28, "<SEL>", 12, 700, "#fff", "middle"))
            sel_x = x + cw / 2
        x += cw + 4
    out_x = min(sel_x - S / 2, x0 + wdt - S)
    xs = [x0, x0 + (out_x - x0) / 2, out_x]
    # row 1: input -> candidate pool -> selected mask
    q = " ".join(query_words(ex["query"]))
    out.append(text(xs[0], y0 + 18, "Query", 12, 600, MUTED))
    out.append(text(xs[0] + 44, y0 + 18, f"“{q}”", 13, 500, INK, italic=True))
    ims = [img, pool_view(img, pool, ng), paint(img, [pool[pick]], [PICK], dim=0.25)]
    for xi, im in zip(xs, ims):
        out.append(image_tag(round(xi, 1), yi, S, S, to_png(im, box, (2 * S, 2 * S)), 6))
        out.append(f'<rect x="{xi:.1f}" y="{yi}" width="{S}" height="{S}" rx="6" fill="none" stroke="{LINE}"/>')
    mid = (xs[0] + S + xs[1]) / 2
    out.append(arrow(xs[0] + S + 10, yi + S / 2, xs[1] - 12, yi + S / 2))
    out.append(text(mid, yi + S / 2 - 10, "SAM 3 (frozen)", 12, 600, INK, "middle"))
    out.append(text(mid, yi + S / 2 + 20, "grid + text prompts", 11, 400, MUTED, "middle"))
    mid2 = (xs[1] + S + xs[2]) / 2
    out.append(arrow(xs[1] + S + 10, yi + S / 2, xs[2] - 12, yi + S / 2, MUTED, 1.4, "4 3"))
    out.append(text(mid2, yi + S / 2 - 10, "the MLLM selects", 12, 600, INK, "middle"))
    out.append(text(mid2, yi + S / 2 + 20, "no mask decoder", 11, 400, MUTED, "middle"))
    out.append(text(xs[1] + S / 2, yi + S + 17, f"{n} candidate masks", 12, 600, INK, "middle"))
    lx = xs[1] + S / 2 - 70
    out.append(f'<rect x="{lx}" y="{yi + S + 26}" width="9" height="9" rx="2" fill="#9cc3e6"/>')
    out.append(text(lx + 13, yi + S + 35, "point grid", 11, 400, MUTED))
    out.append(f'<rect x="{lx + 78}" y="{yi + S + 26}" width="9" height="9" rx="2" fill="none" '
               f'stroke="{hexc(TEXT)}" stroke-width="2"/>')
    out.append(text(lx + 91, yi + S + 35, "SAM 3 text", 11, 400, MUTED))
    out.append(text(xs[2] + S / 2, yi + S + 17, "selected mask", 12, 600, hexc(PICK), "middle"))
    # row 2: the MLLM's input sequence
    bh = cy + 46 + 34 - by
    out.append(f'<rect x="{x0}" y="{by}" width="{wdt}" height="{bh}" rx="10" fill="#f5f6f8" stroke="{LINE}"/>')
    out.append(text(x0 + 14, by + 20, "MLLM input sequence", 12, 700, INK))
    out.append(text(x0 + 14 + text_width("MLLM input sequence", 12, 700) + 8, by + 20,
                    "Qwen3-VL-4B, 64.2M trainable parameters", 12, 400, MUTED))
    out.append(text(x0 + 14, by + 37, "one token per candidate mask; bars: its selection score (softmax over all)",
                    11, 400, MUTED))
    out.extend(row)
    labels = ["query", "image", f"mask tokens ({n})"]
    for (a, b), lab in zip(groups, labels):
        out.append(f'<path d="M{a:.1f},{cy + 54} v5 H{b:.1f} v-5" fill="none" stroke="{MUTED}" stroke-width="1"/>')
        out.append(text((a + b) / 2, cy + 73, lab, 11, 500, MUTED, "middle"))
    px = sx[pick] + ch / 2
    out.append(f'<path d="M{sel_x - 10:.1f},{cy + 3} C{sel_x - 30:.1f},{cy - 66} {px + 50:.1f},{cy - 76} '
               f'{px + 14:.1f},{cy - 54}" fill="none" stroke="{hexc(PICK)}" stroke-width="2" marker-end="url(#ahp)"/>')
    # candidates enter as tokens; <SEL> returns the selected mask
    out.append(arrow(xs[1] + S / 2, yi + S + 42, xs[1] + S / 2, by - 2, MUTED, 1.4))
    ux = min(sel_x + 12, xs[2] + S - 12)
    out.append(arrow(ux, cy + 2, ux, yi + S + 24, hexc(PICK), 2))
    return "".join(out), bh + by - y0


# ---------------------------------------------------------------- (b) three kinds of answers
def answer_panel(ex: dict, kind: str, x0: float, y0: float, pw: float, ph: float) -> str:
    img = load(ex)
    h, w = img.shape[:2]
    pool = [dec(r) for r in ex["pool"]]
    out = []
    title = {"part": "Part", "multi": "Several targets", "none": "No target"}[kind]
    out.append(text(x0, y0 + 14, title, 13, 700, INK))
    lines = wrap(f"“{' '.join(query_words(ex['query']))}”", 12, pw, 500)[:2]
    for k, ln in enumerate(lines):
        out.append(text(x0, y0 + 33 + 15 * k, ln, 12, 500, INK, italic=True))
    iy = y0 + 56
    lg = np.asarray(ex["logits"])
    if kind == "part":
        m = pool[ex["pick"]]
        focus = bbox(dec(ex["gt_object"]) | m)
        arr = paint(img, [m], [PICK], dim=0.2)
        badge, bcol = f"argmax: 1 mask, IoU {ex['pick_iou']:.2f}", PICK
    elif kind == "multi":
        ms = [pool[i] for i in ex["chosen"]]
        focus = bbox(np.logical_or.reduce(ms))
        arr = paint(img, ms, SET_COLORS * 4, dim=0.2)
        badge, bcol = f"σ ≥ {ex['t']:.2f}: {len(ms)} masks, IoU {ex['iou']:.2f}", SET_COLORS[1]
    else:
        best = int(np.argmax(lg))
        focus = bbox(pool[best])
        arr = paint(img, [], [], dim=0.3, dashed=(pool[best], (40, 40, 40)))
        badge, bcol = f"best σ = {1 / (1 + np.exp(-lg[best])):.2f} < {ex['t']:.2f}: empty", (60, 60, 60)
    box = crop_box(h, w, focus, pw / ph, margin=0.3, min_frac=0.6)
    out.append(image_tag(round(x0, 1), iy, round(pw, 1), ph, to_png(arr, box, (int(2 * pw), int(2 * ph))), 6))
    out.append(f'<rect x="{x0:.1f}" y="{iy}" width="{pw:.1f}" height="{ph}" rx="6" fill="none" stroke="{LINE}"/>')
    if kind == "none":
        out.append(f'<rect x="{x0 + pw / 2 - 54:.1f}" y="{iy + ph - 44}" width="108" height="32" rx="16" '
                   f'fill="#fff" fill-opacity="0.94" stroke="{INK}" stroke-width="1.5"/>')
        out.append(text(x0 + pw / 2, iy + ph - 23, "∅  no target", 13, 700, INK, "middle"))
    out.append(text(x0, iy + ph + 17, badge, 11, 600, hexc(bcol)))
    out.append(text(x0, iy + ph + 32, STATS[kind], 11, 400, MUTED))
    return "".join(out)


def build(a: dict, hero_i: int, part_i: int, multi_i: int, none_i: int) -> str:
    defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>'
            '<marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{hexc(PICK)}"/></marker></defs>')
    left, lw = 14, 830
    rx = left + lw + 28
    hero_svg, hh = hero(a["part"][hero_i], left, 30, lw, H)
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>',
            text(left, 22, "(a) Ask, don’t draw: every candidate mask is a token, and the MLLM points at one", 15, 700),
            hero_svg,
            f'<line x1="{rx - 14}" y1="12" x2="{rx - 14}" y2="{H - 12}" stroke="{LINE}"/>',
            text(rx, 22, "(b) One read-out: parts, groups, absent targets", 15, 700)]
    gap = 14
    pw = (W - rx - 14 - 2 * gap) / 3
    ph = H - 34 - 56 - 44 - 12
    for k, (kind, ex) in enumerate([("part", a["part"][part_i]), ("multi", a["multi"][multi_i]),
                                    ("none", a["no_target"][none_i])]):
        body.append(answer_panel(ex, kind, rx + k * (pw + gap), 34, pw, ph))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}">{defs}{"".join(body)}</svg>')


def render(svg_path: Path, pdf_path: Path, png_path: Path | None) -> None:
    js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
          f"'{svg_path}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:{{width:{W},"
          f"height:{H}}},deviceScaleFactor:2}});await p.setContent(`<!doctype html><html><head><style>@font-face{{"
          f"font-family:Inter;src:url(file://{INTER[400]})}}@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0}}"
          f"</style></head><body>${{svg}}</body></html>`);"
          + (f"await p.screenshot({{path:'{png_path}',clip:{{x:0,y:0,width:{W},height:{H}}}}});" if png_path else "")
          + f"await p.pdf({{path:'{pdf_path}',width:'{W}px',height:'{H}px',printBackground:true,pageRanges:'1'}});"
          "await b.close();})();")
    subprocess.run(["node", "-e", js], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ROOT / "results/figures/teaser_assets.json"))
    ap.add_argument("--hero", type=int, default=0)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--multi", type=int, default=0)
    ap.add_argument("--none", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "teaser"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    a = json.loads(Path(args.assets).read_text())
    svg = build(a, args.hero, args.part, args.multi, args.none)
    out = Path(args.out)
    out.with_suffix(".svg").write_text(svg)
    render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None)
    print(out.with_suffix(".pdf"), "assets", a["config"])


if __name__ == "__main__":
    main()
