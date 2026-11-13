"""Teaser figure (two-column): (a) candidate masks enter the MLLM as tokens and <SEL> points at one; (b) one model
answers part queries, multi-target expressions and no-target expressions.

Every pixel shown is a real model output: results/figures/teaser_assets.json (scripts/teaser_assets.py, pool_soft_fam
on test queries). Scores are the model's own: (a) softmax of s_v over the whole pool (the part decision is argmax);
(b) sigmoid(s_v) against the gRefCOCO set threshold t chosen on val.
usage: python paper/figures/make_teaser.py --hero 1 --part 0 --multi 2 --none 6 --png   (the paper version)
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

W, H = 1400, 590
# type sizes (px on the 1400-px canvas; at \linewidth 1 px ~ 0.35 pt): title 7.4 pt, labels 6 pt, small 5.6 pt
F_TITLE, F_HEAD, F_LABEL, F_SMALL, F_CHIP = 26, 23, 21, 19, 20
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1f2633", "#5b6474", "#c9ced6"
FRAME, WORD, IMG = "#8d96a3", "#fbeff5", "#2f8f6b"     # dashed group outline; text-stream tint; image stream (as Fig. 2)
PICK = (176, 48, 111)                                  # the selected mask (as in the overview figure)
GRID, TEXT = (59, 111, 182), (176, 48, 111)            # candidate provenance: point grid / SAM 3 text (as Fig. 2)
SET_COLORS = [PICK, (59, 111, 182), (47, 143, 107), (96, 76, 170)]
# test-split results of the 4B model, Tab. 1: gains over the best <=4B baseline (PACO part, gRefCOCO testA), N-acc testA
STATS = {"part": "PACO +14.5", "multi": "gRefCOCO +8.0", "none": "N-acc 94.0%"}


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
    x = img.astype(float) * 0.75 + 255 * 0.25
    for i in sorted(range(ng), key=lambda i: -int(pool[i].sum())):
        c = np.array(Image.new("HSV", (1, 1), (int(rng.integers(256)), 160, 245)).convert("RGB").getpixel((0, 0)))
        x = np.where(pool[i][..., None], 0.45 * x + 0.55 * c, x)
        x[edge(pool[i], 1)] = 255
    for i in range(ng, len(pool)):
        x[edge(pool[i], 2)] = TEXT
    return x.clip(0, 255).astype(np.uint8)


def token_row(ex, img, box, pool, k, ch, width):
    # chips (kind, payload, width); returns None if they do not fit in `width`
    words = query_words(ex["query"])
    chips = [("word", wd, text_width(wd, F_CHIP, 500) + 10) for wd in words] + [("gap", None, 6)]
    chips += [("tile", t, ch - 6) for t in range(2)] + [("dots", None, 18), ("gap", None, 6)]
    shown = shown_tokens(ex, k)
    for j, i in enumerate(shown):
        if j and i != shown[j - 1] + 1:
            chips.append(("dots", None, 18))
        chips.append(("mask", i, ch))
    if shown[-1] != len(pool) - 1:
        chips.append(("dots", None, 18))
    chips += [("gap", None, 6), ("sel", None, 66)]
    total = sum(c[2] for c in chips) + 4 * len(chips)
    return chips if total <= width else None


def hero(ex: dict, x0: float, y0: float, wdt: float) -> str:
    img = load(ex)
    h, w = img.shape[:2]
    pool = [dec(r) for r in ex["pool"]]
    n, ng, pick = len(pool), ex["n_grid"], ex["pick"]
    obj = dec(ex["gt_object"]) if "gt_object" in ex else pool[pick]
    box = crop_box(h, w, bbox(obj | pool[pick]), 1.0, margin=0.25)
    out = []
    # the token row decides where <SEL> sits; the selected-mask image goes right above it
    ch = 50
    chips = next(c for k in (7, 6, 5, 4, 3, 2) if (c := token_row(ex, img, box, pool, k, ch, wdt - 28)))
    S = 160
    yi = y0 + 40
    by = yi + S + 74
    cy = by + 122
    x = x0 + wdt - 14 - (sum(c[2] for c in chips) + 4 * (len(chips) - 1))
    p = softmax(np.asarray(ex["logits"]))
    pmax = max(p[c[1]] for c in chips if c[0] == "mask")
    groups, start, sx, sel_x = [], None, {}, None
    row = []
    mid_y = cy + ch / 2 + F_CHIP * 0.36                 # text baseline centred on the chip row
    for kind, val, cw in chips:
        if kind == "word":
            start = x if start is None else start
            row.append(f'<rect x="{x:.1f}" y="{cy + 5}" width="{cw:.1f}" height="{ch - 10}" rx="6" fill="{WORD}" '
                       f'stroke="{hexc(PICK)}" stroke-width="1.6"/>')
            row.append(text(x + cw / 2, mid_y, val, F_CHIP, 500, INK, "middle"))
        elif kind == "gap":
            if start is not None:
                groups.append((start, x - 4))
                start = None
        elif kind == "tile":
            start = x if start is None else start
            t = (box[2] - box[0]) // 3
            tx, ty = box[0] + t * (0, 2)[val], box[1] + t * (0, 1)[val]
            row.append(image_tag(round(x, 1), cy + 3, cw, cw, to_png(img, (tx, ty, tx + t, ty + t), (2 * int(cw),) * 2), 4))
            row.append(f'<rect x="{x:.1f}" y="{cy + 3}" width="{cw}" height="{cw}" rx="4" fill="none" stroke="{IMG}" '
                       f'stroke-width="2"/>')
        elif kind == "dots":
            start = x if start is None else start
            row.append(text(x + cw / 2, mid_y, "⋯", 20, 700, MUTED, "middle"))
        elif kind == "mask":
            start = x if start is None else start
            col = PICK if val == pick else GRID if val < ng else TEXT
            row.append(f'<rect x="{x - 1:.1f}" y="{cy - 1}" width="{cw + 2}" height="{cw + 2}" rx="6" fill="#fff" '
                       f'stroke="{hexc(col)}" stroke-width="{3.5 if val == pick else 2}"/>')
            row.append(image_tag(round(x + 1, 1), cy + 1, cw - 2, cw - 2, cutout(img, pool[val], 2 * cw), 5))
            bar = 40 * p[val] / pmax
            row.append(f'<rect x="{x + 10:.1f}" y="{cy - 7 - bar:.1f}" width="{cw - 20}" height="{max(bar, 2):.1f}" '
                       f'rx="2" fill="{hexc(PICK) if val == pick else hexc(col)}" fill-opacity="{1 if val == pick else 0.45}"/>')
            if val == pick:
                row.append(text(x + 4, cy - 7 - bar / 2 + F_SMALL * 0.36, f"{p[val]:.2f}", F_SMALL, 700, hexc(PICK), "end"))
            sx[val] = x
        elif kind == "sel":
            row.append(f'<rect x="{x:.1f}" y="{cy + 3}" width="{cw}" height="{ch - 6}" rx="7" fill="{hexc(PICK)}"/>')
            row.append(text(x + cw / 2, mid_y, "<SEL>", F_CHIP, 700, "#fff", "middle"))
            sel_x = x + cw / 2
        x += cw + 4
    out_x = min(sel_x - S / 2, x0 + wdt - S)
    xs = [x0, x0 + (out_x - x0) / 2, out_x]
    # row 1: input -> candidate pool -> selected mask
    q = " ".join(query_words(ex["query"]))
    out.append(text(xs[0], y0 + 22, "Query", F_LABEL, 600, MUTED))
    out.append(text(xs[0] + text_width("Query", F_LABEL, 600) + 10, y0 + 22, f"“{q}”", F_LABEL, 500, INK, italic=True))
    ims = [img, pool_view(img, pool, ng), paint(img, [pool[pick]], [PICK], line=3)]
    for xi, im, fr in zip(xs, ims, (IMG, hexc(GRID), hexc(PICK))):
        out.append(image_tag(round(xi, 1), yi, S, S, to_png(im, box, (2 * S, 2 * S)), 7))
        out.append(f'<rect x="{xi:.1f}" y="{yi}" width="{S}" height="{S}" rx="7" fill="none" stroke="{fr}" '
                   f'stroke-width="2.5"/>')
    for (xa, xb), top, sub, col, dash in (((xs[0], xs[1]), "SAM 3", "frozen", hexc(GRID), None),
                                          ((xs[1], xs[2]), "MLLM picks", "no decoder", hexc(PICK), "5 4")):
        mid = (xa + S + xb) / 2
        out.append(arrow(xa + S + 10, yi + S / 2, xb - 12, yi + S / 2, col, 2.5, dash))
        tw = text_width(top, F_LABEL - 1, 700) + 18
        out.append(f'<rect x="{mid - tw / 2:.1f}" y="{yi + S / 2 - 40}" width="{tw:.1f}" height="{F_LABEL + 12}" '
                   f'rx="{(F_LABEL + 12) / 2}" fill="{col}"/>')
        out.append(text(mid, yi + S / 2 - 14, top, F_LABEL - 1, 700, "#fff", "middle"))
        out.append(text(mid, yi + S / 2 + 26, sub, F_SMALL, 400, MUTED, "middle"))
    out.append(text(xs[1] + S / 2, yi + S + 24, f"{n} candidate masks", F_LABEL, 600, INK, "middle"))
    lw1, lw2 = text_width("point grid", F_SMALL, 400), text_width("SAM 3 text", F_SMALL, 400)
    lx = xs[1] + S / 2 - (lw1 + lw2 + 50) / 2
    ly = yi + S + 47
    out.append(f'<rect x="{lx:.1f}" y="{ly - 11}" width="12" height="12" rx="2" fill="#9cc3e6" stroke="{hexc(GRID)}" '
               f'stroke-width="1.5"/>')
    out.append(text(lx + 17, ly, "point grid", F_SMALL, 400, MUTED))
    lx2 = lx + 17 + lw1 + 16
    out.append(f'<rect x="{lx2:.1f}" y="{ly - 11}" width="12" height="12" rx="2" fill="none" '
               f'stroke="{hexc(TEXT)}" stroke-width="2.5"/>')
    out.append(text(lx2 + 17, ly, "SAM 3 text", F_SMALL, 400, MUTED))
    out.append(text(xs[2] + S / 2, yi + S + 24, "selected mask", F_LABEL, 700, hexc(PICK), "middle"))
    # row 2: the MLLM's input sequence
    bh = cy + ch + 44 - by
    out.append(f'<rect x="{x0}" y="{by}" width="{wdt}" height="{bh}" rx="14" fill="none" stroke="{FRAME}" '
               f'stroke-width="1.4" stroke-dasharray="6 5"/>')
    t1, t2 = "MLLM input sequence", "Qwen3-VL-4B + LoRA"
    tw = text_width(t1, F_LABEL, 700) + 18
    out.append(f'<rect x="{x0 + 12}" y="{by - 13}" width="{tw:.1f}" height="26" fill="#fff"/>')
    out.append(text(x0 + 21, by + 7, t1, F_LABEL, 700, INK))
    out.append(text(x0 + wdt - 56, by + 42, t2, F_SMALL, 600, MUTED, "end"))
    out.append(text(x0 + 16, by + 42, "one token per candidate mask; bar = its selection score", F_SMALL, 400, MUTED))
    out.extend(row)
    labels = ["query", "image", f"mask tokens ({n})"]
    for (a, b), lab in zip(groups, labels):
        out.append(f'<path d="M{a:.1f},{cy + ch + 8} v6 H{b:.1f} v-6" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
        out.append(text((a + b) / 2, cy + ch + 32, lab, F_SMALL, 500, MUTED, "middle"))
    px = sx[pick] + ch / 2
    out.append(f'<path d="M{sel_x - 12:.1f},{cy + 2} C{sel_x - 34:.1f},{cy - 74} {px + 56:.1f},{cy - 86} '
               f'{px + 16:.1f},{cy - 62}" fill="none" stroke="{hexc(PICK)}" stroke-width="2.5" marker-end="url(#ahp)"/>')
    # candidates enter as tokens; <SEL> returns the selected mask
    out.append(arrow(xs[1] + S / 2, yi + S + 56, xs[1] + S / 2, by - 2, MUTED, 1.8))
    ux = min(sel_x + 14, xs[2] + S - 14)
    out.append(arrow(ux, cy + 1, ux, yi + S + 32, hexc(PICK), 2.5))
    return "".join(out)


# ---------------------------------------------------------------- (b) three kinds of answers
def answer_panel(ex: dict, kind: str, x0: float, y0: float, pw: float, ph: float) -> str:
    img = load(ex)
    h, w = img.shape[:2]
    pool = [dec(r) for r in ex["pool"]]
    out = []
    title = {"part": "Part", "multi": "Multi-target", "none": "No target"}[kind]
    accent = {"part": hexc(PICK), "multi": hexc(SET_COLORS[1]), "none": INK}[kind]
    out.append(f'<rect x="{x0:.1f}" y="{y0 - 2}" width="{text_width(title, F_HEAD, 700) + 20:.1f}" height="{F_HEAD + 12}" '
               f'rx="{(F_HEAD + 12) / 2}" fill="{accent}"/>')
    out.append(text(x0 + 10, y0 + F_HEAD + 1, title, F_HEAD, 700, "#fff"))
    lines = wrap(f"“{' '.join(query_words(ex['query']))}”", F_SMALL, pw, 500)[:2]
    for k, ln in enumerate(lines):
        out.append(text(x0, y0 + 52 + 23 * k, ln, F_SMALL, 500, INK, italic=True))
    iy = y0 + 84
    lg = np.asarray(ex["logits"])
    if kind == "part":
        m = pool[ex["pick"]]
        focus, frac = bbox(m), 0.5
        arr = paint(img, [m], [PICK], line=3)
        badge, bcol = f"1 mask, IoU {ex['pick_iou']:.2f}", PICK
    elif kind == "multi":
        ms = [pool[i] for i in ex["chosen"]]
        focus, frac = bbox(np.logical_or.reduce(ms)), 0.6
        arr = paint(img, ms, SET_COLORS * 4, line=3)
        badge, bcol = f"{len(ms)} masks, IoU {ex['iou']:.2f}", SET_COLORS[1]
    else:
        best = int(np.argmax(lg))
        focus, frac = bbox(pool[best]), 1.0             # whole image: the scene shows why nothing matches
        arr = paint(img, [], [], dim=0.15, line=3, dashed=(pool[best], (40, 40, 40)))
        badge, bcol = f"max σ {1 / (1 + np.exp(-lg[best])):.2f} < {ex['t']:.2f}", (60, 60, 60)
    box = crop_box(h, w, focus, pw / ph, margin=0.3, min_frac=frac)
    out.append(image_tag(round(x0, 1), iy, round(pw, 1), ph, to_png(arr, box, (int(2 * pw), int(2 * ph))), 7))
    out.append(f'<rect x="{x0:.1f}" y="{iy}" width="{pw:.1f}" height="{ph}" rx="7" fill="none" stroke="{accent}" '
               f'stroke-width="2.5"/>')
    if kind == "none":
        bw = text_width("∅  no target", F_LABEL, 700) + 30
        out.append(f'<rect x="{x0 + pw / 2 - bw / 2:.1f}" y="{iy + ph - 50}" width="{bw:.1f}" height="38" rx="19" '
                   f'fill="#fff" fill-opacity="0.95" stroke="{INK}" stroke-width="1.8"/>')
        out.append(text(x0 + pw / 2, iy + ph - 25, "∅  no target", F_LABEL, 700, INK, "middle"))
    out.append(text(x0, iy + ph + 26, badge, 19, 700, hexc(bcol)))
    out.append(text(x0, iy + ph + 50, f"▲ {STATS[kind]}" if "+" in STATS[kind] else STATS[kind], 18, 600,
                    IMG if "+" in STATS[kind] else MUTED))
    return "".join(out)


def badge(x, base, letter, color) -> str:
    return (f'<circle cx="{x + 13}" cy="{base - F_TITLE * 0.36:.1f}" r="14" fill="{color}"/>'
            + text(x + 13, base - F_TITLE * 0.36 + 6.5, letter, 19, 700, "#fff", "middle"))


def build(a: dict, hero_i: int, part_i: int, multi_i: int, none_i: int) -> str:
    # fixed-size filled heads (user units), as in the method figure
    defs = ('<defs><marker id="ah" viewBox="0 0 12 12" refX="10.5" refY="6" markerWidth="12" markerHeight="12" '
            'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0.5 L12,6 L0,11.5 z" '
            'fill="context-stroke"/></marker>'
            '<marker id="ahp" viewBox="0 0 12 12" refX="10.5" refY="6" markerWidth="12" markerHeight="12" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0.5 L12,6 L0,11.5 z" '
            f'fill="{hexc(PICK)}"/></marker></defs>')
    left, lw = 14, 800
    rx = left + lw + 30
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>',
            badge(left, 30, "a", INK), text(left + 34, 30, "Ask, don’t draw: masks are tokens, the MLLM picks one", F_TITLE, 700),
            hero(a["part"][hero_i], left, 40, lw),
            f'<line x1="{rx - 15}" y1="14" x2="{rx - 15}" y2="{H - 14}" stroke="{LINE}" stroke-width="1.2"/>',
            badge(rx, 30, "b", INK), text(rx + 34, 30, "One read-out, three kinds of answer", F_TITLE, 700)]
    gap = 14
    pw = (W - rx - 12 - 2 * gap) / 3
    ph = H - 48 - 84 - 64
    for k, (kind, ex) in enumerate([("part", a["part"][part_i]), ("multi", a["multi"][multi_i]),
                                    ("none", a["no_target"][none_i])]):
        body.append(answer_panel(ex, kind, rx + k * (pw + gap), 48, pw, ph))
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
