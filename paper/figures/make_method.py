"""Method figure (two-column), bottom-up: inputs (image, query, SAM 3 candidate pool) -> one tokenizer per stream ->
image, query and mask tokens -> Qwen3-VL-4B + LoRA -> selection head -> one answer mask; insets detail the mask
tokenizer (appearance, geometry, source), region-restricted attention and the containment tree.

Real data: the PACO test image, candidate masks, pool indices and the model's selection scores come from
results/figures/overview_assets_mug_handle.json (scripts/fig_assets.py, pool_soft_fam). Equations are typeset by MathJax
(tex2svg.js; mathjax-full@3.2.2 under MJ_DIR/node_modules).
usage: MJ_DIR=<dir with node_modules> python paper/figures/make_method.py --png
"""
import argparse
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_overview import INTER, bbox, dec, math_svg, png_b64, square, text_width, typeset  # noqa: E402
from make_teaser import esc  # noqa: E402

W, H = 1400, 684
D = 52                                         # room between the MLLM and the selection head for the read-out
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1f2633", "#5b6474", "#cfd5de"
BLUE, GREEN, PINK = "#3b6fb6", "#2f8f6b", "#b0306f"      # mask stream, image stream, text stream
FRAME = "#8d96a3"
TINT = {BLUE: "#eef3fb", PINK: "#fbeff5", GREEN: "#eaf5ef"}
F_STAGE, F_TITLE, F_BODY, F_SMALL, F_MATH = 21, 18, 16, 15, 18
F_IN = 16.5                                    # body text of the inset panels

TEX = {
    "phi": r"\phi(x)",
    "par1": r"\mathrm{parent}(v)=\arg\min_{u}\,|m_u|",
    "par2": r"\mathrm{s.t.}\ \ |m_u\cap m_v|\geq 0.8\,|m_v|",
    "mv": r"m_v",
    "app": r"a_v=\textstyle\sum_j \tilde{c}_{v,j}\,e_j",
    "geo": r"g_v\in\mathbb{R}^{9}",
    "src": r"q_v\in\mathbb{R}^{5}",
    "tok": r"\hat{x}_v=\mathrm{MLP}\big([\,a_v;\,g_v;\,q_v\,]\big)",
    "xv": r"\hat{x}_v",
    "ej": r"e_j",
    "head": r"s_v=\lambda\,\cos\!\big(\psi_q(h_{\langle\mathrm{SEL}\rangle}),\,\psi_k(z_v)\big)+\beta",
    "argmax": r"v^{\star}=\arg\max_{v}\,s_v",
    "mstar": r"m_{v^{\star}}",
    "psiq": r"\psi_q",
    "psik": r"\psi_k",
    "hsel": r"h_{\langle\mathrm{SEL}\rangle}",
    "zv": r"z_v",
    "lstar": r"\ell^{\star}",
    "bias": r"b_{\ell,\,\mathrm{rel}(u,v)}",
}


# ---------------------------------------------------------------- svg primitives
def text(x, y, s, size=F_BODY, weight=400, fill=INK, anchor="start", italic=False) -> str:
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{st} xml:space="preserve">{esc(s)}</text>')


def rect(x, y, w, h, fill="#fff", stroke=LINE, sw=1.5, r=10, dash=None, op=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{op}"' if op is not None else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{o} '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def marker_id(color: str) -> str:
    return "ah" + color.lstrip("#")


def markers(colors) -> str:
    # fixed-size filled heads (user units), so every arrow ends in the same clear triangle
    return "".join(f'<marker id="{marker_id(c)}" viewBox="0 0 12 12" refX="10.5" refY="6" markerWidth="12" '
                   f'markerHeight="12" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
                   f'<path d="M0,0.5 L12,6 L0,11.5 z" fill="{c}"/></marker>' for c in colors)


def arrow(pts, color=INK, sw=2.2, dash=None) -> str:
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{ds} '
            f'marker-end="url(#{marker_id(color)})"/>')


def curve(x0, y0, c1, c2, x1, y1, color, sw=2.2) -> str:
    return (f'<path d="M{x0},{y0} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {x1},{y1}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker_id(color)})"/>')


def img_tag(x, y, w, h, b64, r=6, kind="png") -> str:
    cid = f"c{abs(hash((x, y, w, h))) % 10**9}"
    return (f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}"/></clipPath>'
            f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="none" '
            f'clip-path="url(#{cid})" href="data:image/{kind};base64,{b64}"/>')


def chip(x, y, s, fill, color="#fff", size=F_SMALL, weight=600, stroke=None, pad=10) -> tuple[str, float]:
    w = text_width(s, size, weight) + 2 * pad
    h = size + 12
    st = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="{h / 2}" fill="{fill}"{st}/>'
            + text(x + pad, y + size + 2.5, s, size, weight, color)), w


class Math:
    def __init__(self, tex: dict):
        self.t = tex

    def width(self, key, size=F_MATH) -> float:
        return self.t[key]["w_ex"] * 0.442 * size

    def __call__(self, key, x, baseline, size=F_MATH, color=INK, anchor="start") -> str:
        w = self.width(key, size)
        x0 = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
        return math_svg(self.t[key], x0, baseline, size, color)[0]


# ---------------------------------------------------------------- real image patches
def cutout(img, m, side, board=True) -> str:
    h, w = m.shape
    crop = square(bbox(m), w, h, margin=0.18, min_side=24)
    yy, xx = np.mgrid[:h, :w]
    s = max(1, (crop[2] - crop[0]) // 6)
    bg = np.where(((yy // s) + (xx // s)) % 2 == 0, 242, 228)[..., None] if board else 255
    out = np.where(m[..., None], img, bg).astype(np.uint8)
    return png_b64(Image.fromarray(out).crop(crop).resize((side, side), Image.LANCZOS))


def scene(img, m, region, side, color=(176, 48, 111), fill=0.55) -> str:
    h, w = m.shape
    crop = square(bbox(region), w, h, margin=0.12)
    x = img.astype(float)
    if m is not None:
        x = np.where(m[..., None], (1 - fill) * x + fill * np.array(color), x)
        e = m ^ np.pad(m, 3, mode="edge")[6:, 3:-3] | m ^ np.pad(m, 3, mode="edge")[3:-3, 6:]
        x[e] = color
    return png_b64(Image.fromarray(x.clip(0, 255).astype(np.uint8)).crop(crop).resize((side, side), Image.LANCZOS))


# ---------------------------------------------------------------- the three stages
def trapezoid(cx, y, wb, wt, h, color, label) -> str:
    # a tokenizer: wide at the bottom (input), narrow at the top (tokens)
    pts = f"{cx - wb / 2},{y + h} {cx + wb / 2},{y + h} {cx + wt / 2},{y} {cx - wt / 2},{y}"
    return (f'<polygon points="{pts}" fill="{color}" stroke="none"/>'
            + text(cx, y + h / 2 + 6.5, label, 18, 700, "#fff", "middle"))


def dashed_box(x, y, w, h, title, sub="") -> str:
    tw = text_width(title, F_TITLE + 1, 700) + (text_width(sub, F_BODY, 500) + 8 if sub else 0) + 18
    out = rect(x, y, w, h, "none", FRAME, 1.4, 14, dash="6 5")
    out += f'<rect x="{x + 12}" y="{y - 12}" width="{tw:.1f}" height="24" fill="#fff"/>'
    out += text(x + 21, y + 6, title, F_TITLE + 1, 700, INK)
    if sub:
        out += text(x + 21 + text_width(title + " ", F_TITLE + 1, 700), y + 6, sub, F_BODY, 500, MUTED)
    return out


def tokens(x, y, n, color, w=22, h=26, gap=6) -> tuple[list[str], float]:
    out = [rect(x + k * (w + gap), y, w, h, TINT[color], color, 2, 5) for k in range(n)]
    return out, x + n * (w + gap) - gap


def inputs(ex, img, M, grid_ids, text_ids) -> list[str]:
    pool = [dec(r) for r in ex["pool"]]
    obj = dec(ex["gt_object"])
    out = [dashed_box(20, 452 + D, 910, 168, "Inputs")]
    y = 470 + D
    # image
    out.append(rect(36, y, 250, 128, "#fff", GREEN, 1.8, 12))
    shown = obj | pool[grid_ids[1]]
    crop = square(bbox(shown), img.shape[1], img.shape[0], margin=0.12)
    out.append(img_tag(48, y + 12, 104, 104, png_b64(Image.fromarray(img).crop(crop).resize((208, 208), Image.LANCZOS)), 8))
    out.append(text(166, y + 30, "Image", F_TITLE, 700, GREEN))
    out.append(text(166, y + 54, "PACO test", F_SMALL, 400, MUTED))
    # query and its phrases
    out.append(rect(300, y, 290, 128, "#fff", PINK, 1.8, 12))
    out.append(text(314, y + 26, "Query", F_TITLE, 700, PINK))
    out.append(text(314, y + 52, "“Locate the handle of the mug.”", 16, 500, INK, italic=True))
    out.append(text(314, y + 80, "phrases", F_SMALL, 600, MUTED) + M("phi", 314 + text_width("phrases ", F_SMALL, 600),
                                                                  y + 80, 16, MUTED))
    c1, w1 = chip(314, y + 90, "the handle of the mug", "#fff", PINK, 14, 600, PINK, 8)
    c2, _ = chip(314 + w1 + 6, y + 90, "the mug", "#fff", PINK, 14, 600, PINK, 8)
    out += [c1, c2]
    # the candidate pool, built by SAM 3 from the image and the phrases
    out.append(rect(604, y, 312, 128, "#fff", BLUE, 1.8, 12))
    out.append(text(618, y + 26, "Candidates", F_TITLE, 700, BLUE)
               + text(618 + text_width("Candidates ", F_TITLE, 700), y + 26, "from SAM 3 (frozen)", F_SMALL, 500, MUTED))
    t, x = 46, 618
    for k, i in enumerate(grid_ids[:3] + text_ids):
        c = BLUE if k < 3 else PINK
        out.append(img_tag(x, y + 40, t, t, cutout(img, pool[i], 2 * t), 5))
        out.append(rect(x, y + 40, t, t, "none", c, 2.2, 5))
        if k >= 3:
            out.append(chip(x + 2, y + 42, "#1", PINK, "#fff", 11, 700, pad=4)[0])
        x += t + 8 if k != 2 else t + 26
        if k == 2:
            out.append(text(x - 17, y + 70, "…", 18, 700, MUTED, "middle"))
    out.append(text(618, y + 108, f"N = {len(pool)}:", F_SMALL, 700, INK)
               + text(618 + text_width(f"N = {len(pool)}: ", F_SMALL, 700), y + 108, "point grid + text phrases (#rank)",
                      F_SMALL, 400, MUTED))
    out.append(arrow([(590, y + 104), (603, y + 104)], PINK, 2))
    out.append(arrow([(161, y + 128), (161, y + 134), (900, y + 134), (900, y + 130)], GREEN, 1.8))
    return out


def stack(ex, img, M) -> list[str]:
    out = []
    # tokenizers over their inputs, tokens, the MLLM
    out.append(trapezoid(161, 380 + D, 230, 180, 44, GREEN, "Vision encoder"))
    out.append(trapezoid(445, 380 + D, 230, 180, 44, PINK, "Text tokenizer"))
    out.append(trapezoid(760, 380 + D, 240, 190, 44, BLUE, "Mask tokenizer"))
    for cx, c in ((161, GREEN), (445, PINK), (760, BLUE)):
        out.append(arrow([(cx, 468 + D), (cx, 427 + D)], c))
    ty = 336 + D
    t1, e1 = tokens(161 - 82, ty, 6, GREEN)
    t2, e2 = tokens(445 - 112, ty, 5, PINK)
    sel_x = e2 + 6
    t3, e3 = tokens(760 - 96, ty, 7, BLUE)
    out += t1 + t2 + t3
    out.append(rect(sel_x, ty, 64, 26, PINK, PINK, 1, 5) + text(sel_x + 32, ty + 18.5, "<SEL>", 13, 700, "#fff", "middle"))
    out.append(M("ej", e1 + 8, ty + 20, 16, GREEN) + M("xv", e3 + 8, ty + 20, 16, BLUE))
    for cx, c in ((161, GREEN), (445, PINK), (760, BLUE)):
        out.append(arrow([(cx, 379 + D), (cx, 364 + D)], c, 2))
    bar = 252 + D
    out.append(rect(20, bar, 910, 70, "url(#llm)", "none", 0, 16))
    out.append(text(475, bar + 31, "Qwen3-VL-4B + LoRA", 22, 700, "#fff", "middle"))
    out.append(text(475, bar + 56, "one pass over image, query and mask tokens; LoRA rank r = 16", F_SMALL, 500,
                    "#fff", "middle"))
    # selection head on <SEL> and the mask tokens' states
    hx, hy, hw, hh, tab = 380, 158, 550, 70, 112
    out.append(rect(hx, hy, hw, hh, "#fff", PINK, 2.2, 12))
    out.append(f'<path d="M{hx + 12},{hy} H{hx + tab} V{hy + hh} H{hx + 12} a12,12 0 0 1 -12,-12 V{hy + 12} '
               f'a12,12 0 0 1 12,-12 z" fill="{PINK}"/>')
    out.append(text(hx + tab / 2, hy + 31, "Selection", 17, 700, "#fff", "middle"))
    out.append(text(hx + tab / 2, hy + 53, "head", 17, 700, "#fff", "middle"))
    ex_ = hx + tab + (hw - tab) / 2
    out.append(M("head", ex_, hy + 29, 19, INK, "middle"))
    out.append(M("argmax", ex_, hy + 58, 19, INK, "middle"))
    # read-out: the chosen layer's states go through one MLP each before the cosine
    for cx, c, key, state, side in ((sel_x + 32, PINK, "psiq", "hsel", 1), (838, BLUE, "psik", "zv", -1)):
        py_, pw_, ph_ = hy + hh + 21, 124, 34
        out.append(arrow([(cx, bar), (cx, py_ + ph_ + 1)], c))
        # a small projection block in the tokenizers' shape: wide at the input, narrow at the output
        top = pw_ - 28
        out.append(f'<polygon points="{cx - pw_ / 2},{py_ + ph_} {cx + pw_ / 2},{py_ + ph_} {cx + top / 2},{py_} '
                   f'{cx - top / 2},{py_}" fill="{TINT[c]}" stroke="{c}" stroke-width="2.2" stroke-linejoin="round"/>')
        tw_ = text_width("MLP ", 18, 700) + M.width(key, 21)
        out.append(text(cx - tw_ / 2, py_ + 24, "MLP", 18, 700, c)
                   + M(key, cx - tw_ / 2 + text_width("MLP ", 18, 700), py_ + 24.5, 21, INK))
        out.append(arrow([(cx, py_), (cx, hy + hh + 2)], c))
        out.append(M(state, cx + side * (pw_ / 2 + 12), py_ + 24, 23, c, "start" if side > 0 else "end"))
    mid, lab = (sel_x + 32 + 838) / 2 + 20, "layer "
    w_ = text_width(lab, 19, 600) + M.width("lstar", 23)
    out.append(text(mid - w_ / 2, hy + hh + 45, lab, 19, 600, MUTED)
               + M("lstar", mid - w_ / 2 + text_width(lab, 19, 600), hy + hh + 46, 23, INK))
    # scores over candidates and the answer
    pool = [dec(r) for r in ex["pool"]]
    pick = ex["pick"]
    lg = np.asarray(ex["logits"])
    p = np.exp(lg - lg.max())
    p /= p.sum()
    by, x = 88, 640
    out.append(text(x, by - 26, "selection scores over the pool", F_SMALL, 600, MUTED))
    for i in (83, 65, 34, 257, pick):
        c = PINK if i == pick else (BLUE if i < ex["n_grid"] else PINK)
        out.append(img_tag(x, by, 44, 44, cutout(img, pool[i], 88), 5))
        out.append(rect(x, by, 44, 44, "none", c, 2.6 if i == pick else 1.6, 5))
        hgt = max(2, 40 * p[i] / p[pick])
        out.append(f'<rect x="{x + 10}" y="{by - 6 - hgt:.1f}" width="24" height="{hgt:.1f}" rx="2" fill="{c}" '
                   f'fill-opacity="{1 if i == pick else 0.5}"/>')
        if i == pick:
            out.append(text(x + 22, by - 12 - hgt, f"{p[i]:.2f}", 14, 700, PINK, "middle"))
        x += 54
    out.append(arrow([(760, hy - 2), (760, by + 48)], PINK))
    obj = dec(ex["gt_object"])
    ax = 412
    out.append(arrow([(634, by + 22), (ax + 126, by + 22)], PINK))
    out.append(img_tag(ax, 24, 120, 120, scene(img, pool[pick], obj, 240), 10))
    out.append(rect(ax, 24, 120, 120, "none", PINK, 2.4, 10))
    out.append(text(ax - 12, 70, "Answer", F_TITLE, 700, PINK, "end") + M("mstar", ax - 12, 96, 18, PINK, "end"))
    out.append(text(ax - 12, 120, "one mask", F_SMALL, 400, MUTED, "end"))
    return out


def tree_inset(M) -> list[str]:
    out = [dashed_box(20, 24, 300, 204, "Containment tree")]
    nodes = {"r": (170, 62), "a": (118, 102), "b": (222, 102), "c": (84, 142), "d": (152, 142)}
    for a, b in (("r", "a"), ("r", "b"), ("a", "c"), ("a", "d")):
        out.append(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}" stroke="{MUTED}" '
                   f'stroke-width="2.6"/>')
    for k, (x, y) in nodes.items():
        out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{PINK if k == "d" else BLUE}" stroke="#fff" stroke-width="2"/>')
    out.append(text(138, 107, "mug", F_SMALL, 600, BLUE) + text(172, 147, "handle", F_SMALL, 600, PINK))
    out.append(M("par1", 170, 184, 19, INK, "middle"))
    out.append(M("par2", 170, 213, 19, INK, "middle"))
    return out


def tokenizer_inset(ex, img, M) -> list[str]:
    pool = [dec(r) for r in ex["pool"]]
    m = pool[ex["pick"]]
    x0, y0 = 950, 24
    out = [dashed_box(x0, y0, 436, 400, "Mask tokenizer", "one token per candidate")]
    out.append(img_tag(x0 + 16, y0 + 76, 84, 84, cutout(img, m, 168), 8))
    out.append(rect(x0 + 16, y0 + 76, 84, 84, "none", PINK, 2.2, 8))
    out.append(text(x0 + 58, y0 + 64, "candidate", F_IN, 600, MUTED, "middle"))
    out.append(M("mv", x0 + 58, y0 + 184, 19, MUTED, "middle"))
    rows = [("app", "Appearance", "MLLM image tokens in the mask", GREEN),
            ("geo", "Geometry", "box, area, depth in the tree", BLUE),
            ("src", "Source", "grid or SAM 3 text: rank, score", BLUE)]
    rx, rw, rh = x0 + 126, 298, 68
    ys = [y0 + 34, y0 + 112, y0 + 190]
    trunk = rx - 14
    out.append(f'<path d="M{x0 + 100},{y0 + 118} H{trunk} M{trunk},{ys[0] + rh / 2} V{ys[2] + rh / 2}" fill="none" '
               f'stroke="{BLUE}" stroke-width="2"/>')
    for (key, title, desc, c), y in zip(rows, ys):
        out.append(arrow([(trunk, y + rh / 2), (rx - 2, y + rh / 2)], BLUE, 2))
        out.append(rect(rx, y, rw, rh, "#fff", c, 1.8, 10) + f'<rect x="{rx}" y="{y}" width="7" height="{rh}" rx="3.5" fill="{c}"/>')
        out.append(text(rx + 18, y + 27, title, 17.5, 700, c))
        out.append(M(key, rx + rw - 10, y + 28, 18, INK, "end"))
        out.append(text(rx + 18, y + 53, desc, F_IN, 400, MUTED))
    my = ys[2] + rh + 26
    out.append(arrow([(rx + rw / 2, ys[2] + rh + 1), (rx + rw / 2, my - 1)], BLUE))
    out.append(text(rx + rw / 2 + 8, ys[2] + rh + 19, "concatenate", 15.5, 600, BLUE))
    out.append(rect(rx + 30, my, rw - 60, 36, BLUE, BLUE, 1, 10) + text(rx + rw / 2, my + 24.5, "MLP + RMS norm", 17.5, 700, "#fff", "middle"))
    out.append(M("tok", x0 + 16, my + 72, 19, INK))
    out.append(text(x0 + 16 + M.width("tok", 19) + 14, my + 71, "position: mask centroid", F_IN, 400, MUTED))
    return out


def attention_inset(M) -> list[str]:
    x0, y0 = 950, 404 + D
    out = [dashed_box(x0, y0, 436, 216, "Region-restricted attention", "every layer")]
    cell, n_img, n_txt, n_m = 14, 4, 3, 3
    mx, my = x0 + 16, y0 + 50
    n = n_img + n_txt + n_m
    own = {(7, 1), (7, 2), (8, 0), (8, 1), (9, 2), (9, 3)}
    for r in range(n):
        for c in range(n):
            if r < n_img + n_txt:
                on, col = c <= r, GREEN if c < n_img else PINK
            else:
                on = c >= n_img or (r, c) in own
                col = GREEN if c < n_img else PINK if c < n_img + n_txt else BLUE
            fill = col if on else "#ffffff"
            op = 0.85 if on and col != BLUE else 0.45 if on else 1
            out.append(f'<rect x="{mx + c * cell}" y="{my + r * cell}" width="{cell - 1.5}" height="{cell - 1.5}" rx="2" '
                       f'fill="{fill}" fill-opacity="{op}" stroke="{LINE}" stroke-width="0.8"/>')
    for k, lab in ((n_img / 2, "img"), (n_img + n_txt / 2, "txt"), (n_img + n_txt + n_m / 2, "mask")):
        out.append(text(mx + k * cell, my - 7, lab, 14, 600, MUTED, "middle"))
    out.append(f'<rect x="{mx + (n_img + n_txt) * cell - 1}" y="{my + (n_img + n_txt) * cell - 1}" width="{n_m * cell + 0.5}" '
               f'height="{n_m * cell + 0.5}" fill="none" stroke="{BLUE}" stroke-width="2" rx="3"/>')
    lx = mx + n * cell + 16
    keys = [(GREEN, "a mask token sees only its", "own region and the query"),
            (None, "image and query tokens", "never see the mask tokens"),
            (BLUE, "between masks: learned bias", None)]
    for i, (c, l1, l2) in enumerate(keys):
        yy = my + 10 + i * 50
        k = f'fill="{c}" fill-opacity="0.85"' if c else f'fill="#fff" stroke="{FRAME}" stroke-width="1.5"'
        out.append(f'<rect x="{lx}" y="{yy - 11}" width="12" height="12" rx="3" {k}/>')
        out.append(text(lx + 20, yy, l1, F_IN, 400, INK))
        if l2:
            out.append(text(lx + 20, yy + 21, l2, F_IN, 400, INK))
        else:
            out.append(M("bias", lx + 20, yy + 23, 16.5, INK)
                       + text(lx + 26 + M.width("bias", 16.5), yy + 21, "by tree relation", F_IN, 400, INK))
    return out


def build(ex, img, M, grid_ids, text_ids) -> str:
    defs = ('<defs><linearGradient id="llm" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#2f8f6b"/>'
            '<stop offset="0.45" stop-color="#b0306f"/><stop offset="1" stop-color="#3b6fb6"/></linearGradient>'
            + markers([INK, BLUE, PINK, GREEN]) + "</defs>")
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>', defs]
    body += tree_inset(M) + inputs(ex, img, M, grid_ids, text_ids) + stack(ex, img, M)
    body += tokenizer_inset(ex, img, M) + attention_inset(M)
    # insets hang off what they detail
    body.append(f'<path d="M880,{402 + D} L950,330" fill="none" stroke="{FRAME}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    body.append(f'<path d="M930,{300 + D} L950,{470 + D}" fill="none" stroke="{FRAME}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}">{"".join(body)}</svg>')


def render(svg_path: Path, pdf_path: Path, png_path: Path | None) -> None:
    faces = "".join(f"@font-face{{font-family:Inter;font-weight:{w};src:url(file://{p})}}" for w, p in INTER.items())
    js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
          f"'{svg_path}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:{{width:{W},"
          f"height:{H}}},deviceScaleFactor:2}});await p.setContent(`<!doctype html><html><head><style>{faces}"
          f"@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0}}</style></head><body>${{svg}}</body></html>`);"
          "await p.evaluate(()=>document.fonts.ready);"
          + (f"await p.screenshot({{path:'{png_path}',clip:{{x:0,y:0,width:{W},height:{H}}}}});" if png_path else "")
          + f"await p.pdf({{path:'{pdf_path}',width:'{W}px',height:'{H}px',printBackground:true,pageRanges:'1'}});"
          "await b.close();})();")
    subprocess.run(["node", "-e", js], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ROOT / "results/figures/overview_assets_mug_handle.json"))
    ap.add_argument("--example", type=int, default=6)
    ap.add_argument("--grid", default="83,15,65,34")
    ap.add_argument("--text", default="256,257")
    ap.add_argument("--out", default=str(HERE / "method"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    ex = json.loads(Path(args.assets).read_text())["examples"][args.example]
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(ex["image_jpeg_b64"]))).convert("RGB"))
    M = Math(typeset(list(TEX.values())))
    M.t = {k: M.t[v] for k, v in TEX.items()}
    svg = build(ex, img, M, [int(i) for i in args.grid.split(",")], [int(i) for i in args.text.split(",")])
    out = Path(args.out)
    out.with_suffix(".svg").write_text(svg)
    render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None)
    print(out.with_suffix(".pdf"), "image", ex["image_id"], "pick IoU", round(ex["pick_iou"], 2))


if __name__ == "__main__":
    main()
