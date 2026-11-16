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
    "av": r"a_v", "gv": r"g_v", "qv": r"q_v",
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
    # query and its phrases
    out.append(rect(300, y, 290, 128, "#fff", PINK, 1.8, 12))
    out.append(text(314, y + 26, "Query", F_TITLE, 700, PINK))
    out.append(text(314, y + 52, "“Locate the handle of the mug.”", 16, 500, INK, italic=True))
    c1, w1 = chip(314, y + 82, "the handle of the mug", "#fff", PINK, 14, 600, PINK, 8)
    c2, _ = chip(314 + w1 + 6, y + 82, "the mug", "#fff", PINK, 14, 600, PINK, 8)
    out += [c1, c2]
    # the candidate pool, built by SAM 3 from the image and the phrases
    out.append(rect(604, y, 312, 128, "#fff", BLUE, 1.8, 12))
    out.append(text(618, y + 26, "Candidates", F_TITLE, 700, BLUE)
               + text(618 + text_width("Candidates ", F_TITLE, 700), y + 26, "frozen SAM 3", F_SMALL, 500, MUTED))
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
    out.append(f'<rect x="618" y="{y + 97}" width="12" height="12" rx="3" fill="none" stroke="{BLUE}" stroke-width="2.2"/>'
               + text(636, y + 108, "point grid", F_SMALL, 400, MUTED)
               + f'<rect x="730" y="{y + 97}" width="12" height="12" rx="3" fill="none" stroke="{PINK}" stroke-width="2.2"/>'
               + text(748, y + 108, "query phrases", F_SMALL, 400, MUTED))
    out.append(arrow([(590, y + 94), (603, y + 94)], PINK, 2))
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
    out.append(text(350, bar + 44, "MLLM + LoRA", 25, 700, "#fff", "middle"))
    cx_ = 664
    out.append(rect(cx_, bar + 8, 254, 54, "#fff", "none", 0, 10, op=0.93))
    out += attention_icon(cx_ + 8, bar + 12, 4.6)
    out.append(text(cx_ + 66, bar + 31, "region-restricted", 16, 600, INK))
    out.append(text(cx_ + 66, bar + 51, "attention", 16, 600, INK))
    # selection head on <SEL> and the mask tokens' states
    hx, hy, hw, hh, tab = 380, 176, 550, 52, 150
    out.append(rect(hx, hy, hw, hh, "#fff", PINK, 2.2, 12))
    out.append(f'<path d="M{hx + 12},{hy} H{hx + tab} V{hy + hh} H{hx + 12} a12,12 0 0 1 -12,-12 V{hy + 12} '
               f'a12,12 0 0 1 12,-12 z" fill="{PINK}"/>')
    out.append(text(hx + tab / 2, hy + 33, "Selection head", 17, 700, "#fff", "middle"))
    ex_ = hx + tab + (hw - tab) / 2
    out.append(M("head", ex_, hy + 34, 20, INK, "middle"))
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
    by, x = 96, 640
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
    out.append(arrow([(632, by + 22), (ax + 126, by + 22)], PINK))
    out.append(text((632 + ax + 126) / 2, by + 13, "argmax", F_SMALL, 600, PINK, "middle"))
    out.append(img_tag(ax, 30, 120, 120, scene(img, pool[pick], obj, 240), 10))
    out.append(rect(ax, 30, 120, 120, "none", PINK, 2.4, 10))
    out.append(text(ax - 12, 84, "Answer", F_TITLE, 700, PINK, "end") + M("mstar", ax - 12, 110, 19, PINK, "end"))
    return out


def tree_inset(M) -> list[str]:
    # the nesting of the candidates: image -> objects -> parts (the handle's path is highlighted)
    out = [dashed_box(20, 24, 290, 160, "Containment tree")]
    nodes = {"r": (165, 56), "mug": (82, 98), "glass": (165, 98), "plate": (248, 98),
             "m1": (42, 142), "m2": (82, 142), "h": (122, 142), "g1": (165, 142), "p1": (222, 142), "p2": (274, 142)}
    edges = [("r", "mug"), ("r", "glass"), ("r", "plate"), ("mug", "m1"), ("mug", "m2"), ("mug", "h"), ("glass", "g1"),
             ("plate", "p1"), ("plate", "p2")]
    path = {("r", "mug"), ("mug", "h")}
    for a, b in edges:
        on = (a, b) in path
        out.append(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}" '
                   f'stroke="{PINK if on else FRAME}" stroke-width="{3 if on else 2.2}"/>')
    for k, (x, y) in nodes.items():
        col = PINK if k == "h" else BLUE if k in ("mug", "r") else "#9db7dc"
        out.append(f'<circle cx="{x}" cy="{y}" r="{10 if k in ("h", "mug", "r") else 8}" fill="{col}" stroke="#fff" '
                   f'stroke-width="2"/>')
    out.append(text(181, 61, "image", F_SMALL, 600, MUTED))
    out.append(text(66, 103, "mug", F_SMALL, 600, BLUE, "end") + text(180, 103, "glass", F_SMALL, 500, MUTED)
               + text(263, 103, "plate", F_SMALL, 500, MUTED))
    out.append(text(122, 172, "handle", F_SMALL, 700, PINK, "middle"))
    return out


def attention_icon(x, y, cell) -> list[str]:
    # the attention pattern as a small matrix (image | text | mask tokens); details in the attention figure
    n_img, n_txt, n_m = 4, 3, 3
    n = n_img + n_txt + n_m
    own = {(7, 1), (7, 2), (8, 0), (8, 1), (9, 2), (9, 3)}
    out = []
    for r in range(n):
        for c in range(n):
            if r < n_img + n_txt:
                on, col = c <= r, GREEN if c < n_img else PINK
            else:
                on = c >= n_img or (r, c) in own
                col = GREEN if c < n_img else PINK if c < n_img + n_txt else BLUE
            out.append(f'<rect x="{x + c * cell:.1f}" y="{y + r * cell:.1f}" width="{cell - 0.8:.1f}" height="{cell - 0.8:.1f}" '
                       f'rx="1" fill="{col if on else "#e9ecf1"}"/>')
    return out


def vector(x, y, n, color, cell=13) -> str:
    return "".join(f'<rect x="{x + k * cell}" y="{y}" width="{cell - 2}" height="{cell + 3}" rx="2.5" fill="{color}" '
                   f'fill-opacity="{0.45 + 0.5 * ((k * 7 + 3) % 5) / 4:.2f}"/>' for k in range(n))


def tile_appearance(img, m, side, cells=6) -> str:
    # the image tokens under the mask: token grid over a crop, covered cells tinted
    h, w = m.shape
    crop = square(bbox(m), w, h, margin=0.55)
    a = np.asarray(Image.fromarray(img).crop(crop).resize((2 * side, 2 * side), Image.LANCZOS)).astype(float)
    cov = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).crop(crop).resize((cells, cells), Image.BOX)) > 12
    c = 2 * side // cells
    g = np.array([47, 143, 107])
    for r in range(cells):
        for k in range(cells):
            blk = a[r * c:(r + 1) * c, k * c:(k + 1) * c]
            blk[:] = 0.45 * blk + 0.55 * g if cov[r, k] else 0.55 * blk + 0.45 * 255
    for k in range(1, cells):
        a[k * c - 1:k * c + 1, :] = 255
        a[:, k * c - 1:k * c + 1] = 255
    return png_b64(Image.fromarray(a.clip(0, 255).astype(np.uint8)))


def tile_geometry(img, m, obj, side) -> tuple[str, tuple]:
    # where the mask lies: the object crop with the mask and its box
    h, w = m.shape
    crop = square(bbox(obj), w, h, margin=0.12)
    a = img.astype(float) * 0.5 + 255 * 0.5
    a = np.where(m[..., None], 0.35 * a + 0.65 * np.array([59, 111, 182]), a)
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8)).crop(crop).resize((2 * side, 2 * side), Image.LANCZOS)
    x0, y0, x1, y1 = bbox(m)
    f = side / (crop[2] - crop[0])
    return png_b64(im), ((x0 - crop[0]) * f, (y0 - crop[1]) * f, (x1 - x0) * f, (y1 - y0) * f)


def tokenizer_inset(ex, img, M) -> list[str]:
    pool = [dec(r) for r in ex["pool"]]
    m, obj = pool[ex["pick"]], dec(ex["gt_object"])
    x0, y0 = 950, 24
    out = [dashed_box(x0, y0, 436, 636, "Mask tokenizer")]
    T, tx = 104, x0 + 138                                   # tile size and left edge
    ys = [y0 + 40, y0 + 160, y0 + 280]
    # the candidate and the three things read from it
    th = ys[1] + T / 2 - 48
    out.append(img_tag(x0 + 16, th, 96, 96, cutout(img, m, 192), 8) + rect(x0 + 16, th, 96, 96, "none", PINK, 2.4, 8))
    out.append(M("mv", x0 + 64, th + 124, 21, MUTED, "middle"))
    trunk = tx - 14
    out.append(f'<path d="M{x0 + 114},{th + 48} H{trunk} M{trunk},{ys[0] + T / 2} V{ys[2] + T / 2}" fill="none" '
               f'stroke="{FRAME}" stroke-width="2"/>')
    geo, box = tile_geometry(img, m, obj, T)
    rows = [("Appearance", "av", GREEN, 7), ("Geometry", "gv", BLUE, 4), ("Source", "qv", PINK, 3)]
    lx, xc = tx + T + 16, x0 + 414
    for k, ((title, key, c, n), y) in enumerate(zip(rows, ys)):
        out.append(arrow([(trunk, y + T / 2), (tx - 3, y + T / 2)], FRAME, 2))
        if k == 0:
            out.append(img_tag(tx, y, T, T, tile_appearance(img, m, T), 8))
        elif k == 1:
            out.append(img_tag(tx, y, T, T, geo, 8))
            out.append(f'<rect x="{tx + box[0]:.1f}" y="{y + box[1]:.1f}" width="{box[2]:.1f}" height="{box[3]:.1f}" '
                       f'fill="none" stroke="{BLUE}" stroke-width="2.4" stroke-dasharray="5 3"/>')
            for a_, b_ in (((tx + 26, y + 14), (tx + 14, y + 30)), ((tx + 26, y + 14), (tx + 38, y + 30))):
                out.append(f'<line x1="{a_[0]}" y1="{a_[1]}" x2="{b_[0]}" y2="{b_[1]}" stroke="{MUTED}" stroke-width="1.8"/>')
            for (px_, py_), col in (((tx + 26, y + 14), MUTED), ((tx + 14, y + 30), MUTED), ((tx + 38, y + 30), BLUE)):
                out.append(f'<circle cx="{px_}" cy="{py_}" r="5.5" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
        else:
            # point grid or text prompt, with the rank and the score of a text instance
            for r in range(4):
                for q in range(4):
                    out.append(f'<circle cx="{tx + 14 + q * 9}" cy="{y + 34 + r * 9}" r="2.4" fill="{FRAME}"/>')
            out.append(text(tx + 27, y + 88, "grid", 14, 500, MUTED, "middle"))
            out.append(rect(tx + 58, y + 24, 40, 26, PINK, PINK, 1, 6) + text(tx + 78, y + 42, "text", 14, 700, "#fff", "middle"))
            out.append(chip(tx + 58, y + 58, "#1", "#fff", PINK, 13, 700, PINK, 6)[0])
            out.append(f'<rect x="{tx + 58}" y="{y + 88}" width="40" height="6" rx="3" fill="{LINE}"/>'
                       f'<rect x="{tx + 58}" y="{y + 88}" width="37" height="6" rx="3" fill="{PINK}"/>')
        out.append(rect(tx, y, T, T, "none", c, 2.2, 8))
        out.append(text(lx, y + 30, title, 18, 700, c))
        out.append(M(key, lx + text_width(title + " ", 18, 700) + 2, y + 31, 21, INK))
        out.append(vector(lx, y + 52, n, c))
        end = lx + n * 13 + 4
        out.append(f'<path d="M{end},{y + 60} H{xc}" fill="none" stroke="{c}" stroke-width="2"/>')
    # concatenation -> MLP -> one token
    cy = ys[2] + T + 44
    cells = sum(r[3] for r in rows)
    vx = x0 + 218 - cells * 13 / 2
    out.append(f'<path d="M{xc},{ys[0] + 60} V{cy + 8} H{vx + cells * 13 + 6}" fill="none" stroke="{FRAME}" stroke-width="2" '
               f'marker-end="url(#{marker_id(FRAME)})"/>')
    off = vx
    for _, _, c, n in rows:
        out.append(vector(off, cy, n, c))
        off += n * 13
    mx_ = x0 + 218
    out.append(arrow([(mx_, cy + 22), (mx_, cy + 46)], BLUE))
    out.append(trapezoid(mx_, cy + 48, 200, 150, 40, BLUE, "MLP"))
    out.append(arrow([(mx_, cy + 90), (mx_, cy + 114)], BLUE))
    out.append(rect(mx_ - 56, cy + 116, 112, 50, TINT[BLUE], BLUE, 2.6, 10))
    out.append(img_tag(mx_ - 48, cy + 122, 38, 38, cutout(img, m, 76), 5))
    out.append(M("xv", mx_ + 24, cy + 150, 23, INK, "middle"))
    out.append(text(mx_ + 70, cy + 147, "mask token", F_BODY, 600, BLUE))
    return out


def build(ex, img, M, grid_ids, text_ids) -> str:
    defs = ('<defs><linearGradient id="llm" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#2f8f6b"/>'
            '<stop offset="0.45" stop-color="#b0306f"/><stop offset="1" stop-color="#3b6fb6"/></linearGradient>'
            + markers([INK, BLUE, PINK, GREEN, FRAME]) + "</defs>")
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>', defs]
    body += tree_inset(M) + inputs(ex, img, M, grid_ids, text_ids) + stack(ex, img, M)
    body += tokenizer_inset(ex, img, M)
    # insets hang off what they detail
    body.append(f'<path d="M880,{402 + D} L950,330" fill="none" stroke="{FRAME}" stroke-width="1.4" stroke-dasharray="5 4"/>')
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
