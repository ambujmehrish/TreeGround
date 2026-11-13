"""Method figure (two-column): (1) SAM 3 proposes a query-conditioned candidate pool, (2) every candidate becomes one
mask token, (3) the MLLM reads image, query and mask tokens and selects one.

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

W, H = 1400, 676
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1f2633", "#5b6474", "#cfd5de"
BLUE, GREEN, PINK = "#3b6fb6", "#2f8f6b", "#b0306f"      # mask stream, image stream, text stream
FRAME = "#8d96a3"
TINT = {BLUE: "#eef3fb", PINK: "#fbeff5", GREEN: "#eaf5ef"}
F_STAGE, F_TITLE, F_BODY, F_SMALL, F_MATH = 21, 18, 16, 15, 18
COLS = [(14, 452, BLUE, "1", "Propose candidates", "SAM 3, frozen"),
        (466, 862, BLUE, "2", "Mask tokenization", ""),
        (876, 1386, PINK, "3", "The MLLM selects", "")]

TEX = {
    "phi": r"\phi(x)",
    "parent": r"\mathrm{parent}(v)=\arg\min_{u}\,|m_u|\ \ \mathrm{s.t.}\ |m_u\cap m_v|\geq 0.8\,|m_v|",
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
def stage_frames() -> list[str]:
    # thin dashed group outlines on white; the title sits in a gap of the top edge
    out = []
    for x0, x1, c, num, title, sub in COLS:
        out.append(rect(x0, 30, x1 - x0, H - 40, "none", FRAME, 1.4, 16, dash="6 5"))
        tw = text_width(title, F_STAGE - 2, 700) + (text_width(sub, F_SMALL, 500) + 10 if sub else 0) + 24
        out.append(f'<rect x="{x0 + 14}" y="18" width="{tw:.1f}" height="26" fill="#fff"/>')
        out.append(text(x0 + 26, 38, title, F_STAGE - 2, 700, INK))
        if sub:
            out.append(text(x0 + 26 + text_width(title + " ", F_STAGE - 2, 700), 38, sub, F_SMALL, 500, MUTED))
    return out


def stage1(ex, img, M, grid_ids, text_ids) -> list[str]:
    out = []
    pool = [dec(r) for r in ex["pool"]]
    obj = dec(ex["gt_object"])
    # input image and query
    S = 150
    shown = obj | pool[grid_ids[1]]
    out.append(img_tag(30, 72, S, S, png_b64(Image.fromarray(img).crop(square(bbox(shown), img.shape[1], img.shape[0], margin=0.12))
                               .resize((2 * S, 2 * S), Image.LANCZOS)), 10))
    out.append(text(30 + S / 2, 72 + S + 22, "image", F_SMALL, 600, MUTED, "middle"))
    bx, by, bw = 196, 74, 242
    out.append(rect(bx, by, bw, 60, "#fff", LINE, 1.5, 12))
    out.append(text(bx + 12, by + 25, "“Locate the handle", 17, 500, INK, italic=True))
    out.append(text(bx + 12, by + 47, "of the mug.”", 17, 500, INK, italic=True))
    out.append(text(bx, by + 86, "phrases", F_SMALL, 600, MUTED) + M("phi", bx + text_width("phrases ", F_SMALL, 600),
                                                                      by + 86, 16, MUTED))
    c1, w1 = chip(bx, by + 96, "the handle of the mug", "#fff", PINK, F_SMALL, 600, PINK)
    c2, _ = chip(bx, by + 128, "the mug", "#fff", PINK, F_SMALL, 600, PINK)
    out += [c1, c2]
    # the two SAM 3 calls
    gy = 280
    out.append(rect(30, gy, 196, 64, BLUE, BLUE, 1, 12))
    out.append(text(128, gy + 26, "SAM 3 · point grid", F_BODY, 700, "#fff", "middle"))
    out.append(text(128, gy + 49, "32×32, query-free, ≤ 256", F_SMALL, 400, "#e3ebf7", "middle"))
    out.append(rect(242, gy, 196, 64, PINK, PINK, 1, 12))
    out.append(text(340, gy + 26, "SAM 3 · text", F_BODY, 700, "#fff", "middle"))
    out.append(text(340, gy + 49, "top 3 per phrase", F_SMALL, 400, "#f7e1ec", "middle"))
    out.append(arrow([(105, 72 + S + 30), (105, gy - 3)], BLUE))
    out.append(arrow([(330, by + 158), (330, gy - 3)], PINK))
    # the pool
    py = 380
    out.append(rect(30, py, 408, 132, "#fff", LINE, 1.5, 12))
    out.append(text(44, py + 25, "Candidate pool", F_TITLE, 700, INK))
    out.append(text(424, py + 25, f"N = {len(pool)}", F_BODY, 600, MUTED, "end"))
    t, gap, x = 58, 8, 44
    for i in grid_ids:
        out.append(img_tag(x, py + 40, t, t, cutout(img, pool[i], 2 * t), 6))
        out.append(rect(x, py + 40, t, t, "none", BLUE, 2.2, 6))
        x += t + gap
    out.append(text(x + 2, py + 76, "…", 20, 700, MUTED))
    x += 22
    for k, i in enumerate(text_ids):
        out.append(img_tag(x, py + 40, t, t, cutout(img, pool[i], 2 * t), 6))
        out.append(rect(x, py + 40, t, t, "none", PINK, 2.2, 6))
        out.append(chip(x + 2, py + 42, "#1", PINK, "#fff", 12, 700, pad=5)[0])
        x += t + gap
    out.append(f'<rect x="44" y="{py + 112}" width="12" height="12" rx="3" fill="none" stroke="{BLUE}" stroke-width="2.2"/>'
               + text(62, py + 123, "grid masks", F_SMALL, 400, MUTED))
    out.append(f'<rect x="170" y="{py + 112}" width="12" height="12" rx="3" fill="none" stroke="{PINK}" stroke-width="2.2"/>'
               + text(188, py + 123, "SAM 3 text instances (#rank)", F_SMALL, 400, MUTED))
    out.append(arrow([(128, gy + 64), (128, py - 3)], BLUE))
    out.append(arrow([(340, gy + 64), (340, py - 3)], PINK))
    # containment tree
    ty = 532
    out.append(rect(30, ty, 408, 122, "#fff", LINE, 1.5, 12))
    out.append(text(44, ty + 25, "Containment tree", F_TITLE, 700, INK))
    nodes = {"r": (370, ty + 14), "a": (340, ty + 34), "b": (400, ty + 34), "c": (322, ty + 52), "d": (358, ty + 52)}
    for p, c in (("r", "a"), ("r", "b"), ("a", "c"), ("a", "d")):
        out.append(f'<line x1="{nodes[p][0]}" y1="{nodes[p][1]}" x2="{nodes[c][0]}" y2="{nodes[c][1]}" stroke="{MUTED}" '
                   f'stroke-width="1.8"/>')
    for k, (x_, y_) in nodes.items():
        col = PINK if k == "d" else BLUE
        out.append(f'<circle cx="{x_}" cy="{y_}" r="7" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
    out.append(M("parent", 44, ty + 80, 14.5))
    out.append(text(44, ty + 104, "e.g. mug ⊃ handle; used between mask tokens", F_SMALL, 400, MUTED))
    return out


def stage2(ex, img, M) -> list[str]:
    out = []
    pool = [dec(r) for r in ex["pool"]]
    m = pool[ex["pick"]]
    x0 = 482
    # the candidate
    out.append(text(x0, 92, "candidate", F_SMALL, 600, MUTED) + M("mv", x0 + text_width("candidate ", F_SMALL, 600), 92, 16, MUTED))
    out.append(img_tag(x0, 102, 96, 96, cutout(img, m, 192), 8))
    out.append(rect(x0, 102, 96, 96, "none", PINK, 2.4, 8))
    # three descriptors
    cards = [("app", "Appearance", "what it shows: the MLLM's", "image tokens under the mask", GREEN),
             ("geo", "Geometry", "where it is: box, area,", "depth in the tree", BLUE),
             ("src", "Source", "where it came from: grid or", "SAM 3 text (rank, score)", BLUE)]
    cx, cw, ch = 604, 244, 92
    ys = [76, 178, 280]
    for (key, title, l1, l2, c), y in zip(cards, ys):
        out.append(rect(cx, y, cw, ch, "#fff", c, 2, 12))
        out.append(f'<rect x="{cx}" y="{y}" width="8" height="{ch}" rx="4" fill="{c}"/>')
        out.append(text(cx + 20, y + 26, title, F_BODY, 700, c))
        out.append(M(key, cx + cw - 12, y + 27, 17, INK, "end"))
        out.append(text(cx + 20, y + 54, l1, F_SMALL, 400, MUTED))
        out.append(text(cx + 20, y + 74, l2, F_SMALL, 400, MUTED))
    tx_ = cx - 18
    out.append(f'<path d="M{x0 + 98},150 H{tx_} M{tx_},{ys[0] + ch / 2} V{ys[2] + ch / 2}" fill="none" stroke="{BLUE}" '
               f'stroke-width="2.2"/>')
    for y in ys:
        out.append(arrow([(tx_, y + ch / 2), (cx - 3, y + ch / 2)], BLUE))
    my = 420
    out.append(arrow([(cx + cw / 2, ys[2] + ch + 2), (cx + cw / 2, my - 3)], BLUE))
    out.append(text(cx + cw / 2 + 10, ys[2] + ch + 30, "concatenate", F_SMALL, 600, BLUE))
    out.append(rect(cx + 14, my, cw - 28, 46, BLUE, BLUE, 1, 12))
    out.append(text(cx + cw / 2, my + 30, "MLP + RMS norm", F_BODY, 700, "#fff", "middle"))
    out.append(M("tok", cx + cw / 2, my + 80, 19, INK, "middle"))
    # the token
    out.append(arrow([(cx + cw / 2, my + 90), (cx + cw / 2, my + 116)], BLUE))
    tx = cx + cw / 2 - 52
    out.append(rect(tx, my + 120, 104, 48, "#fff", BLUE, 2.6, 10))
    out.append(img_tag(tx + 6, my + 125, 38, 38, cutout(img, m, 76), 5))
    out.append(M("xv", tx + 74, my + 152, 21, INK, "middle"))
    out.append(text(x0, my + 140, "position (mRoPE):", F_SMALL, 600, MUTED))
    out.append(text(x0, my + 159, "mask centroid", F_SMALL, 400, MUTED))
    # framing
    out.append(text(665, 648, "like a word token: content + position + type", F_SMALL, 600, MUTED, "middle"))
    return out


def stage3(ex, img, M) -> list[str]:
    out = []
    pool = [dec(r) for r in ex["pool"]]
    pick = ex["pick"]
    x0, x1 = 892, 1372
    # token sequence
    y = 78
    seq = [("img", GREEN)] * 3 + [("dots", None)] + [("txt", PINK)] * 3 + [("sel", PINK)] + \
          [("m", BLUE), ("m", BLUE), ("dots", None), ("m", BLUE), ("m", BLUE)]
    x, spans, cur, start = x0 + 4, {}, None, None
    for kind, c in seq:
        group = {"img": "image", "txt": "query", "sel": "query", "m": "mask", "dots": cur}[kind]
        if group != cur:
            if cur is not None:
                spans[cur] = (start, x - 6)
            cur, start = group, x
        if kind == "dots":
            out.append(text(x + 9, y + 26, "…", 18, 700, MUTED, "middle"))
            x += 22
        elif kind == "sel":
            out.append(rect(x, y + 4, 62, 30, PINK, PINK, 1, 7) + text(x + 31, y + 25, "<SEL>", 14, 700, "#fff", "middle"))
            x += 68
        else:
            out.append(rect(x, y + 4, 24, 30, TINT.get(c, "#fff"), c, 2, 6))
            x += 30
    spans[cur] = (start, x - 6)
    labels = {"image": "image tokens", "query": "query", "mask": "mask tokens (one pass)"}
    for g, (a, b) in spans.items():
        out.append(f'<path d="M{a},{y + 42} v6 H{b} v-6" fill="none" stroke="{MUTED}" stroke-width="1.3"/>')
        out.append(text((a + b) / 2, y + 68, labels[g], F_SMALL, 600, MUTED, "middle"))
    out.append(M("ej", spans["image"][0] + 2, y - 4, 15, GREEN) + M("xv", spans["mask"][1] - 2, y - 4, 15, BLUE, "end"))
    # the LLM
    ly = 172
    out.append(arrow([(1130, y + 76), (1130, ly - 3)], INK))
    out.append(rect(x0, ly, x1 - x0, 64, "url(#llm)", "none", 0, 14))
    out.append(text((x0 + x1) / 2, ly + 30, "Large language model", 20, 700, "#fff", "middle"))
    out.append(text((x0 + x1) / 2, ly + 52, "Qwen3-VL-4B + LoRA, 64.2M trainable parameters", F_SMALL, 500, "#ffffff", "middle"))
    # region-restricted attention: matrix + three short lines
    ay = 262
    out.append(text(x0, ay + 18, "Region-restricted attention", F_TITLE, 700, INK)
               + text(x0 + text_width("Region-restricted attention ", F_TITLE, 700), ay + 18, "(every layer)", F_SMALL, 400, MUTED))
    cell, n_img, n_txt, n_m = 15, 4, 3, 3
    mx, my = x0 + 6, ay + 52
    n = n_img + n_txt + n_m
    own = {(7, 1), (7, 2), (8, 0), (8, 1), (9, 2), (9, 3)}       # image cells under each mask (with its 1-cell margin)
    for r in range(n):
        for c in range(n):
            if r < n_img + n_txt:
                on = c <= r
                col = GREEN if c < n_img else PINK
            else:
                on = c >= n_img or (r, c) in own
                col = GREEN if c < n_img else PINK if c < n_img + n_txt else BLUE
            fill = col if on else "#ffffff"
            op = 0.85 if on and col != BLUE else 0.45 if on else 1
            out.append(f'<rect x="{mx + c * cell}" y="{my + r * cell}" width="{cell - 1.5}" height="{cell - 1.5}" rx="2" '
                       f'fill="{fill}" fill-opacity="{op}" stroke="{LINE}" stroke-width="0.8"/>')
    for k, lab in ((n_img / 2, "img"), (n_img + n_txt / 2, "txt"), (n_img + n_txt + n_m / 2, "mask")):
        out.append(text(mx + k * cell, my - 6, lab, 12, 600, MUTED, "middle"))
        out.append(text(mx - 4, my + k * cell + 4, lab, 12, 600, MUTED, "end") if False else "")
    out.append(f'<rect x="{mx + (n_img + n_txt) * cell - 1}" y="{my + (n_img + n_txt) * cell - 1}" width="{n_m * cell + 0.5}" '
               f'height="{n_m * cell + 0.5}" fill="none" stroke="{BLUE}" stroke-width="2.2" rx="3"/>')
    lx = mx + n * cell + 22
    lines = [(GREEN, "a mask token sees only its own", "image region, plus the query"),
             (PINK, "image and query tokens never", "see the mask tokens"),
             (BLUE, "between mask tokens: a learned", None)]
    for i, (c, l1, l2) in enumerate(lines):
        yy = my + 8 + i * 46
        key = (f'fill="{c}" fill-opacity="0.85"' if i != 1 else f'fill="#fff" stroke="{LINE}" stroke-width="1.5"')
        out.append(f'<rect x="{lx}" y="{yy - 11}" width="12" height="12" rx="3" {key}/>')
        out.append(text(lx + 20, yy, l1, F_SMALL, 400, INK))
        if l2:
            out.append(text(lx + 20, yy + 19, l2, F_SMALL, 400, INK))
        else:
            out.append(text(lx + 20, yy + 19, "bias", F_SMALL, 400, INK)
                       + M("bias", lx + 20 + text_width("bias ", F_SMALL, 400), yy + 20, 15, INK)
                       + text(lx + 20 + text_width("bias ", F_SMALL, 400) + M.width("bias", 15) + 4, yy + 19,
                              "by tree relation", F_SMALL, 400, INK))
    # selection head
    hy = 470
    out.append(arrow([(1130, ly + 64), (1130, ly + 80), (1376, ly + 80), (1376, hy + 30), (x1 + 2, hy + 30)], PINK)
               if False else "")
    out.append(rect(x0, hy, x1 - x0, 66, "#fff", PINK, 2.2, 12))
    out.append(text(x0 + 14, hy + 26, "Selection head", F_BODY, 700, PINK))
    out.append(M("head", x0 + 14, hy + 54, 18.5, INK))
    out.append(M("argmax", x1 - 14, hy + 27, 18.5, INK, "end"))
    out.append(arrow([(x1 - 22, ly + 64), (x1 - 22, hy - 3)], PINK))
    out.append(text(x1 - 30, ly + 100, "", F_SMALL))
    # scores over the shown candidates and the answer
    lg = np.asarray(ex["logits"])
    p = np.exp(lg - lg.max())
    p /= p.sum()
    by = 566
    out.append(arrow([(x0 + 300, hy + 66), (x0 + 300, by + 54)], PINK))
    out.append(text(x0, by + 8, "scores over the pool", F_SMALL, 600, MUTED))
    cand = [83, 65, 34, 257, pick]
    bx = x0
    for i in cand:
        c = PINK if i == pick else (BLUE if i < ex["n_grid"] else PINK)
        out.append(img_tag(bx, by + 46, 40, 40, cutout(img, pool[i], 80), 5))
        out.append(rect(bx, by + 46, 40, 40, "none", c, 2.4 if i == pick else 1.6, 5))
        hgt = max(2, 30 * p[i] / p[pick])
        out.append(f'<rect x="{bx + 8}" y="{by + 42 - hgt:.1f}" width="24" height="{hgt:.1f}" rx="2" fill="{c}" '
                   f'fill-opacity="{1 if i == pick else 0.55}"/>')
        if i == pick:
            out.append(text(bx + 20, by + 42 - hgt - 4, f"{p[i]:.2f}", 13, 700, PINK, "middle"))
        bx += 50
    out.append(arrow([(bx + 4, by + 66), (1262, by + 66)], PINK))
    obj = dec(ex["gt_object"])
    out.append(img_tag(1268, by - 2, 96, 96, scene(img, pool[pick], obj, 192), 10))
    out.append(rect(1268, by - 2, 96, 96, "none", PINK, 2.4, 10))
    out.append(text(1258, by + 14, "answer", F_SMALL, 700, PINK, "end") + M("mstar", 1258, by + 36, 17, PINK, "end"))
    return out


def build(ex, img, M, grid_ids, text_ids) -> str:
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>', "<defs>" + '<linearGradient id="llm" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#2f8f6b"/>'
            '<stop offset="0.45" stop-color="#b0306f"/><stop offset="1" stop-color="#3b6fb6"/></linearGradient>' + markers([INK, BLUE, PINK, GREEN]) + "</defs>"]
    body += stage_frames()
    body += stage1(ex, img, M, grid_ids, text_ids)
    body += stage2(ex, img, M)
    body += stage3(ex, img, M)
    # stage links
    body.append(arrow([(438, 446), (458, 446), (458, 150), (478, 150)], BLUE))
    body.append(arrow([(778, 564), (869, 564), (869, 100), (892, 100)], BLUE))
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
