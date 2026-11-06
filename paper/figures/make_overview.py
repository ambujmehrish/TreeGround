"""Method overview figure: overview.svg -> overview.pdf.

Equations are typeset by MathJax (tex2svg.js; `npm install mathjax-full@3.2.2` next to it, or set MJ_DIR) and embedded as
vector paths. Image slots use real assets when results/figures/overview_assets_mug_handle.json exists (written by
scripts/fig_assets.py on the cluster), otherwise drawn placeholders.
usage: python paper/figures/make_overview.py [--example N] [--pdf]
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 1500, 640
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED = "#1d2433", "#5b6475"
C = {"img": ("#fdebe1", "#e07a45"), "mask": ("#e3f0fa", "#2f7fb6"), "text": ("#e6f4e3", "#3f8f3a"),
     "sel": ("#fbe3ee", "#b0306f"), "llm": ("#eaf5e6", "#4d9a3f"), "panel": ("#f6f8fb", "#c9d2df"),
     "seg": ("#f1f2f5", "#3b4250")}

MATH = {
    "V": r"\mathcal{V}=\mathcal{G}\,\cup\,\mathcal{T}",
    "G": r"\mathcal{G}", "T": r"\mathcal{T}",
    "phi": r"\phi(x)",
    "parent": r"\displaystyle\mathrm{parent}(v)=\mathop{\mathrm{arg\,min}}_{u\,:\,|m_u\cap m_v|\,\geq\,0.8\,|m_v|}|m_u|",
    "cv": r"c_v", "mv": r"m_v",
    "pv": r"\displaystyle p_v=\frac{\sum_j c_{v,j}\,e_j}{\sum_j c_{v,j}}",
    "gv": r"g_v\in\mathbb{R}^{9}", "qv": r"q_v\in\mathbb{R}^{5}",
    "xv": r"\hat{x}_v=\mathrm{RMSNorm}\big(\mathrm{MLP}([\,p_v;\,g_v;\,q_v\,])\big)",
    "pos": r"\mathrm{pos}(\hat{x}_v)=(t_{\mathrm{end}},\,\bar{r}_v,\,\bar{c}_v)",
    "Rv": r"\mathcal{R}_v=\mathrm{dilate}\big(\mathrm{supp}\,c_v\big)",
    "bias": r"b_{\ell,\,\mathrm{rel}(v,u)}",
    "sv": r"s_v=\lambda\,\cos\big(\psi_q(h_{\mathrm{SEL}}),\,\psi_k(z_v)\big)+\beta",
    "vstar": r"v^{\star}=\arg\max_{v\in\mathcal{V}}\,s_v",
    "rej": r"\max_v\,\sigma(s_v)<\tau\;\Rightarrow\;\text{no target}",
    "out": r"m_{v^{\star}}", "ej": r"e_j", "xhat": r"\hat{x}_v", "hsel": r"h_{\mathrm{SEL}}", "zv": r"z_v",
}


def typeset(math: dict) -> dict:
    mj = Path(os.environ.get("MJ_DIR", HERE))
    with tempfile.TemporaryDirectory() as d:
        i, o = Path(d) / "in.json", Path(d) / "out.json"
        i.write_text(json.dumps(math))
        subprocess.run(["node", str(HERE / "tex2svg.js"), str(i), str(o)], check=True,
                       env={**os.environ, "NODE_PATH": str(mj / "node_modules")})
        return json.loads(o.read_text())


class Fig:
    def __init__(self, tex: dict):
        self.tex, self.parts, self.defs = tex, [], []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def box(self, x, y, w, h, kind, r=10, sw=1.6, dash=None):
        fill, stroke = C[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, weight=400, anchor="start", color=INK, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{color}"{st}>{s}</text>')

    def math(self, key, x, y, size=15, anchor="start", color=INK):
        # (x, y) = baseline position like <text>; MathJax units: 1ex ~ 0.442 em
        m = self.tex[key]
        ex = 0.442 * size
        w, h, va = m["w_ex"] * ex, m["h_ex"] * ex, m["va_ex"] * ex
        x0 = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
        top = y - (h + va)
        svg = re.sub(r'^<svg[^>]*?(viewBox="[^"]+")[^>]*>', rf'<svg x="{x0:.1f}" y="{top:.1f}" width="{w:.1f}" '
                     rf'height="{h:.1f}" \1 overflow="visible">', m["svg"])
        self.add(f'<g style="color:{color}">{svg}</g>')
        return w

    def arrow(self, d, color=INK, w=1.6, dash=None):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{ds} marker-end="url(#ar)"/>')

    def image(self, x, y, w, h, png_b64):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" '
                 f'href="data:image/png;base64,{png_b64}"/>')

    def svg(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
                '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
                '<pattern id="chk" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" '
                'fill="#fff"/><rect width="4" height="4" fill="#e8e8e8"/><rect x="4" y="4" width="4" height="4" '
                'fill="#e8e8e8"/></pattern>' + "".join(self.defs) + '</defs>'
                f'<rect width="{W}" height="{H}" fill="#fff"/>')
        return head + "\n".join(self.parts) + "</svg>"


# ---------------------------------------------------------------- image slots: real assets or drawn placeholders
class Assets:
    def __init__(self, path: Path | None, example: int):
        self.ex = None
        if path and path.exists():
            self.ex = json.loads(path.read_text())["examples"][example]
            from PIL import Image
            self.Image = Image
            self.img = Image.open(io.BytesIO(base64.b64decode(self.ex["image_jpeg_b64"]))).convert("RGB")

    def mask(self, i):
        from pycocotools import mask as mu
        import numpy as np
        r = self.ex["pool"][i] if isinstance(i, int) else self.ex[i]
        return mu.decode({"size": r["size"], "counts": r["counts"].encode()}).astype(bool), np

    def crop_box(self):
        m, np = self.mask("gt_object")
        ys, xs = np.nonzero(m)
        h, w = m.shape
        cx, cy, s = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(xs.ptp(), ys.ptp()) * 1.35
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        return x0, y0, int(min(w, x0 + s)), int(min(h, y0 + s))

    def png(self, mask_key=None, crop=True, cutout=False, color=(47, 127, 182)):
        import numpy as np
        im = np.asarray(self.img).astype(float)
        if mask_key is not None:
            m, _ = self.mask(mask_key)
            if cutout:
                chk = ((np.indices(m.shape).sum(0) // 8) % 2) * 22 + 232
                im = np.where(m[..., None], im, np.repeat(chk[..., None], 3, 2))
            else:
                im = np.where(m[..., None], 0.45 * im + 0.55 * np.array(color), im * 0.85)
        out = self.Image.fromarray(im.clip(0, 255).astype("uint8"))
        if crop:
            out = out.crop(self.crop_box())
        out.thumbnail((360, 360))
        b = io.BytesIO()
        out.save(b, format="PNG", optimize=True)
        return base64.b64encode(b.getvalue()).decode()


def placeholder_scene(f: Fig, x, y, s, part="all"):
    # drawn stand-in until the real assets exist: a mug with a handle on a table
    k = s / 100
    g = []
    if part == "all":
        g += [f'<rect x="0" y="0" width="100" height="100" fill="#f3efe6"/>',
              '<rect x="0" y="66" width="100" height="34" fill="#c9a77c"/>']
    else:
        g += ['<rect x="0" y="0" width="100" height="100" fill="url(#chk)"/>']
    if part in ("all", "mug", "object"):
        g += ['<rect x="28" y="26" width="38" height="46" rx="5" fill="#3f78b5"/>',
              '<ellipse cx="47" cy="26" rx="19" ry="4.5" fill="#2c5a8c"/>']
    if part in ("all", "handle", "object"):
        g += ['<path d="M66,36 C84,36 84,62 66,62" fill="none" stroke="#3f78b5" stroke-width="7"/>']
    if part == "table":
        g += ['<rect x="0" y="66" width="100" height="34" fill="#c9a77c"/>']
    if part == "all":
        g += ['<rect x="6" y="44" width="13" height="24" fill="#b5653a"/>', '<circle cx="12.5" cy="38" r="10" fill="#4e9a52"/>']
    f.add(f'<g transform="translate({x},{y}) scale({k})">' + "".join(g) + "</g>")


def slot(f: Fig, a: Assets, x, y, s, key, border, badge=None, cutout=True):
    if a.ex is not None:
        f.image(x, y, s, s, a.png(key, cutout=cutout))
    else:
        placeholder_scene(f, x, y, s, {"img": "all", "pick": "handle", "gt_object": "object"}.get(key, key))
    f.add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="4" fill="none" stroke="{border}" stroke-width="2"/>')
    if badge:
        f.add(f'<rect x="{x+3}" y="{y+3}" width="22" height="15" rx="3" fill="{border}"/>')
        f.text(x + 14, y + 14.5, badge, size=10, weight=700, anchor="middle", color="#fff")


def build(a: Assets, tex: dict) -> str:
    f = Fig(tex)
    grid_c, text_c, sel_c = "#8a93a3", C["sel"][1], C["sel"][1]

    def panel(x, w, num, title):
        f.box(x, 14, w, H - 28, "panel", r=14, sw=1.2)
        f.add(f'<circle cx="{x+24}" cy="40" r="13" fill="{INK}"/>')
        f.text(x + 24, 45, str(num), size=14, weight=700, anchor="middle", color="#fff")
        f.text(x + 46, 46, title, size=17, weight=700)

    # ---------------------------------------------------------------- panel 1: candidate pool
    panel(14, 520, 1, "Query-conditioned candidate pool")
    f.box(32, 66, 484, 150, "seg", r=10, sw=1.2)
    if a.ex is not None:
        f.image(44, 76, 130, 130, a.png(None, crop=False))
    else:
        placeholder_scene(f, 44, 76, 130)
    f.add(f'<rect x="44" y="76" width="130" height="130" rx="4" fill="none" stroke="#9aa3b2"/>')
    f.text(190, 100, "Query", size=12, weight=700, color=MUTED)
    f.text(190, 124, "“Locate the handle of the mug.”", size=15, italic=True)
    wphi = f.math("phi", 190, 160, size=15, color=MUTED)
    f.text(190 + wphi + 6, 160, "= { “the handle of the mug”, “the mug” }", size=13, color=MUTED)
    f.text(190, 192, "SAM 3 is prompted with the query’s phrases", size=11.5, color=MUTED)

    f.arrow("M120,216 L120,236")
    f.arrow("M380,216 L380,236")
    f.box(32, 238, 236, 54, "seg", r=9)
    f.text(150, 260, "SAM 3 · point grid", size=14, weight=700, anchor="middle")
    f.text(150, 280, "query-free, 32×32 points", size=11.5, anchor="middle", color=MUTED)
    f.box(280, 238, 236, 54, "sel", r=9)
    f.text(398, 260, "SAM 3 · text", size=14, weight=700, anchor="middle")
    f.text(398, 280, "top-3 instances per phrase", size=11.5, anchor="middle", color=MUTED)
    f.arrow("M150,292 L150,312")
    f.arrow("M398,292 L398,312")

    f.box(32, 314, 484, 128, "seg", r=10, sw=1.2)
    f.math("V", 44, 336, size=15)
    f.text(140, 336, "one pool per query", size=11.5, color=MUTED)
    gx = [44, 116, 188]
    keys = ["grid0", "grid1", "grid2"]
    if a.ex is not None:
        import numpy as np
        ng = a.ex["n_grid"]
        lg = np.array(a.ex["logits"][:ng])
        keys = [int(i) for i in np.argsort(-lg)[:3]]
    else:
        keys = ["table", "mug", "object"]
    for x, k in zip(gx, keys):
        slot(f, a, x, 348, 64, k, grid_c)
    f.math("G", 44 + 98, 432, size=13, anchor="middle", color=MUTED)
    f.text(166, 432, "grid masks", size=11, color=MUTED)
    f.add('<line x1="268" y1="346" x2="268" y2="436" stroke="#c3c9d4" stroke-dasharray="3 3"/>')
    if a.ex is not None:
        txt = [(a.ex["n_grid"] + i, f'#{m["rank"]+1}') for i, m in enumerate(a.ex["instances"])][:3]
    else:
        txt = [("handle", "#1"), ("object", "#1"), ("mug", "#2")]
    for (k, b), x in zip(txt, [282, 354, 426]):
        slot(f, a, x, 348, 64, k, text_c, badge=b)
    f.math("T", 392, 432, size=13, anchor="middle", color=text_c)
    f.text(404, 432, "SAM 3 text instances", size=11, color=text_c)

    f.arrow("M274,442 L274,462")
    f.box(32, 464, 484, 156, "seg", r=10, sw=1.2)
    f.text(44, 488, "Containment tree", size=14, weight=700)
    f.math("parent", 44, 526, size=14)
    f.text(44, 572, "every candidate links to its smallest container;", size=11.5, color=MUTED)
    f.text(44, 590, "the relation of two candidates biases attention (panel 3)", size=11.5, color=MUTED)
    for (x1, y1, x2, y2) in ((450, 490, 420, 530), (450, 490, 480, 530), (480, 530, 500, 572), (480, 530, 462, 572)):
        f.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#7d8696" stroke-width="1.4"/>')
    for (x, y, c) in ((450, 490, C["mask"][1]), (420, 530, grid_c), (480, 530, C["mask"][1]),
                      (500, 572, text_c), (462, 572, grid_c)):
        f.add(f'<circle cx="{x}" cy="{y}" r="7" fill="{c}" stroke="#fff" stroke-width="1.5"/>')

    # ---------------------------------------------------------------- panel 2: mask tokens
    panel(548, 372, 2, "One token per candidate")
    hk = "pick" if a.ex is None else a.ex["pick"]
    slot(f, a, 568, 72, 78, hk, C["mask"][1])
    f.math("mv", 607, 168, size=14, anchor="middle")
    f.arrow("M652,111 L684,111")
    cx0, cy0, cs = 690, 76, 70
    f.add(f'<rect x="{cx0}" y="{cy0}" width="{cs}" height="{cs}" fill="#fff" stroke="#9fb3c8"/>')
    for i in range(1, 7):
        f.add(f'<line x1="{cx0+i*10}" y1="{cy0}" x2="{cx0+i*10}" y2="{cy0+cs}" stroke="#d3dde8"/>')
        f.add(f'<line x1="{cx0}" y1="{cy0+i*10}" x2="{cx0+cs}" y2="{cy0+i*10}" stroke="#d3dde8"/>')
    for (cx, cy, op) in ((4, 2, .9), (4, 3, .55), (4, 4, .9), (3, 2, .3), (3, 4, .3), (5, 3, .15)):
        f.add(f'<rect x="{cx0+cx*10}" y="{cy0+cy*10}" width="10" height="10" fill="{C["mask"][1]}" opacity="{op}"/>')
    f.math("cv", cx0 + cs / 2, 168, size=14, anchor="middle")
    f.text(cx0 + cs + 12, 100, "coverage of each", size=11, color=MUTED)
    f.text(cx0 + cs + 12, 115, "image-token cell", size=11, color=MUTED)

    y = 186
    for (key, label, kind, x, w) in (("pv", "appearance", "mask", 562, 128), ("gv", "geometry", "mask", 698, 100),
                                     ("qv", "provenance", "sel", 806, 100)):
        f.box(x, y, w, 70, kind, r=8, sw=1.3)
        f.text(x + w / 2, y + 18, label, size=11, weight=700, anchor="middle", color=MUTED)
        f.math(key, x + w / 2, y + (50 if key == "pv" else 48), size=12 if key == "pv" else 14, anchor="middle")
    f.text(748, y + 64, "box, area, depth", size=9.5, anchor="middle", color=MUTED)
    f.text(856, y + 64, "text?, rank, score", size=9.5, anchor="middle", color=MUTED)
    for x in (626, 748, 856):
        f.add(f'<line x1="{x}" y1="{y+70}" x2="{x}" y2="{y+84}" stroke="{INK}" stroke-width="1.4"/>')
    f.add(f'<line x1="626" y1="{y+84}" x2="856" y2="{y+84}" stroke="{INK}" stroke-width="1.4"/>')
    f.arrow(f"M734,{y+84} L734,{y+100}")
    f.box(566, y + 102, 336, 44, "mask", r=8, sw=1.6)
    f.math("xv", 734, y + 130, size=15, anchor="middle")

    f.box(566, y + 160, 336, 112, "seg", r=8, sw=1.2)
    f.text(580, y + 182, "Where the token sits and what it sees", size=12, weight=700)
    f.math("pos", 580, y + 210, size=14)
    f.text(580, y + 228, "placed at the mask centroid (mRoPE)", size=11, color=MUTED)
    f.math("Rv", 580, y + 254, size=14)
    f.text(800, y + 254, "image region", size=11, color=MUTED)

    f.text(734, 616, "one forward pass for all candidates", size=11.5, anchor="middle", color=MUTED, italic=True)

    # ---------------------------------------------------------------- panel 3: selection
    panel(934, 552, 3, "Selection inside the LLM")
    # input sequence
    sy = 70
    f.box(952, sy, 516, 64, "seg", r=8, sw=1.1)
    toks = [("img", 4), ("text", 4), ("sel", 1), ("mask", 6)]
    x = 964
    for kind, n in toks:
        for i in range(n):
            fill, stroke = C[kind]
            wd = 34 if kind == "sel" else 18
            f.add(f'<rect x="{x}" y="{sy+12}" width="{wd}" height="26" rx="3" fill="{fill if kind != "sel" else "#e58ac0"}" '
                  f'stroke="{stroke}"/>')
            if kind == "sel":
                f.text(x + 17, sy + 29, "SEL", size=9, weight=700, anchor="middle")
            x += wd + 3
        x += 14
    f.math("ej", 1006, sy + 56, size=12, anchor="middle", color=MUTED)
    f.text(1090, sy + 56, "query", size=11, anchor="middle", color=MUTED)
    f.math("hsel", 1176, sy + 56, size=12, anchor="middle", color=MUTED)
    f.math("xhat", 1290, sy + 56, size=12, anchor="middle", color=MUTED)
    f.text(1350, sy + 56, "appended mask tokens", size=11, color=MUTED)
    f.arrow(f"M1210,{sy+64} L1210,{sy+84}")
    f.box(952, sy + 86, 516, 54, "llm", r=12, sw=2)
    f.text(1210, sy + 110, "Qwen3-VL-4B + LoRA (r = 16)", size=16, weight=700, anchor="middle")
    f.text(1210, sy + 128, "text and image tokens are unchanged: they never attend to mask tokens", size=11,
           anchor="middle", color=MUTED)

    # attention mask inset
    ax, ay, cell = 1000, 262, 21
    f.text(952, 248, "Attention of a mask-token row", size=12.5, weight=700)
    cols = [("image", 4), ("text", 3), ("mask", 3)]
    names = ["image"] * 4 + ["text"] * 3 + ["mask"] * 3
    for r_ in range(10):
        for c_ in range(10):
            rn, cn = names[r_], names[c_]
            fill = "#ffffff"
            if rn != "mask":
                if c_ <= r_ and cn != "mask":
                    fill = C["img"][0] if cn == "image" else C["text"][0]
            else:
                if cn == "text":
                    fill = "#8fd18b"
                elif cn == "image":
                    fill = "#f0a27a" if (r_, c_) in ((7, 1), (8, 3), (9, 0), (9, 1)) else "#ffffff"
                else:
                    fill = C["mask"][1]
            f.add(f'<rect x="{ax+c_*cell}" y="{ay+r_*cell}" width="{cell}" height="{cell}" fill="{fill}" '
                  f'stroke="#c6ccd6" stroke-width="0.6"/>')
    for k in (4, 7):
        f.add(f'<line x1="{ax+k*cell}" y1="{ay}" x2="{ax+k*cell}" y2="{ay+10*cell}" stroke="{INK}" stroke-width="1.3"/>')
        f.add(f'<line x1="{ax}" y1="{ay+k*cell}" x2="{ax+10*cell}" y2="{ay+k*cell}" stroke="{INK}" stroke-width="1.3"/>')
    f.add(f'<rect x="{ax}" y="{ay}" width="{10*cell}" height="{10*cell}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    for (lab, a0, n) in (("image", 0, 4), ("text", 4, 3), ("mask", 7, 3)):
        f.text(ax + (a0 + n / 2) * cell, ay - 6, lab, size=10.5, anchor="middle", color=MUTED)
        f.text(ax - 6, ay + (a0 + n / 2) * cell + 4, lab, size=10.5, anchor="end", color=MUTED)
    f.math("bias", ax + 8.5 * cell, ay + 8.5 * cell + 6, size=14, anchor="middle", color="#fff")
    lx = ax + 10 * cell + 22
    for i, (fill, s1, s2) in enumerate((("#8fd18b", "query text:", "always visible"),
                                        ("#f0a27a", "image: only its region", None),
                                        (C["mask"][1], "other candidates:", "bias by tree relation"),
                                        ("#ffffff", "blocked", None))):
        yy = ay + 8 + i * 46
        f.add(f'<rect x="{lx}" y="{yy-11}" width="14" height="14" fill="{fill}" stroke="#9aa3b2"/>')
        f.text(lx + 22, yy, s1, size=11.5)
        if s2:
            f.text(lx + 22, yy + 15, s2, size=11, color=MUTED)
    f.math("Rv", lx + 22 + 0, ay + 8 + 46 + 18, size=11, color=MUTED)

    # selection head + output
    hy = 486
    f.box(952, hy, 330, 132, "llm", r=10, sw=1.6)
    f.text(966, hy + 22, "Selection head", size=13.5, weight=700)
    f.math("sv", 966, hy + 54, size=14.5)
    f.math("vstar", 966, hy + 88, size=14.5)
    f.math("rej", 966, hy + 118, size=13, color=MUTED)
    f.add(f'<path d="M1458,{sy+140} L1458,{hy-8} L1290,{hy-8} L1290,{hy+60} L1284,{hy+60}" fill="none" stroke="{INK}" '
          f'stroke-width="1.5" marker-end="url(#ar)"/>')
    f.math("zv", 1466, (sy + 140 + hy) / 2, size=12, color=MUTED)
    f.arrow(f"M1282,{hy+66} L1308,{hy+66}")
    f.box(1310, hy - 4, 158, 140, "seg", r=10, sw=1.6)
    f.text(1389, hy + 16, "Output", size=14, weight=700, anchor="middle")
    slot(f, a, 1349, hy + 26, 80, "pick" if a.ex is None else a.ex["pick"], sel_c, cutout=False)
    f.math("out", 1389, hy + 128, size=14, anchor="middle")
    # pool -> tokens, tokens -> sequence
    f.arrow("M516,378 L560,378 L560,111 L566,111")
    f.arrow(f"M902,{186+124} L926,{186+124} L926,{sy+25} L952,{sy+25}")
    return f.svg()


def render_pdf(svg_path: Path, pdf_path: Path, png_path: Path | None = None) -> None:
    js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
          f"'{svg_path}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:{{width:{W},"
          f"height:{H}}}}});await p.setContent(`<!doctype html><html><head><style>@page{{size:{W}px {H}px;margin:0}}"
          f"html,body{{margin:0}}</style></head><body>${{svg}}</body></html>`);"
          + (f"await p.screenshot({{path:'{png_path}',clip:{{x:0,y:0,width:{W},height:{H}}}}});" if png_path else "")
          + f"await p.pdf({{path:'{pdf_path}',width:'{W}px',height:'{H}px',printBackground:true,pageRanges:'1'}});"
          "await b.close();})();")
    subprocess.run(["node", "-e", js], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, default=0)
    ap.add_argument("--assets", default=str(ROOT / "results/figures/overview_assets_mug_handle.json"))
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    a = Assets(Path(args.assets), args.example)
    svg = build(a, typeset(MATH))
    (HERE / "overview.svg").write_text(svg)
    render_pdf(HERE / "overview.svg", HERE / "overview.pdf", Path(args.png) if args.png else None)
    print("real assets" if a.ex is not None else "placeholders", HERE / "overview.pdf")


if __name__ == "__main__":
    main()
