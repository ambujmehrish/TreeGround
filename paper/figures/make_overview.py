"""Method overview figure: overview_template.svg (the agreed layout) -> overview.svg -> overview.pdf.

Three substitutions on the template, nothing else moves:
  1. image slots: the real PACO test image, real candidate masks and the model's pick, from
     results/figures/overview_assets_mug_handle.json (scripts/fig_assets.py, run on the cluster);
  2. symbols and equations typeset by MathJax (tex2svg.js; `npm install mathjax-full@3.2.2` next to it, or MJ_DIR);
  3. box strokes unified.
usage: python paper/figures/make_overview.py [--example 6] [--png preview.png]
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

import numpy as np
from PIL import Image, ImageFont
from pycocotools import mask as mu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 1400, 640
INTER = {400: "/usr/share/fonts/opentype/inter/Inter-Regular.otf", 600: "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
         700: "/usr/share/fonts/opentype/inter/Inter-Bold.otf"}

# pool slots of the template (grid x3, text x2) and the example's pool indices shown there; output = the model's pick
SLOTS = {"grid": [(282, 274), (342, 274), (402, 274)], "text": [(478, 274), (542, 274)]}

# template text (exact content) -> replacement parts: ("t", text) or ("m", tex)
TEXT = {
    "phrases φ(x):": [("t", "phrases "), ("m", r"\phi(x)"), ("t", ":")],
    "parent(v): smallest u with": [("m", r"\mathrm{parent}(v)=\arg\min_{u}\,|m_u|")],
    "|m_u ∩ m_v| ≥ 0.8 |m_v|": [("t", "s.t. "), ("m", r"|m_u\cap m_v|\ \geq\ 0.8\,|m_v|")],
    "Visual tokens e_j": [("t", "Visual tokens "), ("m", r"e_j")],
    "Candidate mask tokens x̂_v (one per mask)": [("t", "Candidate mask tokens "), ("m", r"\hat{x}_v"),
                                                 ("t", " (one per mask)")],
    "mask m_v": [("t", "mask "), ("m", r"m_v")],
    "coverage c_v": [("t", "coverage "), ("m", r"c_v")],
    "pooled p_v": [("t", "appearance")],
    "Σ c·e / Σ c": [("m", r"p_v=\textstyle\sum_j \tilde{c}_{v,j}\,e_j")],
    "geometry g_v": [("t", "geometry")],
    "box, area, depth": [("m", r"g_v\in\mathbb{R}^{9}")],
    "provenance q_v": [("t", "provenance")],
    "text?, rank, score": [("m", r"q_v\in\mathbb{R}^{5}")],
    "x̂_v": [("m", r"\hat{x}_v")],
    "mRoPE position = centroid of m_v": [("t", "mRoPE position "), ("m", r"(\bar{r}_v,\bar{c}_v)"),
                                         ("t", " = centroid of "), ("m", r"m_v")],
    "b[ℓ, rel]": [("m", r"b_{\ell,\,\mathrm{rel}(v,u)}")],
    "s_v = λ · cos( f(h_SEL), g(z_v) ) + β ,   v* = argmax_v s_v":
        [("m", r"s_v=\lambda\,\cos\big(\psi_q(h_{\mathrm{SEL}}),\,\psi_k(z_v)\big)+\beta,\qquad "
               r"v^{\star}=\arg\max_{v}\,s_v")],
    "mask m_v* (“handle”)": [("t", "mask "), ("m", r"m_{v^{\star}}"), ("t", " (“handle”)")],
}
SIZE = {"Σ c·e / Σ c": 10.5, "box, area, depth": 11.5, "text?, rank, score": 11.5, "x̂_v": 15,
        "b[ℓ, rel]": 12.5, "s_v = λ · cos( f(h_SEL), g(z_v) ) + β ,   v* = argmax_v s_v": 15,
        "parent(v): smallest u with": 12.5, "|m_u ∩ m_v| ≥ 0.8 |m_v|": 12}


def typeset(texs: list[str]) -> dict:
    mj = Path(os.environ.get("MJ_DIR", HERE))
    with tempfile.TemporaryDirectory() as d:
        i, o = Path(d) / "in.json", Path(d) / "out.json"
        i.write_text(json.dumps({str(k): t for k, t in enumerate(texs)}))
        subprocess.run(["node", str(HERE / "tex2svg.js"), str(i), str(o)], check=True,
                       env={**os.environ, "NODE_PATH": str(mj / "node_modules")})
        out = json.loads(o.read_text())
    return {t: out[str(k)] for k, t in enumerate(texs)}


def text_width(s: str, size: float, weight: int) -> float:
    return ImageFont.truetype(INTER.get(weight, INTER[400]), size=100).getlength(s) * size / 100


def math_svg(m: dict, x: float, baseline: float, size: float, color: str) -> tuple[str, float]:
    ex = 0.442 * size                              # MathJax ex for a body size `size`
    w, h, va = m["w_ex"] * ex, m["h_ex"] * ex, m["va_ex"] * ex
    top = baseline - (h + va)
    svg = re.sub(r'^<svg[^>]*?(viewBox="[^"]+")[^>]*>', rf'<svg x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" '
                 rf'height="{h:.1f}" \1 overflow="visible">', m["svg"])
    return f'<g style="color:{color}">{svg}</g>', w


def replace_texts(svg: str, tex: dict) -> str:
    def sub(mt):
        attrs, content = mt.group(1), mt.group(2)
        plain = content.replace("&lt;", "<").replace("&gt;", ">")
        if plain not in TEXT:
            return mt.group(0)
        a = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
        x, y = float(a["x"]), float(a["y"])
        size = SIZE.get(plain, float(a.get("font-size", 13)))
        weight = int(a.get("font-weight", 400))
        color = a.get("fill", "#111")
        parts = TEXT[plain]
        widths = [text_width(s, size, weight) if k == "t" else tex[s]["w_ex"] * 0.442 * size for k, s in parts]
        total = sum(widths)
        anchor = a.get("text-anchor", "start")
        x0 = x - total / 2 if anchor == "middle" else x - total if anchor == "end" else x
        out = []
        for (k, s), w in zip(parts, widths):
            if k == "t":
                out.append(f'<text x="{x0:.1f}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                           f'xml:space="preserve">{s}</text>')
            else:
                out.append(math_svg(tex[s], x0, y, size, color)[0])
            x0 += w
        return "".join(out)
    return re.sub(r'<text ([^>]*)>([^<]*)</text>', sub, svg)


# ---------------------------------------------------------------- real image patches
def dec(r: dict) -> np.ndarray:
    return mu.decode({"size": r["size"], "counts": r["counts"].encode()}).astype(bool)


def png_b64(im: Image.Image) -> str:
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def square(box, w, h, margin=0.18, min_side=0):
    x0, y0, x1, y1 = box
    s = max(x1 - x0, y1 - y0, min_side) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(s, w, h)
    ax = min(max(0, cx - s / 2), w - s)
    ay = min(max(0, cy - s / 2), h - s)
    return int(ax), int(ay), int(ax + s), int(ay + s)


def bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def cutout(img: np.ndarray, m: np.ndarray, side: int) -> str:
    # the mask's pixels on a transparent ground (the template's checkerboard shows through), square crop
    h, w = m.shape
    crop = square(bbox(m), w, h, margin=0.15, min_side=24)
    rgba = np.dstack([img, (m * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, "RGBA").crop(crop).resize((side, side), Image.LANCZOS)
    return png_b64(im)


def overlay(img: np.ndarray, m: np.ndarray, region: np.ndarray, side: int, color=(176, 48, 111)) -> str:
    h, w = m.shape
    crop = square(bbox(region), w, h, margin=0.25)
    x = img.astype(float)
    x = np.where(m[..., None], 0.35 * x + 0.65 * np.array(color), x * 0.72 + 0.28 * 255)
    edge = m ^ np.pad(m, 2, mode="edge")[4:, 2:-2] | m ^ np.pad(m, 2, mode="edge")[2:-2, 4:]
    x[edge] = color
    return png_b64(Image.fromarray(x.clip(0, 255).astype(np.uint8)).crop(crop).resize((side, side), Image.LANCZOS))


def image_tag(x, y, s, b64):
    return (f'<image x="{x}" y="{y}" width="{s}" height="{s}" preserveAspectRatio="none" '
            f'href="data:image/png;base64,{b64}"/>')


def put_images(svg: str, ex: dict, grid_ids: list[int], text_ids: list[int]) -> str:
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(ex["image_jpeg_b64"]))).convert("RGB"))
    h, w = img.shape[:2]
    pool = [dec(r) for r in ex["pool"]]
    obj = dec(ex["gt_object"])
    shown = obj | pool[grid_ids[1]]                     # the scene crop holds the mug and the second grid object
    scene = Image.fromarray(img).crop(square(bbox(shown), w, h, margin=0.12)).resize((260, 260), Image.LANCZOS)
    # 1. the input image
    svg = re.sub(r'<g transform="translate\(34,28\) scale\(1.3\)">.*?</g>',
                 image_tag(34, 28, 130, png_b64(scene)), svg, count=1, flags=re.S)
    # 2. pool thumbnails (template order: grid 282 / 342 / 402, text 478 / 542)
    olds = ['<g transform="scale(0.58)"><use href="#table"/></g>', '<g transform="scale(0.58)"><use href="#plant"/></g>',
            '<g transform="scale(0.58)"><use href="#mugbody"/></g>',
            '<g transform="scale(0.58)"><use href="#mugbody"/><use href="#handle"/></g>',
            '<g transform="scale(0.58)"><use href="#handle"/></g>']
    for old, i in zip(olds, grid_ids + text_ids):
        assert svg.count(old) == 1, old
        svg = svg.replace(old, image_tag(0, 0, 58, cutout(img, pool[i], 116)))
    # 3. mask-tokenization thumbnail and the output: the model's pick
    pick = pool[ex["pick"]]
    old = '<g transform="scale(0.54)"><use href="#handle"/></g>'
    assert svg.count(old) == 1
    svg = svg.replace(old, image_tag(0, 0, 54, cutout(img, pick, 108)))
    old = '<g transform="scale(0.8)"><use href="#handle"/></g>'
    assert svg.count(old) == 1
    svg = svg.replace(old, image_tag(0, 0, 80, overlay(img, pick, obj, 160)))
    return svg


def polish(svg: str) -> str:
    svg = svg.replace('font-family="Inter, DejaVu Sans, sans-serif"', 'font-family="Inter, Helvetica, Arial, sans-serif"')
    svg = svg.replace('stroke="#111" stroke-width="2.5"', 'stroke="#2b3240" stroke-width="2"')
    svg = svg.replace('stroke="#111" stroke-width="2"', 'stroke="#2b3240" stroke-width="1.8"')
    svg = svg.replace('stroke="#111" stroke-width="1.6"', 'stroke="#2b3240" stroke-width="1.5"')
    return svg


def render_pdf(svg_path: Path, pdf_path: Path, png_path: Path | None) -> None:
    js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
          f"'{svg_path}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:{{width:{W},"
          f"height:{H}}},deviceScaleFactor:2}});await p.setContent(`<!doctype html><html><head><style>@page{{size:{W}px "
          f"{H}px;margin:0}}html,body{{margin:0}}</style></head><body>${{svg}}</body></html>`);"
          + (f"await p.screenshot({{path:'{png_path}',clip:{{x:0,y:0,width:{W},height:{H}}}}});" if png_path else "")
          + f"await p.pdf({{path:'{pdf_path}',width:'{W}px',height:'{H}px',printBackground:true,pageRanges:'1'}});"
          "await b.close();})();")
    subprocess.run(["node", "-e", js], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ROOT / "results/figures/overview_assets_mug_handle.json"))
    ap.add_argument("--example", type=int, default=6)
    ap.add_argument("--grid", default="83,15,65", help="pool indices of the three grid thumbnails")
    ap.add_argument("--text", default="257,256", help="pool indices of the two SAM 3 text thumbnails")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    svg = (HERE / "overview_template.svg").read_text()
    ex = json.loads(Path(args.assets).read_text())["examples"][args.example]
    svg = put_images(svg, ex, [int(i) for i in args.grid.split(",")], [int(i) for i in args.text.split(",")])
    tex = typeset(sorted({s for parts in TEXT.values() for k, s in parts if k == "m"}))
    svg = polish(replace_texts(svg, tex))
    (HERE / "overview.svg").write_text(svg)
    render_pdf(HERE / "overview.svg", HERE / "overview.pdf", Path(args.png) if args.png else None)
    print(HERE / "overview.pdf", "image", ex["image_id"], "pick IoU", round(ex["pick_iou"], 2))


if __name__ == "__main__":
    main()
