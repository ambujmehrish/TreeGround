"""Main-paper qualitative figure (two-column): one row per task (part grounding, part reasoning, region-grounded
editing), two selected test examples per row, each shown as the request, our answer and the strongest same-size
baseline (Qwen3-VL-4B box -> SAM 3).

Every pixel is a real model output on a test query: results/figures/qual_{part,reason,edit}_showcase.json
(scripts/qual_assets.py --showcase; ours = pool_soft_fam, reasoning with D51 query rewriting, editing with
Qwen-Image-Edit). Part examples also show the model's three top-ranked candidates with their softmax shares over the
pool; reasoning examples show the two strongest baselines. IoU against the ground-truth mask is printed under each answer; the examples are selected (clear wins), and the
random draws are in the supplementary.
usage: python paper/figures/make_showcase.py --part 13 9 --reason 6 14 --edit 0 2 --png
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
from make_overview import INTER, bbox, dec, text_width  # noqa: E402
from make_teaser import crop_box, edge, hexc, text, wrap  # noqa: E402

W = 1400
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, MUTED, LINE, BUBBLE = "#1f2633", "#5b6474", "#d5d9e0", "#eef1f6"
PICK, BASE, GT = (176, 48, 111), (59, 111, 182), (36, 150, 72)
ALT = [(217, 130, 43), (24, 128, 120)]                  # outlines of the next-ranked candidates
F_ROW, F_Q, F_LAB, F_SMALL = 27, 22, 21, 20      # at \\linewidth 1 px ~ 0.35 pt: 7 pt and up
BASELINE = "qwen_box_sam"
ASPECT = 4 / 3


def load(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def mask(r, hw) -> np.ndarray:
    return dec(r) if r else np.zeros(hw, bool)


def img_tag(x, y, w, h, b64, r=6) -> str:
    cid = f"k{abs(hash((x, y, w, h))) % 10**9}"
    return (f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}"/></clipPath>'
            f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="none" '
            f'clip-path="url(#{cid})" href="data:image/jpeg;base64,{b64}"/>')


def cell(a, box, size, fills=(), outlines=(), r=3) -> str:
    # crop, resize to the output size, then draw: masks are resampled smoothly so outlines stay clean when upscaled
    x0, y0, x1, y1 = box
    im = np.asarray(Image.fromarray(a[y0:y1, x0:x1]).resize(size, Image.LANCZOS))

    def rs(m):
        f = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        return np.asarray(f) > 127
    out = overlay(im, [(rs(m), c) for m, c in fills], [(rs(m), c) for m, c in outlines], line=r)
    b = io.BytesIO()
    Image.fromarray(out).save(b, format="JPEG", quality=93)
    return base64.b64encode(b.getvalue()).decode()


def overlay(img, fills, outlines=(), line=4, fill=0.5) -> np.ndarray:
    x = img.astype(float)
    for m, c in fills:
        x = np.where(m[..., None], (1 - fill) * x + fill * np.array(c), x)
        x[edge(m, line)] = c
    for m, c in outlines:                                  # dashed outline only
        e = edge(m, line)
        yy, xx = np.nonzero(e)
        on = ((yy + xx) // 12) % 2 == 0
        x[yy[on], xx[on]] = c
    return x.clip(0, 255).astype(np.uint8)


def focus(ms, hw):
    ms = [m for m in ms if m.any()]
    if not ms:
        return 0, 0, hw[1], hw[0]
    b = np.array([bbox(m) for m in ms])
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


def chip(x, y, s, color, size=F_SMALL, weight=600) -> str:
    w = text_width(s, size, weight) + 12
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{size + 8}" rx="{(size + 8) / 2:.1f}" '
            f'fill="{color}"/>' + text(x + 6, y + size + 1.5, s, size, weight, "#fff")), w


def bubble(x, y, w, s, max_lines=4) -> tuple[str, float]:
    lines = wrap(s, F_Q, w - 64)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;") + " …"
    h = 16 + len(lines) * (F_Q + 6)
    out = [f'<circle cx="{x + 15}" cy="{y + 19}" r="15" fill="{INK}"/>',
           text(x + 15, y + 25.5, "Q", 17, 700, "#fff", "middle"),
           f'<rect x="{x + 38}" y="{y}" width="{w - 38}" height="{h}" rx="12" fill="{BUBBLE}"/>',
           f'<path d="M{x + 38} {y + 13} l-8 6 l8 5 z" fill="{BUBBLE}"/>']
    out += [text(x + 50, y + 8 + (i + 1) * (F_Q + 6) - 5, l, F_Q, 400, INK, italic=True) for i, l in enumerate(lines)]
    return "".join(out), h


def thumb(a, m, color, side) -> str:
    # square crop around the candidate, the candidate filled
    h, w = m.shape
    x0, y0, x1, y1 = bbox(m)
    e = min(max(x1 - x0, y1 - y0) * 1.5, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = int(min(max(0, cx - e / 2), w - e)), int(min(max(0, cy - e / 2), h - e))
    return cell(a, (ax, ay, int(ax + e), int(ay + e)), (side * 2, side * 2), [(m, color)], r=3)


def side_panel(x, y, w, h, s, a, gt, box, kind) -> list[str]:
    # right column: (part) the top-3 candidates with their shares, then the baseline; (reasoning) two baselines
    hw = a.shape[:2]
    lab = F_SMALL + 8
    out = []

    def base_cell(by, bh, name, label):
        m = mask(s["methods"][name]["mask"], hw)
        bw = bh * ASPECT
        out.append(text(x, by + F_SMALL, label, F_SMALL, 600, hexc(BASE)))
        out.append(text(x + w, by + F_SMALL, f"IoU {s['methods'][name]['iou']:.2f}", F_SMALL, 400, MUTED, "end"))
        out.append(img_tag(x + w - bw, by + lab, bw, bh, cell(a, box, (int(bw * 2), int(bh * 2)), [(m, BASE)], [(gt, GT)], 4)))
        if not m.any():
            out.append(chip(x + w - bw + 6, by + lab + 6, "no mask", "#7a8190")[0])

    if kind == "part":
        t = (w - 2 * 8) / 3
        out.append(text(x, y + F_SMALL, "Top-3 candidates", F_SMALL, 600, INK))
        for k, c in enumerate(s["candidates"][:3]):
            col = PICK if k == 0 else ALT[k - 1]
            tx = x + k * (t + 8)
            out.append(img_tag(tx, y + lab, t, t, thumb(a, mask(c["mask"], hw), col, int(t)), 5))
            out.append(chip(tx + 4, y + lab + t - F_SMALL - 12, f"{c['p'] * 100:.0f}%", hexc(col))[0])
        bh = h - (lab + t + 14) - lab
        base_cell(y + lab + t + 14, bh, BASELINE, "Box + SAM 3")
    else:
        bh = (h - 2 * lab - 12) / 2
        base_cell(y, bh, BASELINE, "Box + SAM 3")
        base_cell(y + lab + bh + 12, bh, "sa2va", "Sa2VA-4B")
    return out


def grounding_card(s, x, y, w, kind) -> tuple[str, float]:
    a = load(s["image"])
    hw = a.shape[:2]
    gt, ours = mask(s["gt"], hw), mask(s["methods"]["ours"]["mask"], hw)
    base = mask(s["methods"][BASELINE]["mask"], hw)
    box = crop_box(hw[0], hw[1], focus([gt, ours, base], hw), ASPECT, margin=0.45, min_frac=0.6)
    q = s["query"].replace("Locate ", "").replace(" <SEL>", "")
    body, bh = bubble(x, y, w, q[0].upper() + q[1:])
    iy = y + bh + 10
    sw = (w - 14) * 0.36
    mw = w - 14 - sw
    mh = mw / ASPECT
    out = [body, img_tag(x, iy, mw, mh, cell(a, box, (int(mw * 2), int(mh * 2)), [(ours, PICK)], r=5))]
    out += side_panel(x + mw + 14, iy, sw, mh, s, a, gt, box, kind)
    ty = iy + mh + 27
    out.append(text(x, ty, "Ours", F_LAB, 700, hexc(PICK)))
    out.append(text(x + text_width("Ours ", F_LAB, 700) + 4, ty, f"IoU {s['methods']['ours']['iou']:.2f}", F_SMALL, 400, MUTED))
    out.append(text(x + w, ty, "dashed: ground truth", F_SMALL, 400, hexc(GT), "end"))
    return "".join(out), ty - y + 8


def edit_card(s, x, y, w) -> tuple[str, float]:
    a = load(s["image"])
    hw = a.shape[:2]
    ours = mask(s["masks"]["ours_softfam"], hw)
    eo, eb = load(s["edits"]["qwen_image_edit/ours_softfam_blend"]), load(s["edits"]["qwen_image_edit/qwen_box_sam_blend"])
    body, bh = bubble(x, y, w, s["query"], 2)
    iy = y + bh + 10
    cw = (w - 16) / 3
    ch = cw / ASPECT
    if eo.shape != a.shape or eb.shape != a.shape:
        raise SystemExit(f"{s['qid']}: edits are not at the source size")
    full = crop_box(hw[0], hw[1], focus([ours], hw), ASPECT, margin=0.1, min_frac=1.0)   # the largest 4:3 window
    cells = [([(ours, PICK)], a, "Our mask", PICK), ([], eo, "Our edit", PICK),
             ([], eb, "Box + SAM 3 edit", BASE)]
    out = [body]
    for k, (fills, im, lab, c) in enumerate(cells):
        cx = x + k * (cw + 8)
        out.append(img_tag(cx, iy, cw, ch, cell(im, full, (int(cw * 2), int(ch * 2)), fills, r=4)))
        out.append(text(cx, iy + ch + 27, lab, F_LAB, 700, hexc(c)))
    mi = s["mask_iou"]
    out.append(text(x, iy + ch + 54, f"mask IoU: ours {mi['ours_softfam']:.2f}, Box + SAM 3 {mi[BASELINE]:.2f}",
                    F_SMALL, 400, MUTED))
    return "".join(out), iy + ch + 62 - y


def figure(part, reason, edit) -> tuple[str, int]:
    pad, gap, lab_w = 14, 26, 40
    cw = (W - 2 * pad - lab_w - gap) / 2
    rows, y = [], 12
    titles = ["(a) Part grounding", "(b) Reasoning", "(c) Editing"]
    for t, (kind, exs) in zip(titles, [("part", part), ("reason", reason), ("edit", edit)]):
        hs, parts = [], []
        for k, s in enumerate(exs):
            cx = pad + lab_w + k * (cw + gap)
            svg, h = edit_card(s, cx, y, cw) if kind == "edit" else grounding_card(s, cx, y, cw, kind)
            parts.append(svg)
            hs.append(h)
        h = max(hs)
        rows.append(f'<g transform="rotate(-90 {pad + 16} {y + h / 2})">' + text(pad + 16, y + h / 2 + 9, t, F_ROW, 700,
                                                                                INK, "middle") + "</g>")
        rows.append(f'<line x1="{pad + lab_w + cw + gap / 2}" y1="{y}" x2="{pad + lab_w + cw + gap / 2}" '
                    f'y2="{y + h}" stroke="{LINE}" stroke-width="1.2"/>')
        rows += parts
        y += h + 14
        if kind != "edit":
            rows.append(f'<line x1="{pad}" y1="{y - 7}" x2="{W - pad}" y2="{y - 7}" stroke="{LINE}" stroke-width="1.2"/>')
    H = int(y)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}"><rect width="{W}" height="{H}" fill="#fff"/>{"".join(rows)}</svg>'), H


def render(svg_path: Path, pdf_path: Path, png_path: Path | None, H: int) -> None:
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
    ap.add_argument("--part", type=int, nargs=2, default=[13, 9])
    ap.add_argument("--reason", type=int, nargs=2, default=[6, 14])
    ap.add_argument("--edit", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--out", default=str(HERE / "qualitative" / "showcase"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    S = {f: json.loads((ROOT / f"results/figures/qual_{f}_showcase.json").read_text())["samples"] for f in ("part", "reason", "edit")}
    for f, idx in (("part", args.part), ("reason", args.reason)):
        for i in idx:                                  # the figure shows the scored answer: recomputed IoU = run IoU
            s = S[f][i]
            if abs(s["run_iou"]["ours"] - s["methods"]["ours"]["iou"]) > 0.02 or \
                    abs(s["run_iou"][BASELINE] - s["methods"][BASELINE]["iou"]) > 0.02:
                raise SystemExit(f"{f} #{i}: recomputed IoU differs from the run's")
    svg, H = figure([S["part"][i] for i in args.part], [S["reason"][i] for i in args.reason],
                    [S["edit"][i] for i in args.edit])
    out = Path(args.out)
    out.with_suffix(".svg").write_text(svg)
    render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None, H)
    print(out, H)


if __name__ == "__main__":
    main()
