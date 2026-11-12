"""Qualitative figures from results/figures/qual_{part,reason,edit}.json (scripts/qual_assets.py).

Every mask is a stored model output on a test query; the samples are the asset job's seeded random picks and its
largest-margin picks (ours vs Box + SAM 3), labelled as such. Rows = queries, columns = methods; the ground truth is
drawn as a dashed outline on every prediction cell, the IoU under each cell is the asset job's recomputation of that
mask (checked there against the table's run: equal for ours, within 0.01 for the baselines; SAM 3 at its released
threshold, as in Tab. 1).
usage: python paper/figures/make_qual.py part [--rows instructpart:ours_better:0,...] [--name qual_part_main] [--png]
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
from make_overview import INTER, bbox, dec  # noqa: E402
from make_teaser import FONT, INK, LINE, MUTED, PICK, crop_box, paint, text, wrap  # noqa: E402

GT = (36, 150, 72)
CELL_W, CELL_H, GAP, PAD = 230, 172, 8, 14
F_HEAD, F_QUERY, F_SMALL = 17, 15, 14
COLUMNS = {
    "part": [("ours", "Ours"), ("qwen_box_sam", "Box + SAM 3"), ("sam3_text", "SAM 3 text"), ("sa2va", "Sa2VA")],
    "reason": [("ours", "Ours"), ("qwen_box_sam", "Box + SAM 3"), ("sa2va", "Sa2VA"),
               ("sam3_text_na", "SAM 3 (always answers)")],
}
BENCH = {"instructpart": "InstructPart", "partimagenet": "PartImageNet", "pascalpart": "PascalPart-116",
         "instructpart_human": "InstructPart (human)", "reasonseg": "ReasonSeg",
         "gedit_full": "GEdit", "imgedit_full": "ImgEdit"}
GROUP = {"random": "random", "ours_better": "ours better", "ours_worse": "ours worse"}
EDIT_COLS = [("gt", "GT region"), ("ours_softfam", "Ours"), ("qwen_box_sam", "Box + SAM 3"),
             ("sam3text", "SAM 3 text"), ("free", "source / no-mask edit")]
EDIT_COND = {"gt": "gt", "ours_softfam": "ours_softfam", "qwen_box_sam": "qwen_box_sam", "sam3text": "sam3text"}


def jpeg(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def mask(r, hw) -> np.ndarray:
    return dec(r) if r else np.zeros(hw, bool)


def clean(q: str) -> str:
    q = q.replace("<SEL>", "").strip()
    return q[len("Locate "):].rstrip(".") if q.startswith("Locate ") else q


def jpeg_b64(im: Image.Image, quality: int = 90) -> str:
    # photos with overlays: JPEG keeps the PDFs within the conference size limits (PNG cells were ~1 MB each)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=quality, subsampling=0)
    return base64.b64encode(b.getvalue()).decode()


def image_tag(x, y, w, h, b64) -> str:
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" '
            f'href="data:image/jpeg;base64,{b64}"/>')


def cell(img: np.ndarray, pred: np.ndarray | None, gt: np.ndarray, box, color=PICK) -> str:
    arr = paint(img, [pred] if pred is not None and pred.any() else [], [color], fill=0.5, line=2,
                dashed=(gt, GT) if gt.any() else None)
    return jpeg_b64(Image.fromarray(arr).crop(box).resize((CELL_W * 2, CELL_H * 2), Image.LANCZOS))


def fit_box(h: int, w: int) -> tuple:
    # the whole image, padded to the cell aspect by cropping the long side as little as possible
    return crop_box(h, w, (0, 0, w, h), CELL_W / CELL_H, margin=0.0, min_frac=1.0)


def grid_svg(rows: list[dict], heads: list[str]) -> tuple[str, int, int]:
    ncol = len(heads)
    width = PAD * 2 + ncol * CELL_W + (ncol - 1) * GAP
    out, y = [], PAD + F_HEAD
    for j, h in enumerate(heads):
        out.append(text(PAD + j * (CELL_W + GAP) + CELL_W / 2, y, h, F_HEAD, 600, INK, "middle"))
    y += 10
    for r in rows:
        lines = wrap(r["title"], F_QUERY, width - 2 * PAD)
        for ln in lines:
            y += F_QUERY + 4
            out.append(text(PAD, y, ln, F_QUERY, 400, INK))
        out.append(text(width - PAD, y, r["tag"], F_SMALL, 400, MUTED, "end", italic=True))
        y += 6
        for cy, line in enumerate(r["cells"]):
            for j, c in enumerate(line):
                x = PAD + j * (CELL_W + GAP)
                if c is None:
                    continue
                out.append(image_tag(x, y, CELL_W, CELL_H, c["png"]))
                if c.get("note"):
                    out.append(text(x + CELL_W / 2, y + CELL_H + F_SMALL + 2, c["note"], F_SMALL, 400,
                                    c.get("note_color", MUTED), "middle"))
            y += CELL_H + (F_SMALL + 8 if any(c and c.get("note") for c in line) else GAP)
        y += 6
        out.append(f'<line x1="{PAD}" y1="{y}" x2="{width - PAD}" y2="{y}" stroke="{LINE}" stroke-width="1"/>')
        y += 4
    height = y + PAD
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="{FONT}"><rect width="100%" height="100%" fill="white"/>{"".join(out)}</svg>')
    return svg, width, height


NO_ANSWER = {"sam3_text": "no mask (below threshold)", "qwen_box_sam": "no box returned", "sa2va": "no mask"}


def iou_note(m: str, v: dict) -> tuple[str, str]:
    if v.get("accepted") is False or (v["mask"] is None and v["iou"] == 0):
        return NO_ANSWER.get(m, "no mask"), MUTED
    return f"IoU {v['iou']:.2f}", INK


def grounding_rows(samples: list[dict], cols) -> list[dict]:
    rows = []
    for s in samples:
        img = jpeg(s["image"])
        hw = img.shape[:2]
        gt = mask(s["gt"], hw)
        preds = {m: mask(s["methods"][m]["mask"], hw) for m, _ in cols}
        shown = {m: (preds[m] if s["methods"][m].get("accepted", True) else None) for m, _ in cols}
        # framed on the target (the IoU under each cell covers whatever falls outside the frame)
        box = crop_box(hw[0], hw[1], bbox(gt), CELL_W / CELL_H, margin=0.8, min_frac=0.6)
        line = [{"png": cell(img, None, gt, box), "note": " "}]
        for m, _ in cols:
            note, color = iou_note(m, s["methods"][m])
            line.append({"png": cell(img, shown[m], gt, box), "note": note, "note_color": color})
        rows.append({"title": f"“{clean(s['query'])}”", "tag": f"{BENCH[s['benchmark']]} · {GROUP[s['group']]}",
                     "cells": [line]})
    return rows


def edit_rows(samples: list[dict], editor: str) -> list[dict]:
    rows = []
    for s in samples:
        img = jpeg(s["image"])
        hw = img.shape[:2]
        gt = mask(s["gt"], hw)
        box = fit_box(*hw)
        masks, edits = [], []
        for k, _ in EDIT_COLS:
            if k == "free":
                masks.append({"png": cell(img, None, np.zeros(hw, bool), box), "note": "source"})
            elif k == "gt":
                masks.append({"png": cell(img, gt, np.zeros(hw, bool), box, GT), "note": "GT edit region",
                              "note_color": GT})
            else:
                m = mask(s["masks"][k], hw)
                masks.append({"png": cell(img, m, gt, box),
                              "note": f"mask IoU {s['mask_iou'][k]:.2f}" if m.any() else "no mask"})
            cond = "free" if k == "free" else f"{EDIT_COND[k]}_{'blend' if editor == 'qwen_image_edit' else 'fill'}"
            e = s["edits"].get(f"{editor}/{cond}")
            edits.append(None if e is None else {"png": cell(jpeg(e), None, np.zeros(hw, bool), fit_box(*jpeg(e).shape[:2]))})
        rows.append({"title": f"“{s['query']}”  (target: {s['referent']})",
                     "tag": f"{BENCH[s['benchmark']]} · {GROUP[s['group']]}", "cells": [masks, edits]})
    return rows


def pick(samples: list[dict], spec: str | None) -> list[dict]:
    if not spec:
        return samples
    out = []
    for item in spec.split(","):
        bench, group, k = item.split(":")
        sel = [s for s in samples if s["benchmark"] == bench and s["group"] == group]
        out.append(sel[int(k)])
    return out


def render(svg: str, w: int, h: int, stem: Path, png: bool) -> None:
    stem.with_suffix(".svg").write_text(svg)
    js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
          f"'{stem.with_suffix('.svg')}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:"
          f"{{width:{w},height:{h}}},deviceScaleFactor:2}});await p.setContent(`<!doctype html><html><head><style>"
          f"@font-face{{font-family:Inter;src:url(file://{INTER[400]})}}@font-face{{font-family:Inter;font-weight:600;"
          f"src:url(file://{INTER[600]})}}@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0}}</style></head><body>"
          f"${{svg}}</body></html>`);"
          + (f"await p.screenshot({{path:'{stem.with_suffix('.png')}',clip:{{x:0,y:0,width:{w},height:{h}}}}});"
             if png else "")
          + f"await p.pdf({{path:'{stem.with_suffix('.pdf')}',width:'{w}px',height:'{h}px',printBackground:true,"
          "pageRanges:'1'});await b.close();})();")
    subprocess.run(["node", "-e", js], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=("part", "reason", "edit"))
    ap.add_argument("--rows", help="bench:group:k,... (default: every sample)")
    ap.add_argument("--editor", default="qwen_image_edit", choices=("qwen_image_edit", "flux_fill"))
    ap.add_argument("--name")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    data = json.loads((ROOT / "results" / "figures" / f"qual_{args.family}.json").read_text())
    samples = pick(data["samples"], args.rows)
    if args.family == "edit":
        rows, heads = edit_rows(samples, args.editor), [h for _, h in EDIT_COLS]
    else:
        cols = COLUMNS[args.family]
        rows, heads = grounding_rows(samples, cols), ["Ground truth"] + [h for _, h in cols]
    svg, w, h = grid_svg(rows, heads)
    render(svg, w, h, HERE / (args.name or f"qual_{args.family}"), args.png)
    print(HERE / (args.name or f"qual_{args.family}"), w, h)


if __name__ == "__main__":
    main()
