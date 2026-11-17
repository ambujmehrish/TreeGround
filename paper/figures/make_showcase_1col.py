"""Single-column version of the main qualitative figure (make_showcase.py): same test examples and model outputs,
laid out so that the text is 5.5-6.5 pt at \\columnwidth.

Method names move to one colour legend, and each IoU is printed on or beside its image, so no label has to fit
under a small image. Part examples keep the three top-scored candidates with their softmax shares.
usage: python paper/figures/make_showcase_1col.py --png
"""
import argparse
import json
from pathlib import Path

import make_showcase as MS
from make_overview import bbox, text_width
from make_teaser import crop_box, hexc, text, wrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = 700                                             # at \columnwidth (237 pt) 1 px ~ 0.34 pt
F_ROW, F_Q, F_LAB, F_NUM = 19, 18, 17, 16
INK, MUTED, LINE, BUBBLE = MS.INK, MS.MUTED, MS.LINE, MS.BUBBLE
PICK, BASE, GT = MS.PICK, MS.BASE, MS.GT
SA2VA, ALT = (217, 130, 43), (91, 100, 116)
ASPECT = 3 / 2
GAP = 6
# a long request is abridged for the single-column figure; the cut is marked in the text
ABRIDGED = {"We cannot breathe underwater": "We cannot breathe underwater […] What in the picture can help human "
                                            "accomplish this task?"}


def shown(q: str) -> str:
    q = q.replace("Locate ", "").replace(" <SEL>", "")
    q = next((v for k, v in ABRIDGED.items() if q.startswith(k)), q)
    return q[0].upper() + q[1:]


def bubble_h(s, w) -> float:
    return 12 + len(wrap(s, F_Q, w - 20)) * (F_Q + 4)


def bubble(x, y, w, s, h=0) -> tuple[str, float]:
    # h: the height of the tallest request in the row, so that the two cards of a row line up
    lines = wrap(s, F_Q, w - 20)
    own = 12 + len(lines) * (F_Q + 4)
    h = max(h, own)
    y0 = y + (h - own) / 2
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{BUBBLE}"/>']
    out += [text(x + 10, y0 + 5 + (i + 1) * (F_Q + 4) - 4, l, F_Q, 400, INK, italic=True) for i, l in enumerate(lines)]
    return "".join(out), h


def tag(x, y, s, color, corner="tl", size=F_NUM) -> str:
    # filled label anchored at a corner point of an image
    w, h = text_width(s, size, 700) + 10, size + 6
    x0 = x if "l" in corner else x - w
    y0 = y if "t" in corner else y - h
    return (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h}" rx="4" fill="{color}"/>'
            + text(x0 + 5, y0 + size + 0.5, s, size, 700, "#fff"))


def free_corner(masks, box, fw=0.5, fh=0.22) -> str:
    # the image corner least covered by the drawn masks (ties: bottom left first)
    x0, y0, x1, y1 = box
    bw, bh = int((x1 - x0) * fw), max(1, int((y1 - y0) * fh))
    cov = {}
    for c in ("bl", "br", "tl", "tr"):
        xs = slice(x0, x0 + bw) if "l" in c else slice(x1 - bw, x1)
        ys = slice(y0, y0 + bh) if "t" in c else slice(y1 - bh, y1)
        cov[c] = sum(int(m[ys, xs].sum()) for m in masks)
    return min(cov, key=cov.get)


def picture(x, y, w, h, a, box, fills, outlines, label, color, masks) -> str:
    out = MS.img_tag(x, y, w, h, MS.cell(a, box, (int(w * 3), int(h * 3)), fills, outlines, 5), 5)
    c = free_corner(masks, box, fw=(text_width(label, F_NUM, 700) + 14) / w, fh=(F_NUM + 10) / h)
    px = x + 3 if "l" in c else x + w - 3
    py = y + 3 if "t" in c else y + h - 3
    return out + tag(px, py, label, color, c)


def grounding_card(s, x, y, w, kind, qh=0) -> tuple[str, float]:
    a = MS.load(s["image"])
    hw = a.shape[:2]
    gt, ours = MS.mask(s["gt"], hw), MS.mask(s["methods"]["ours"]["mask"], hw)
    base = MS.mask(s["methods"][MS.BASELINE]["mask"], hw)
    box = crop_box(hw[0], hw[1], MS.focus([gt, ours, base], hw), ASPECT, margin=0.45, min_frac=0.6)
    body, bh = bubble(x, y, w, shown(s["query"]), qh)
    iy = y + bh + GAP
    # reasoning: two stacked baselines of the main picture's aspect fill the card width exactly
    mw = round(w * 0.6) if kind == "part" else (w - GAP + GAP * ASPECT / 2) / 1.5
    mh = mw / ASPECT
    sx, sw = x + mw + GAP, w - mw - GAP
    out = [body, picture(x, iy, mw, mh, a, box, [(ours, PICK)], [], f"{s['methods']['ours']['iou']:.2f}",
                         hexc(PICK), [ours])]

    def baseline(by, bh_, name, color):
        m = MS.mask(s["methods"][name]["mask"], hw)
        bw = min(bh_ * ASPECT, sw)
        out.append(picture(sx + sw - bw, by, bw, bh_, a, box, [(m, color)], [(gt, GT)],
                           f"{s['methods'][name]['iou']:.2f}", hexc(color), [m, gt]))

    if kind == "part":
        t = (sw - 2 * 4) / 3
        for k, c in enumerate(s["candidates"][:3]):
            col = PICK if k == 0 else ALT
            tx = x + mw + GAP + k * (t + 4)
            out.append(MS.img_tag(tx, iy, t, t, MS.thumb(a, MS.mask(c["mask"], hw), col, int(t * 1.5)), 4))
            out.append(text(tx + t / 2, iy + t + F_NUM, f"{c['p'] * 100:.0f}%", F_NUM, 700 if k == 0 else 500,
                            hexc(col), "middle"))
        top = t + F_NUM + 7
        baseline(iy + top, mh - top, MS.BASELINE, BASE)
    else:
        bh_ = (mh - GAP) / 2
        baseline(iy, bh_, MS.BASELINE, BASE)
        baseline(iy + bh_ + GAP, bh_, "sa2va", SA2VA)
    return "".join(out), iy + mh - y


def edit_card(s, x, y, w, qh=0) -> tuple[str, float]:
    a = MS.load(s["image"])
    hw = a.shape[:2]
    ours = MS.mask(s["masks"]["ours_softfam"], hw)
    eo = MS.load(s["edits"]["qwen_image_edit/ours_softfam_blend"])
    eb = MS.load(s["edits"]["qwen_image_edit/qwen_box_sam_blend"])
    body, bh = bubble(x, y, w, shown(s["query"]), qh)
    iy = y + bh + GAP
    cw = (w - 2 * GAP) / 3
    ch = cw / ASPECT
    full = crop_box(hw[0], hw[1], MS.focus([ours], hw), ASPECT, margin=0.1, min_frac=1.0)
    mi = s["mask_iou"]
    cells = [([(ours, PICK)], a, "Our mask", PICK, f"{mi['ours_softfam']:.2f}"), ([], eo, "Our edit", PICK, None),
             ([], eb, "Baseline edit", BASE, f"{mi[MS.BASELINE]:.2f}")]
    out = [body]
    for k, (fills, im, lab, c, iou) in enumerate(cells):
        cx = x + k * (cw + GAP)
        out.append(MS.img_tag(cx, iy, cw, ch, MS.cell(im, full, (int(cw * 3), int(ch * 3)), fills, r=4), 5))
        out.append(text(cx + cw if k == 2 else cx, iy + ch + F_LAB + 2, lab, F_LAB, 700, hexc(c),
                        "end" if k == 2 else "start"))
        if iou:
            out.append(tag(cx + 3, iy + 3, iou, hexc(c), "tl"))
    return "".join(out), iy + ch + F_LAB + 8 - y


def legend(x, y) -> str:
    out = []
    for name, col, dashed in (("Ours", PICK, False), ("Box + SAM 3", BASE, False), ("Sa2VA-4B", SA2VA, False),
                              ("ground truth", GT, True)):
        if dashed:
            out.append(f'<line x1="{x}" y1="{y - 6}" x2="{x + 20}" y2="{y - 6}" stroke="{hexc(col)}" stroke-width="3" '
                       f'stroke-dasharray="6 4"/>')
            x += 25
        else:
            out.append(f'<rect x="{x}" y="{y - 13.5}" width="14" height="14" rx="3" fill="{hexc(col)}"/>')
            x += 19
        out.append(text(x, y, name, F_LAB, 600, hexc(col)))
        x += text_width(name, F_LAB, 600) + 16
    out.append(tag(x, y - F_NUM - 2.5, "0.90", MUTED, "tl"))
    out.append(text(x + text_width("0.90", F_NUM, 700) + 15, y, "mask IoU", F_LAB, 500, MUTED))
    return "".join(out)


def figure(part, reason, edit) -> tuple[str, int]:
    pad, gap, lab_w = 2, 14, 26
    cw = (W - 2 * pad - lab_w - gap) / 2
    rows = [legend(pad + lab_w, 17)]
    y = 28
    titles = ["(a) Part grounding", "(b) Reasoning", "(c) Editing"]
    for t, (kind, exs) in zip(titles, [("part", part), ("reason", reason), ("edit", edit)]):
        hs, parts = [], []
        qh = max(bubble_h(shown(s["query"]), cw) for s in exs)
        for k, s in enumerate(exs):
            cx = pad + lab_w + k * (cw + gap)
            svg, h = edit_card(s, cx, y, cw, qh) if kind == "edit" else grounding_card(s, cx, y, cw, kind, qh)
            parts.append(svg)
            hs.append(h)
        h = max(hs)
        rows.append(f'<g transform="rotate(-90 {pad + 12} {y + h / 2})">'
                    + text(pad + 12, y + h / 2 + 7, t, F_ROW, 700, INK, "middle") + "</g>")
        mid = pad + lab_w + cw + gap / 2
        rows.append(f'<line x1="{mid}" y1="{y}" x2="{mid}" y2="{y + h}" stroke="{LINE}" stroke-width="1"/>')
        rows += parts
        y += h + 12
        if kind != "edit":
            rows.append(f'<line x1="{pad}" y1="{y - 6}" x2="{W - pad}" y2="{y - 6}" stroke="{LINE}" stroke-width="1"/>')
    H = int(y - 8)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{MS.FONT}"><rect width="{W}" height="{H}" fill="#fff"/>{"".join(rows)}</svg>'), H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, nargs=2, default=[13, 9])
    ap.add_argument("--reason", type=int, nargs=2, default=[6, 14])
    ap.add_argument("--edit", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--out", default=str(HERE / "qualitative" / "showcase_1col"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    S = {f: json.loads((ROOT / f"results/figures/qual_{f}_showcase.json").read_text())["samples"]
         for f in ("part", "reason", "edit")}
    for f, idx in (("part", args.part), ("reason", args.reason)):
        for i in idx:                                  # the figure shows the scored answer: recomputed IoU = run IoU
            s = S[f][i]
            if abs(s["run_iou"]["ours"] - s["methods"]["ours"]["iou"]) > 0.02 or \
                    abs(s["run_iou"][MS.BASELINE] - s["methods"][MS.BASELINE]["iou"]) > 0.02:
                raise SystemExit(f"{f} #{i}: recomputed IoU differs from the run's")
    svg, H = figure([S["part"][i] for i in args.part], [S["reason"][i] for i in args.reason],
                    [S["edit"][i] for i in args.edit])
    out = Path(args.out)
    out.with_suffix(".svg").write_text(svg)
    MS.W = W
    MS.render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None, H)
    print(out, W, H, round(H / W, 3))


if __name__ == "__main__":
    main()
