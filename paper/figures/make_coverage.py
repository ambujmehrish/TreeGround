"""Single-column figure replacing the selection-versus-coverage table (tab4_coverage.tex; same test-split numbers).

(a) gIoU of SAM 3's own top instance, of our pick and of the best mask in the pool, per benchmark, with the share
    of the gap our selection closes and how often each pick beats the other by >= 0.1 IoU.
(b) where our pick and the best mask come from (point grid / SAM 3 text).
Colours validated with the dataviz palette script (blue / pink / orange, light surface).
usage: python paper/figures/make_coverage.py --png
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_method as MM  # noqa: E402
from make_method import BLUE, INK, LINE, MUTED, PINK, text  # noqa: E402
from make_overview import text_width  # noqa: E402

W, H = 700, 800
ORANGE, GREY = "#d9822b", "#8d96a3"
F_T, F_B, F_S = 25, 22, 20
# tab4_coverage.tex (test splits): gIoU of SAM 3 top-1 / our pick / best in pool; gap closed; ours better, SAM 3
# better (>= 0.1 IoU); source of our pick (grid, SAM 3 part phrase, SAM 3 object phrase); source of the best mask
DATA = [("PARTONOMY", 42.0, 49.4, 71.0, 26, 25.9, 11.5, (35.3, 57.7, 7.0), (58.8, 41.2)),
        ("InstructPart", 67.4, 68.2, 80.0, 7, 11.3, 11.8, (28.0, 65.2, 6.8), (52.3, 47.7))]
LO, HI = 35, 85


def bar(x, y, w, h, fill) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="4" fill="{fill}"/>'


def panel_a(x0, y0, w) -> list[str]:
    out = [text(x0, y0, "(a) The pool holds a better mask than either pick", F_T, 700, INK)]
    ax0, ax1 = x0 + 8, x0 + w - 8

    def px(v):
        return ax0 + (v - LO) / (HI - LO) * (ax1 - ax0)
    ly = y0 + 38
    for lab, kind, off in (("SAM 3 top-1", "sam", 0), ("Ours", "ours", 190), ("Best in pool", "best", 300)):
        lx = x0 + off
        out.append(marker(lx + 9, ly, kind) + text(lx + 24, ly + 7, lab, F_S, 500, MUTED))
    y = y0 + 108
    for name, sam, ours, best, gap, ob, sb, _, _ in DATA:
        out.append(text(x0, y - 22, name, F_B, 700, INK))
        out.append(text(x0 + w, y - 22, f"{gap}% of the gap closed", F_S, 600, PINK, "end"))
        out.append(f'<line x1="{ax0}" y1="{y + 16}" x2="{ax1}" y2="{y + 16}" stroke="{LINE}" stroke-width="2"/>')
        out.append(f'<line x1="{px(sam):.1f}" y1="{y + 16}" x2="{px(best):.1f}" y2="{y + 16}" stroke="{GREY}" '
                   f'stroke-width="8" stroke-linecap="round" stroke-opacity="0.45"/>')
        out.append(f'<line x1="{px(sam):.1f}" y1="{y + 16}" x2="{px(ours):.1f}" y2="{y + 16}" stroke="{PINK}" '
                   f'stroke-width="8" stroke-linecap="round"/>')
        close = px(ours) - px(sam) < 60
        for v, kind, dy, anchor, dx in ((sam, "sam", 52, "end" if close else "middle", -8 if close else 0),
                                        (ours, "ours", 52, "start" if close else "middle", 8 if close else 0),
                                        (best, "best", 52, "middle", 0)):
            out.append(marker(px(v), y + 16, kind))
            out.append(text(px(v) + dx, y + dy, f"{v:.1f}", F_S, 700 if kind == "ours" else 500, INK, anchor))
        out.append(text(x0, y + 86, f"better by ≥ 0.1 IoU:  Ours {ob:.1f}%   SAM 3 top-1 {sb:.1f}%", F_S, 400, MUTED))
        y += 158
    return out


def marker(cx, cy, kind) -> str:
    if kind == "sam":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="#fff" stroke="{GREY}" stroke-width="3"/>'
    if kind == "ours":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="{PINK}" stroke="#fff" stroke-width="2"/>'
    return (f'<rect x="{cx - 8:.1f}" y="{cy - 8:.1f}" width="16" height="16" fill="{INK}" stroke="#fff" stroke-width="2" '
            f'transform="rotate(45 {cx:.1f} {cy:.1f})"/>')


def stacked(x, y, w, parts, colors, names=None, h=32) -> list[str]:
    out, cx = [], x
    for k, (v, c) in enumerate(zip(parts, colors)):
        sw = w * v / 100
        out.append(bar(cx, y, max(sw - 2, 1), h, c))
        lab = f"{v:.0f}%" + (f" {names[k]}" if names else "")
        if sw > 62:
            out.append(text(cx + (sw - 2) / 2, y + h / 2 + 7, lab, F_S, 700, "#fff", "middle"))
        else:
            out.append(text(cx + (sw - 2) / 2, y - 6, f"{v:.0f}%", 17, 600, MUTED, "middle"))
        cx += sw
    return out


def panel_b(x0, y0, w) -> list[str]:
    out = [text(x0, y0, "(b) Where the masks come from", F_T, 700, INK)]
    ly = y0 + 38
    lx = x0
    for lab, c in (("point grid", BLUE), ("SAM 3 text", PINK), ("SAM 3 text (object phrase)", ORANGE)):
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="16" height="16" rx="4" fill="{c}"/>')
        out.append(text(lx + 22, ly + 6, lab, F_S, 500, MUTED))
        lx += 22 + text_width(lab, F_S, 500) + 18
    lab_w = 150
    y = y0 + 74
    for name, *_, pick, best in DATA:
        out.append(text(x0, y + 20, name, F_B, 700, INK))
        for k, (lab, parts, cols) in enumerate((("our pick", pick, (BLUE, PINK, ORANGE)),
                                                ("best mask", (best[0], best[1]), (BLUE, PINK)))):
            yy = y + 44 + k * 42
            out.append(text(x0 + lab_w - 12, yy + 23, lab, F_S, 500, MUTED, "end"))
            out += stacked(x0 + lab_w, yy, w - lab_w, parts, cols)
        y += 150
    return out


def build() -> str:
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>']
    body += panel_a(12, 30, W - 24)
    body += panel_b(12, 428, W - 24)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{MM.FONT}">{"".join(body)}</svg>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "coverage"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    MM.W, MM.H = W, H
    out = Path(args.out)
    out.with_suffix(".svg").write_text(build())
    MM.render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
