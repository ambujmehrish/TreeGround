"""Single-column bar charts replacing the selection-versus-coverage table (tab4_coverage.tex; same test-split numbers).

(a) gIoU of SAM 3's top instance, of our pick and of the best mask in the pool (oracle), per benchmark.
(b) source of our pick and of the best mask (point grid / SAM 3 text), as shares of the queries.
Colours validated with the dataviz palette script (blue / pink / orange on white).
usage: python paper/figures/make_coverage.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
for f in ("Inter-Regular.otf", "Inter-SemiBold.otf", "Inter-Bold.otf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/opentype/inter/{f}")
plt.rcParams.update({"font.family": "Inter", "font.size": 9.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})

INK, MUTED, GRID = "#1f2633", "#5b6474", "#e3e6eb"
GREY, PINK, BLUE, ORANGE, DARK = "#a9b0bb", "#b0306f", "#3b6fb6", "#d9822b", "#2b3240"
BENCH = ["PARTONOMY", "InstructPart"]
GIOU = {"SAM 3 top-1": [42.0, 67.4], "Ours": [49.4, 68.2], "Best in pool": [71.0, 80.0]}
GAP = [26, 7]                                     # share of the SAM 3 -> oracle gap our selection closes (%)
PICK = [(35.3, 57.7, 7.0), (28.0, 65.2, 6.8)]     # our pick: point grid, SAM 3 part phrase, SAM 3 object phrase
BEST = [(58.8, 41.2), (52.3, 47.7)]               # best mask: point grid, SAM 3 text


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    fig = plt.figure(figsize=(3.4, 3.95))
    a = fig.add_axes([0.13, 0.565, 0.85, 0.295])           # (a) uses the full column width
    b = fig.add_axes([0.40, 0.085, 0.52, 0.235])
    # (a) grouped bars: SAM 3 top-1, ours, oracle (hatched: not a method), with the gain and the remaining gap marked
    w = 0.25
    step = w + 0.03
    for k, (name, col) in enumerate(zip(GIOU, (GREY, PINK, DARK))):
        xs = [i + (k - 1) * step for i in range(2)]
        oracle = name == "Best in pool"
        bars = a.bar(xs, GIOU[name], w, color="#fff" if oracle else col, label=name, linewidth=1.0 if oracle else 0,
                     edgecolor=DARK if oracle else "none", hatch="////" if oracle else None)
        for r, v in zip(bars, GIOU[name]):
            a.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                   color=INK, fontweight="bold" if name == "Ours" else "normal")
    for i in range(2):
        sam, ours, best = (GIOU[n][i] for n in GIOU)
        # the oracle level across the group, and the part of the gap that is still open
        a.plot([i - 1.5 * step, i + 0.5 * step], [best, best], color=DARK, linewidth=0.8, linestyle=(0, (3, 2)))
        a.text(i - 1.5 * step, best + 1.2, f"gap {best - ours:.1f}", ha="left", va="bottom", fontsize=8,
               color=MUTED)
        a.plot([i], [ours / 2], marker="*", markersize=9, color="#fff", markeredgewidth=0)
    a.set_xticks(range(2), [f"{n}\n({g}% of gap closed)" for n, g in zip(BENCH, GAP)])
    a.set_xlim(-0.55, 1.55)
    a.set_ylim(0, 96)
    a.set_yticks(range(0, 81, 20))
    a.set_ylabel("gIoU (%)", color=INK)
    fig.text(0.0, 0.975, "(a) Selection versus the best mask in the pool", fontsize=10, fontweight="bold", color=INK,
             va="top")
    a.legend(ncol=3, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.875), handlelength=1.1,
             handletextpad=0.35, columnspacing=1.0, borderpad=0, bbox_transform=fig.transFigure)
    style(a)
    # (b) horizontal stacked bars, shares of the queries
    rows = [("PARTONOMY: our pick", PICK[0]), ("best mask", BEST[0] + (0,)),
            ("InstructPart: our pick", PICK[1]), ("best mask", BEST[1] + (0,))]
    ys = [3.5, 2.6, 1.2, 0.3]
    left = [0.0] * 4
    for k, (name, col) in enumerate((("point grid", BLUE), ("SAM 3 text", PINK), ("object phrase", ORANGE))):
        vals = [r[1][k] for r in rows]
        b.barh(ys, vals, 0.72, left=left, color=col, label=name, linewidth=0.8, edgecolor="#fff")
        for y, v, lf in zip(ys, vals, left):
            if v >= 12:
                b.text(lf + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="#fff", fontweight="bold")
            elif v > 0:
                b.text(101.5, y, f"{v:.0f}%", ha="left", va="center", fontsize=8.5, color=MUTED)
        left = [lf + v for lf, v in zip(left, vals)]
    b.set_yticks(ys, [r[0] for r in rows])
    b.set_xlim(0, 100)
    b.set_ylim(-0.2, 4.0)
    b.set_xticks(range(0, 101, 25))
    b.set_xlabel("share of queries (%)", color=INK)
    fig.text(0.0, 0.415, "(b) Source of the mask", fontsize=10, fontweight="bold", color=INK, va="top")
    b.legend(ncol=3, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.325), handlelength=0.9,
             handletextpad=0.35, columnspacing=0.9, borderpad=0, bbox_transform=fig.transFigure)
    style(b)
    b.yaxis.grid(False)
    b.xaxis.grid(True, color=GRID, linewidth=0.6)
    b.tick_params(axis="y", length=0, labelcolor=INK)
    fig.savefig(HERE / "coverage_cropped.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "coverage.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
    print(HERE / "coverage_cropped.pdf")


if __name__ == "__main__":
    main()
