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
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.9), gridspec_kw={"height_ratios": [1.2, 1], "hspace": 1.05})
    # (a) grouped bars
    w = 0.24
    for k, (name, col) in enumerate(zip(GIOU, (GREY, PINK, DARK))):
        xs = [i + (k - 1) * (w + 0.03) for i in range(2)]
        bars = a.bar(xs, GIOU[name], w, color=col, label=name, linewidth=0)
        for r, v in zip(bars, GIOU[name]):
            a.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                   color=INK, fontweight="bold" if name == "Ours" else "normal")
    a.set_xticks(range(2), [f"{n}\n({g}% of gap closed)" for n, g in zip(BENCH, GAP)])
    a.set_xlim(-0.55, 1.55)
    a.set_ylim(0, 92)
    a.set_yticks(range(0, 81, 20))
    a.set_ylabel("gIoU (%)", color=INK)
    a.set_title("(a) Selection versus the best mask in the pool", fontsize=10, fontweight="bold", color=INK, loc="left",
                pad=24, x=-0.72)
    a.legend(ncol=3, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(-0.74, 1.0), handlelength=0.9,
             handletextpad=0.35, columnspacing=0.9, borderpad=0)
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
    b.set_xticks(range(0, 101, 25))
    b.set_xlabel("share of queries (%)", color=INK)
    b.set_title("(b) Source of the mask", fontsize=10, fontweight="bold", color=INK, loc="left", pad=24, x=-0.72)
    b.legend(ncol=3, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(-0.74, 1.0), handlelength=0.9,
             handletextpad=0.35, columnspacing=0.9, borderpad=0)
    style(b)
    b.yaxis.grid(False)
    b.xaxis.grid(True, color=GRID, linewidth=0.6)
    b.tick_params(axis="y", length=0, labelcolor=INK)
    fig.savefig(HERE / "coverage_cropped.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "coverage.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
    print(HERE / "coverage_cropped.pdf")


if __name__ == "__main__":
    main()
