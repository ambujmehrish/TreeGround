"""Supplementary qualitative pages in the style of the main qualitative figure (make_showcase.py): one task per page,
two examples per row.

Sources (all test splits, real model outputs; scripts/qual_assets.py):
  results/figures/qual_{part,reason,edit}_showcase.json   selected clear wins (as Fig. 4; reasoning with D51 rewriting)
  results/figures/qual_{part,reason,edit}.json            fixed-seed random draws, largest wins and largest losses
                                                           (reasoning without query rewriting)
Examples already in the main figure and examples whose recomputed IoU differs from the run's are left out.
usage: python paper/figures/make_supp_qual.py --png
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_showcase as MS  # noqa: E402
from make_teaser import text  # noqa: E402

OUT = HERE / "qualitative" / "supp"
MAIN = {"part": {9, 13}, "reason": {6, 14}, "edit": {0, 2}}     # indices of the showcase files used in Fig. 4
PER_PAGE = {"part": 6, "reason": 6, "edit": 6}
TITLE = {"part": "Part grounding", "reason": "Reasoning", "edit": "Editing"}
BASE_KEY = {"part": "qwen_box_sam", "reason": "qwen_box_sam"}


def load(name: str) -> list[dict]:
    return json.loads((ROOT / f"results/figures/{name}.json").read_text())["samples"]


def consistent(s: dict, fam: str) -> bool:
    # the figure must show what the tables scored: recomputed IoU = run IoU (ours, and the Box + SAM 3 baseline)
    if fam == "edit":
        return True
    r, m = s["run_iou"], s["methods"]
    base = r.get(BASE_KEY[fam], r.get("base"))
    return abs(r["ours"] - m["ours"]["iou"]) <= 0.02 and abs(base - m[BASE_KEY[fam]]["iou"]) <= 0.02


def card(s: dict, fam: str, x: float, y: float, w: float) -> tuple[str, float]:
    if fam == "edit":
        return MS.edit_card(s, x, y, w)
    return MS.grounding_card(s, x, y, w, "part" if "candidates" in s else "reason")


def page(samples: list[dict], fam: str, label: str) -> tuple[str, int]:
    pad, gap, lab_w = 14, 26, 40
    cw = (MS.W - 2 * pad - lab_w - gap) / 2
    body, y = [], 12
    for r in range(0, len(samples), 2):
        pair = samples[r:r + 2]
        out = [card(s, fam, pad + lab_w + k * (cw + gap), y, cw) for k, s in enumerate(pair)]
        h = max(o[1] for o in out)
        body += [o[0] for o in out]
        if len(pair) == 2:
            mid = pad + lab_w + cw + gap / 2
            body.append(f'<line x1="{mid}" y1="{y}" x2="{mid}" y2="{y + h}" stroke="{MS.LINE}" stroke-width="1.2"/>')
        y += h + 14
        if r + 2 < len(samples):
            body.append(f'<line x1="{pad + lab_w}" y1="{y - 7}" x2="{MS.W - pad}" y2="{y - 7}" stroke="{MS.LINE}" '
                        f'stroke-width="1.2"/>')
    H = int(y)
    body.insert(0, f'<g transform="rotate(-90 {pad + 16} {H / 2})">'
                + text(pad + 16, H / 2 + 9, label, MS.F_ROW, 700, MS.INK, "middle") + "</g>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{MS.W}" height="{H}" viewBox="0 0 {MS.W} {H}" '
            f'font-family="{MS.FONT}"><rect width="{MS.W}" height="{H}" fill="#fff"/>{"".join(body)}</svg>'), H


def sets() -> list[tuple[str, str, str, list[dict]]]:
    # (file stem, family, row label, samples)
    # every example appears once: the main figure's examples are excluded and qids are not repeated across pages;
    # selected pages are kept full (a short last page is dropped)
    out = []
    for fam in ("part", "reason", "edit"):
        show = load(f"qual_{fam}_showcase")
        seen = {show[i]["qid"] for i in MAIN[fam]}

        def fresh(items):
            keep = []
            for s in items:
                if s["qid"] not in seen and consistent(s, fam):
                    seen.add(s["qid"])
                    keep.append(s)
            return keep
        draws = load(f"qual_{fam}")
        sc = fresh(show)
        if fam == "edit":                                      # the edit showcase has few clear wins of its own
            sc += fresh([s for s in draws if s["group"] == "ours_better"])
        n = PER_PAGE[fam]
        for k in range(len(sc) // n):
            out.append((f"{fam}_selected_{k + 1}", fam, TITLE[fam], sc[k * n:(k + 1) * n]))
        for grp, tag in (("random", "random"), ("ours_worse", "failures")):
            out.append((f"{fam}_{tag}", fam, TITLE[fam], fresh([s for s in draws if s["group"] == grp])[:n]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, fam, label, samples in sets():
        if not samples:
            continue
        svg, H = page(samples, fam, label)
        p = OUT / stem
        p.with_suffix(".svg").write_text(svg)
        MS.render(p.with_suffix(".svg"), p.with_suffix(".pdf"), p.with_suffix(".png") if args.png else None, H)
        print(stem, len(samples), "examples,", H, "px")


if __name__ == "__main__":
    main()
