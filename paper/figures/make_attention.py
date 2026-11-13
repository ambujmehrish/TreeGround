"""Single-column figure: region-restricted attention.

Top: the image-token grid over a real PACO test image; the cells two mask tokens may attend to (the cells their mask
covers plus a one-cell margin, tree_attn.REGION_DILATE). Bottom: what one mask token attends to in the sequence.
Masks are real pool entries of results/figures/overview_assets_mug_handle.json (the model's pick and its tree parent);
the grid is drawn at 32 px per token, the merged patch size of Qwen3-VL (the processor's exact resize is not applied).
usage: MJ_DIR=<dir with node_modules> python paper/figures/make_attention.py --png
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import make_method as MM  # noqa: E402
from make_method import BLUE, FRAME, GREEN, INK, LINE, MUTED, PINK, TINT, Math, arrow, img_tag, rect, text  # noqa: E402
from make_overview import dec, png_b64, text_width, typeset  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground.tree_attn import REGION_DILATE  # noqa: E402

W, H = 700, 822
CELL_PX = 32
F_T, F_B, F_S = 26, 24, 23                      # single column: 1 px ~ 0.34 pt, so 7 pt and up
TEX = {
    "xv": r"\hat{x}_v", "xu": r"\hat{x}_u", "v": r"v", "mv": r"m_v",
    "rv": r"\mathcal{R}_v", "ru": r"\mathcal{R}_u",
    "jin": r"j\in\mathcal{R}_v", "jout": r"j\notin\mathcal{R}_v", "ninf": r"-\infty",
    "bias": r"b_{\ell,\,\mathrm{rel}(u,v)}",
}


def region(m: np.ndarray, gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    # (cells the mask covers, cells the token may attend to) as (gh, gw) bools: tree_attn.build_nodes
    own = (mllm.coverage(m, gh, gw) > 0).reshape(gh, gw)
    k = np.ones((2 * REGION_DILATE + 1,) * 2, np.uint8)
    return own, cv2.dilate(own.astype(np.uint8), k).astype(bool)


def grid_panel(img, handle, mug, x0, y0, side) -> tuple[list[str], dict]:
    h, w = img.shape[:2]
    gh, gw = round(h / CELL_PX), round(w / CELL_PX)
    own_h, reg_h = region(handle, gh, gw)
    own_m, reg_m = region(mug, gh, gw)
    rr, cc = np.nonzero(reg_m)
    n = max(rr.max() - rr.min(), cc.max() - cc.min()) + 3        # a square window of cells around the mug's region
    r0 = int(np.clip(rr.min() - 1, 0, gh - n))
    c0 = int(np.clip(cc.min() - 1, 0, gw - n))
    sy, sx = h / gh, w / gw
    crop = (round(c0 * sx), round(r0 * sy), round((c0 + n) * sx), round((r0 + n) * sy))
    x = img.astype(float)
    x = np.where(handle[..., None], 0.45 * x + 0.55 * np.array((176, 48, 111)), x)
    im = Image.fromarray(x.clip(0, 255).astype(np.uint8)).crop(crop).resize((2 * side, 2 * side), Image.LANCZOS)
    out = [img_tag(x0, y0, side, side, png_b64(im), 8)]
    c = side / n
    for r in range(n):
        for k in range(n):
            R, C = r0 + r, c0 + k
            cx, cy = x0 + k * c, y0 + r * c
            if reg_h[R, C]:
                out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{c:.1f}" height="{c:.1f}" fill="{PINK}" '
                           f'fill-opacity="{0.42 if own_h[R, C] else 0.2}"/>')
            elif not reg_m[R, C]:
                out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{c:.1f}" height="{c:.1f}" fill="#fff" '
                           f'fill-opacity="0.62"/>')
    for k in range(n + 1):
        out.append(f'<path d="M{x0 + k * c:.1f},{y0} v{side} M{x0},{y0 + k * c:.1f} h{side}" stroke="#fff" '
                   f'stroke-opacity="0.75" stroke-width="1"/>')

    def outline(reg, color, sw):
        seg = []
        for r in range(n):
            for k in range(n):
                if not reg[r0 + r, c0 + k]:
                    continue
                cx, cy = x0 + k * c, y0 + r * c
                for dr, dk, d in ((-1, 0, f"M{cx:.1f},{cy:.1f} h{c:.1f}"), (1, 0, f"M{cx:.1f},{cy + c:.1f} h{c:.1f}"),
                                  (0, -1, f"M{cx:.1f},{cy:.1f} v{c:.1f}"), (0, 1, f"M{cx + c:.1f},{cy:.1f} v{c:.1f}")):
                    rr_, kk_ = r0 + r + dr, c0 + k + dk
                    if not (0 <= rr_ < gh and 0 <= kk_ < gw) or not reg[rr_, kk_] or not (0 <= r + dr < n and 0 <= k + dk < n):
                        seg.append(d)
        return f'<path d="{" ".join(seg)}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'

    out.append(outline(reg_m, BLUE, 3.2))
    out.append(outline(reg_h, PINK, 3.2))
    out.append(rect(x0, y0, side, side, "none", LINE, 1.2, 8))
    rh, ch_ = np.nonzero(reg_h)
    anchor = {"handle": (x0 + (ch_.max() + 1 - c0) * c, y0 + (rh.mean() + 0.5 - r0) * c),
              "mug": (x0 + (cc.max() + 1 - c0) * c, y0 + (rr.max() + 0.2 - r0) * c)}
    return out, anchor


def legend(M, x, y, anchor) -> list[str]:
    out = []
    rows = [("xv", PINK, "handle", "rv", ["cells under", "+ 1-cell margin"], "mv"),
            ("xu", BLUE, "mug", "ru", ["region of the parent", "mask (the mug)"], None)]
    for (key, c, name, rkey, lines, mkey), yy in zip(rows, (y, y + 160)):
        out.append(rect(x, yy, 84, 54, TINT[c], c, 2.8, 10) + M(key, x + 42, yy + 38, 32, INK, "middle"))
        out.append(text(x + 100, yy + 37, "sees", 26, 400, INK) + M(rkey, x + 100 + text_width("sees ", 26, 400), yy + 38, 33, c))
        out.append(text(x, yy + 88, lines[0], F_B, 400, MUTED)
                   + (M(mkey, x + text_width(lines[0] + " ", F_B, 400), yy + 89, 29, MUTED) if mkey else ""))
        out.append(text(x, yy + 118, lines[1], F_B, 400, MUTED))
        ax, ay = anchor[name]
        out.append(arrow([(x - 8, yy + 27), (ax + 4, ay)], c, 2.4))
    return out


def sequence(M, x0, y0) -> list[str]:
    # image | query <SEL> | mask tokens; arcs from the mask token v (the handle)
    out, x, pos = [], x0, {}
    cells = [("i", GREEN)] * 5 + [("gap", None)] + [("q", PINK)] * 1 + [("sel", PINK), ("gap", None), ("u", BLUE),
                                                                         ("v", PINK), ("m", BLUE)]
    own = (0, 1)                                             # image tokens j in R_v
    k_img, tw, th = 0, 46, 48
    for kind, c in cells:
        if kind == "gap":
            x += 16
            continue
        w = 84 if kind == "sel" else tw
        if kind == "sel":
            out.append(rect(x, y0, w, th, PINK, PINK, 1, 7) + text(x + w / 2, y0 + 32, "<SEL>", 20, 700, "#fff", "middle"))
        else:
            hit = kind == "i" and k_img in own
            fill = PINK if hit else "#fff" if kind == "v" else TINT[c]
            op = ' fill-opacity="0.45"' if hit else ""
            out.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{th}" rx="7" fill="{fill}"{op} stroke="{c}" '
                       f'stroke-width="{3.4 if kind == "v" else 2.2}"/>')
        key = f"i{k_img}" if kind == "i" else f"q{sum(k.startswith('q') for k in pos)}" if kind == "q" else kind
        if kind in ("u", "v", "m"):
            pos.setdefault("m_first", x)
        pos[key] = x + w / 2
        k_img += kind == "i"
        x += w + 6
    out.append(M("xu", pos["u"], y0 + 34, 26, INK, "middle") + M("xv", pos["v"], y0 + 34, 26, INK, "middle"))
    for a, b, lab in ((pos["i0"] - tw / 2, pos["i4"] + tw / 2, "image tokens"), (pos["q0"] - tw / 2, pos["sel"] + 42, "query"),
                      (pos["m_first"], x - 6, "mask tokens")):
        out.append(f'<path d="M{a:.1f},{y0 + th + 8} v7 H{b:.1f} v-7" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
        out.append(text((a + b) / 2, y0 + th + 40, lab, F_B, 600, MUTED, "middle"))
    hx = pos["v"]

    def arc(tx, hgt, color, dash=None, sw=2.4):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<path d="M{hx:.1f},{y0 - 3} C{hx:.1f},{y0 - hgt} {tx:.1f},{y0 - hgt} {tx:.1f},{y0 - 5}" fill="none" '
                f'stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{MM.marker_id(color)})"/>')

    for k in own:
        out.append(arc(pos[f"i{k}"], 230, PINK))
    out.append(arc(pos["i4"], 160, FRAME, "6 5", 2))
    for k in ("q0", "sel"):
        out.append(arc(pos[k], 92, PINK, None, 2))
    out.append(arc(pos["u"], 44, BLUE, None, 2.6))
    out.append(M("jin", x0 + 2, y0 - 172, 31, PINK))
    out.append(M("jout", pos["i4"] - 20, y0 - 26, 27, MUTED, "end"))
    out.append(M("bias", hx + 22, y0 - 22, 27, BLUE))
    # image and query tokens never attend to mask tokens
    yb = y0 + th + 58
    out.append(f'<path d="M{pos["sel"]:.1f},{yb} C{pos["sel"]:.1f},{yb + 46} {pos["v"]:.1f},{yb + 46} {pos["v"]:.1f},{yb + 4}" '
               f'fill="none" stroke="{FRAME}" stroke-width="2" stroke-dasharray="6 5" '
               f'marker-end="url(#{MM.marker_id(FRAME)})"/>')
    out.append(M("ninf", (pos["sel"] + pos["v"]) / 2, yb + 70, 28, MUTED, "middle"))
    out.append(text(x0, yb + 30, "image, query tokens", F_B, 400, MUTED))
    out.append(text(x0, yb + 60, "never see mask tokens", F_B, 400, MUTED))
    return out


def build(ex, img, M) -> str:
    pool = [dec(r) for r in ex["pool"]]
    handle = pool[ex["pick"]]
    mug = pool[ex["parents"][ex["pick"]]] if 0 <= ex["parents"][ex["pick"]] < len(pool) else pool[257]
    body = [f'<rect width="{W}" height="{H}" fill="#fff"/>', "<defs>" + MM.markers([INK, BLUE, PINK, GREEN, FRAME]) + "</defs>"]
    body.append(text(12, 30, "(a) Region of a mask token on the token grid", F_T, 700, INK))
    g, anchor = grid_panel(img, handle, mug, 12, 46, 330)
    body += g + legend(M, 384, 70, anchor)
    body.append(text(12, 420, "(b) Attention of mask token", F_T, 700, INK)
                + M("v", 12 + text_width("(b) Attention of mask token ", F_T, 700), 421, 27, INK))
    body += sequence(M, 14, 634)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{MM.FONT}">{"".join(body)}</svg>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ROOT / "results/figures/overview_assets_mug_handle.json"))
    ap.add_argument("--example", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "attention"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    ex = json.loads(Path(args.assets).read_text())["examples"][args.example]
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(ex["image_jpeg_b64"]))).convert("RGB"))
    M = Math(typeset(list(TEX.values())))
    M.t = {k: M.t[v] for k, v in TEX.items()}
    MM.W, MM.H = W, H
    out = Path(args.out)
    out.with_suffix(".svg").write_text(build(ex, img, M))
    MM.render(out.with_suffix(".svg"), out.with_suffix(".pdf"), out.with_suffix(".png") if args.png else None)
    print(out.with_suffix(".pdf"), "parent of the pick:", ex["parents"][ex["pick"]])


if __name__ == "__main__":
    main()
