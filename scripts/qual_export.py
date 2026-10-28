import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e_edit_s2 as ES2  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.run import load_config, read_json, resolve_run, write_json  # noqa: E402

# Qualitative panels of region-grounded edits (no metric is computed here). Per edit set: a fixed-seed uniform
# sample ('random'), the samples where our mask beats SAM 3 text by the largest IoU margin ('ours_better') and
# where it loses the most ('ours_worse'), each labelled as such. Panel: source with the GT edit region outlined,
# our selected mask, Qwen-Image-Edit blended with our mask / with the SAM 3 text mask / without a mask.

H = 300
DATASETS = ("partedit", "gedit", "imgedit", "paco_edit")
EDITOR = "qwen_image_edit"


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit(img: Image.Image) -> Image.Image:
    return img.resize((max(1, round(img.width * H / img.height)), H), Image.LANCZOS)


def outline(image: np.ndarray, mask: np.ndarray, color) -> Image.Image:
    m = np.pad(mask, 1)
    inner = m[1:-1, 1:-1] & m[2:, 1:-1] & m[:-2, 1:-1] & m[1:-1, 2:] & m[1:-1, :-2]
    edge = mask & ~inner
    for _ in range(max(1, image.shape[0] // 250)):
        e = np.pad(edge, 1)
        edge = edge | e[2:, 1:-1] | e[:-2, 1:-1] | e[1:-1, 2:] | e[1:-1, :-2]
    out = image.copy()
    out[edge] = color
    return Image.fromarray(out)


def overlay(image: np.ndarray, mask: np.ndarray, color) -> Image.Image:
    out = image.astype(np.float32).copy()
    out[mask] = 0.45 * out[mask] + 0.55 * np.array(color, np.float32)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def shard_rows(d: Path) -> pd.DataFrame:
    files = sorted((d / "shards").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def panel(tiles: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    tiles = [(t, fit(im)) for t, im in tiles]
    gap, cap, top = 8, 26, 34
    w = sum(im.width for _, im in tiles) + gap * (len(tiles) + 1)
    canvas = Image.new("RGB", (w, H + cap + top + gap), "white")
    d = ImageDraw.Draw(canvas)
    d.text((gap, 8), title[:160], fill="black", font=font(18))
    x = gap
    for t, im in tiles:
        canvas.paste(im, (x, top))
        d.text((x, top + H + 4), t, fill="black", font=font(16))
        x += im.width + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per_group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="qual_s2long")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ES, E = ES2.ES, ES2.E
    stage = ES.X2.S2T.s2_stage(cfg)
    out = ROOT / "results" / "qualitative" / args.name
    out.mkdir(parents=True, exist_ok=True)
    index = []
    rng = np.random.default_rng(args.seed)
    for ds in DATASETS:
        items = E.evaluated(E.samples(cfg, ds), False, 0)
        by_id = {s.sample_id: s for s in items}
        ids = [s.sample_id for s in items]
        mrun = resolve_run(cfg, f"e_{EDITOR}_{ds}_{stage}_merge", "latest", False)
        dirs = read_json(mrun / "metrics.json")["images_dir"]
        sel_d14 = ES2.base_paths(cfg, EDITOR, ds, items, False)[0]
        sel_d23 = ES.cache_dir(cfg, ds, ES.identity(cfg, ds, ids, False), False)
        a, b = shard_rows(sel_d14), shard_rows(sel_d23)
        df = a[["sample_id", "sam3_rle", "sam3_iou"]].merge(b[["sample_id", "ours2_rle", "ours2_iou"]], on="sample_id")
        df["margin"] = df.ours2_iou - df.sam3_iou
        groups = {"random": df.iloc[sorted(rng.choice(len(df), size=min(args.per_group, len(df)), replace=False))],
                  "ours_better": df.sort_values("margin", ascending=False).head(args.per_group),
                  "ours_worse": df.sort_values("margin").head(args.per_group)}
        for g, sub in groups.items():
            for r in sub.itertuples():
                s = by_id[r.sample_id]
                image, gt = s.load()
                ours = json.loads(r.ours2_rle)
                ours = M.decode(ours).astype(bool) if ours else np.zeros_like(gt)
                safe = E.safe_name(s.sample_id)

                def img(kind, cond):
                    with Image.open(Path(dirs[kind]) / safe / f"{cond}.png") as im:
                        return im.convert("RGB").copy()
                tiles = [("source + GT region", outline(image, gt, (0, 220, 0))),
                         (f"our mask (IoU {r.ours2_iou:.2f})", overlay(image, ours, (230, 30, 30))),
                         ("edit with our mask", img("d23", "ours2_blend")),
                         (f"edit with SAM 3 mask (IoU {r.sam3_iou:.2f})", img("d14", "sam3text_blend")),
                         ("edit without mask", img("d14", "free"))]
                fn = f"{ds}_{g}_{safe[:40]}.jpg"
                panel(tiles, f"[{ds} | {g}] {s.instruction}").save(out / fn, "JPEG", quality=85)
                index.append({"file": fn, "dataset": ds, "group": g, "sample_id": s.sample_id,
                              "instruction": s.instruction, "referent": s.referent, "ours_iou": float(r.ours2_iou),
                              "sam3_iou": float(r.sam3_iou)})
                print(f"{fn}", flush=True)
    write_json(out / "index.json", {"stage": stage, "editor": EDITOR, "groups": ["random", "ours_better", "ours_worse"],
                                    "note": "ours_better / ours_worse are selected by IoU margin vs SAM 3 text; "
                                            "random is a fixed-seed uniform sample", "panels": index})
    print(out)


if __name__ == "__main__":
    main()
