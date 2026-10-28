import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e_edit_s2 as ES2  # noqa: E402
from treeground import edit_metrics as EM  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.run import load_config, read_json, resolve_run, write_json  # noqa: E402

# Qualitative panels of region-grounded edits (no metric is computed here). Per edit set: a fixed-seed uniform
# sample ('random'), the samples where our mask beats SAM 3 text by the largest IoU margin ('ours_better') and
# where it loses the most ('ours_worse'), labelled as such. Every mask source edits through the same Qwen-Image-Edit
# output of the sample (D14 raw.png) blended with that mask exactly as e_edit does (dilate / feather composite), so
# columns differ only in the mask. Top row: masks (GT, ours, SAM 3 text, baselines); bottom row: the edits, plus
# the unmasked edit. Baseline masks (D13 / D31 predictors on the referent) are computed here for the shown samples.

H = 260
DATASETS = ("partedit", "gedit", "imgedit", "paco_edit")
EDITOR = "qwen_image_edit"
COLORS = {"gt": (0, 200, 0), "ours": (230, 30, 30), "sam3text": (30, 110, 230), "qwen_box_sam": (230, 140, 0),
          "sa2va": (150, 50, 200), "grounded_sam2": (0, 170, 170)}
NAMES = {"gt": "GT region (oracle)", "ours": "ours (S2 long)", "sam3text": "SAM 3 text",
         "qwen_box_sam": "Qwen3-VL box + SAM 3", "sa2va": "Sa2VA-4B", "grounded_sam2": "Grounded-SAM 2"}


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit(img: Image.Image) -> Image.Image:
    return img.resize((max(1, round(img.width * H / img.height)), H), Image.LANCZOS)


def overlay(image: np.ndarray, mask: np.ndarray, color) -> Image.Image:
    out = image.astype(np.float32).copy()
    out[mask] = 0.45 * out[mask] + 0.55 * np.array(color, np.float32)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def shard_rows(d: Path) -> pd.DataFrame:
    files = sorted((d / "shards").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def grid(rows: list[list[tuple[str, Image.Image]]], title: str) -> Image.Image:
    rows = [[(t, fit(im)) for t, im in r] for r in rows]
    gap, cap, top = 8, 24, 34
    w = max(sum(im.width for _, im in r) + gap * (len(r) + 1) for r in rows)
    canvas = Image.new("RGB", (w, top + len(rows) * (H + cap + gap)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((gap, 8), title[:180], fill="black", font=font(18))
    y = top
    for r in rows:
        x = gap
        for t, im in r:
            canvas.paste(im, (x, y))
            d.text((x, y + H + 3), t, fill="black", font=font(14))
            x += im.width + gap
        y += H + cap + gap
    return canvas


def baseline_masks(cfg: dict, method: str, picks: list) -> dict:
    import b_baselines as B
    if method not in B.PREDICTORS:
        import b_baselines2  # noqa: F401
    pred = B.PREDICTORS[method](cfg)
    out = {}
    for key, s in picks:
        image, _ = s.load()
        m, _, _, acc = pred(image, [s.referent])[0]
        out[key] = m.astype(bool) if (acc and m is not None) else np.zeros(image.shape[:2], bool)
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per_group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="qual_s2long")
    ap.add_argument("--baselines", default="qwen_box_sam,sa2va,grounded_sam2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ES, E = ES2.ES, ES2.E
    stage = ES.X2.S2T.s2_stage(cfg)
    mk = cfg["edit"]["mask"]
    out = ROOT / "results" / "qualitative" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    picks, meta = [], {}
    for ds in DATASETS:
        items = E.evaluated(E.samples(cfg, ds), False, 0)
        by_id = {s.sample_id: s for s in items}
        ids = [s.sample_id for s in items]
        mrun = resolve_run(cfg, f"e_{EDITOR}_{ds}_{stage}_merge", "latest", False)
        dirs = read_json(mrun / "metrics.json")["images_dir"]
        a = shard_rows(ES2.base_paths(cfg, EDITOR, ds, items, False)[0])
        b = shard_rows(ES.cache_dir(cfg, ds, ES.identity(cfg, ds, ids, False), False))
        df = a[["sample_id", "sam3_rle"]].merge(b[["sample_id", "ours2_rle", "ours2_iou", "sam3_iou"]], on="sample_id")
        df["margin"] = df.ours2_iou - df.sam3_iou
        groups = {"random": df.iloc[sorted(rng.choice(len(df), size=min(args.per_group, len(df)), replace=False))],
                  "ours_better": df.sort_values("margin", ascending=False).head(args.per_group),
                  "ours_worse": df.sort_values("margin").head(args.per_group)}
        for g, sub in groups.items():
            for r in sub.itertuples():
                key = (ds, g, r.sample_id)
                picks.append((key, by_id[r.sample_id]))
                meta[key] = {"row": r, "dirs": dirs}
    need = {"qwen_box_sam": ["qwen", "sam3"], "sa2va": ["sa2va"], "grounded_sam2": ["grounding_dino", "sam2"]}
    bnames = [m for m in args.baselines.split(",") if m and all(k in cfg["models"] for k in need.get(m, []))
              and m in cfg.get("baselines", {})]
    print("baselines:", bnames, flush=True)
    bmasks = {m: baseline_masks(cfg, m, picks) for m in bnames}
    index = []
    for key, s in picks:
        ds, g, sid = key
        r, dirs = meta[key]["row"], meta[key]["dirs"]
        image, gt = s.load()
        safe = E.safe_name(sid)
        with Image.open(Path(dirs["d14"]) / safe / "raw.png") as im:
            raw = np.asarray(im.convert("RGB"))

        def dec(js):
            v = json.loads(js)
            return M.decode(v).astype(bool) if v else np.zeros(gt.shape, bool)
        masks = {"gt": gt, "ours": dec(r.ours2_rle), "sam3text": dec(r.sam3_rle), **{m: bmasks[m][key] for m in bnames}}

        def edit(m):
            return image if not m.any() else EM.composite(image, raw, EM.blend_alpha(m, mk["dilate_px"], mk["feather_px"]))
        top = [("source", Image.fromarray(image))] + [
            (f"{NAMES[k]}" + ("" if k == "gt" else f"  IoU {iou(m, gt):.2f}"), overlay(image, m, COLORS[k]))
            for k, m in masks.items()]
        bottom = [("edit, no mask", Image.fromarray(raw))] + [
            (f"edit, {NAMES[k]} mask", Image.fromarray(edit(m))) for k, m in masks.items()]
        fn = f"{ds}_{g}_{safe[:40]}.jpg"
        grid([top, bottom], f"[{ds} | {g}]  {s.instruction}").save(out / fn, "JPEG", quality=85)
        index.append({"file": fn, "dataset": ds, "group": g, "sample_id": sid, "instruction": s.instruction,
                      "referent": s.referent, "iou": {k: round(iou(m, gt), 3) for k, m in masks.items() if k != "gt"}})
        print(fn, flush=True)
    write_json(out / "index.json", {
        "stage": stage, "editor": EDITOR, "groups": ["random", "ours_better", "ours_worse"], "baselines": bnames,
        "note": "every column blends the same Qwen-Image-Edit output with its mask (e_edit composite); "
                "ours_better / ours_worse are selected by IoU margin of ours vs SAM 3 text, random is a fixed-seed "
                "uniform sample", "panels": index})
    print(out)


if __name__ == "__main__":
    main()
