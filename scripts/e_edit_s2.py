import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e_edit as E  # noqa: E402
import e_select as ES  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, ensure_deviation_log, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# D23 mask sources in the D14 editors: 'ours2' (S2 flat pick) and 'ours2tree' (S2 tree-constrained pick) from the
# e_select cache, edited like D14's masks. The D14 conditions of the same config are not rerun: their cache (same
# e_edit identity) supplies the scored rows, the selection columns and, for Qwen-Image-Edit, the instruction edit
# raw.png that the new masks are blended with (one Qwen call per sample serves every blend condition). FLUX Fill
# runs once per new mask. --merge pools D14 and D23 rows into one table per (editor, dataset).

SOURCES = ("ours2", "ours2tree")
_sample_masks, _condition_block = E.sample_masks, E.condition_block


def sample_masks(row, gt: np.ndarray) -> dict[str, np.ndarray]:
    out = _sample_masks(row, gt)
    for k in SOURCES:
        r = json.loads(getattr(row, f"{k}_rle"))
        out[k] = M.decode(r).astype(bool) if r is not None else np.zeros(gt.shape, bool)
    return out


def condition_block(g: pd.DataFrame, bs: dict) -> dict:
    out = _condition_block(g, bs)
    if g["mask"].iloc[0] in SOURCES:
        img = g.image.to_numpy()
        out["mask_iou"] = mean_ci(g.mask_iou.to_numpy(), img, bs["n"], bs["seed"])
        out["mask_success@0.5"] = mean_ci((g.mask_iou >= E.THR).to_numpy(float), img, bs["n"], bs["seed"])
        out["mask_area_over_gt"] = float(np.median(g.mask_area_frac / g.gt_area_frac.clip(lower=1e-12)))
    return out


# e_edit resolves mask sources and per-condition blocks through these module globals
E.MASKS = tuple(dict.fromkeys(E.MASKS + SOURCES))
E.sample_masks, E.condition_block = sample_masks, condition_block


def new_conditions(cfg: dict, editor: str) -> list[str]:
    conds = list(cfg["edit_s2"]["conditions"][editor])
    if not conds or {E.mask_source(c) for c in conds} - set(SOURCES):
        raise ProtocolError(f"edit_s2 conditions of {editor} must use {SOURCES}: {conds}")
    return conds


class RawFromBase:
    # the D14 cache's instruction edit of the same sample (same model, seed and settings)
    def __init__(self, base_images: Path):
        self.base_images = base_images

    def __call__(self, image: np.ndarray, s, mask, seed: int) -> np.ndarray:
        f = self.base_images / E.safe_name(s.sample_id) / "raw.png"
        if not f.exists():
            raise ProtocolError(f"missing D14 instruction edit {f}")
        with Image.open(f) as im:
            raw = np.asarray(im.convert("RGB"))
        if raw.shape != image.shape:
            raise ProtocolError(f"{f}: shape {raw.shape} differs from the source image {image.shape}")
        return raw


def base_paths(cfg: dict, editor: str, dataset: str, items: list, smoke: bool) -> tuple[Path, Path, str]:
    sel_run = E.load_script("x_transfer").selector_run(cfg)
    s_hash = canonical_hash(E.select_identity(cfg, dataset, sel_run, [s.sample_id for s in items], smoke))
    e_hash = canonical_hash(E.edit_identity(cfg, editor, s_hash))
    root = E.cache_dir(cfg, smoke)
    return root / "select" / dataset / s_hash[:16], root / editor / dataset / e_hash[:16], e_hash


def identity(cfg: dict, editor: str, dataset: str, items: list, smoke: bool) -> dict:
    _, _, e_hash = base_paths(cfg, editor, dataset, items, smoke)
    sel = ES.identity(cfg, dataset, [s.sample_id for s in items], smoke)
    return {"kind": "edit_s2", "editor": editor, "base_edit": e_hash, "select_s2": canonical_hash(sel),
            "conditions": new_conditions(cfg, editor),
            "code": hashlib.sha256((ROOT / "scripts" / "e_edit_s2.py").read_bytes()).hexdigest()}


def out_dir(cfg: dict, editor: str, dataset: str, ident: dict, smoke: bool) -> Path:
    return E.cache_dir(cfg, smoke) / f"{editor}_s2" / dataset / canonical_hash(ident)[:16]


def required(path: Path, what: str) -> Path:
    if not path.exists():
        raise ProtocolError(f"missing {what}: {path}")
    return path


def approvals(cfg: dict) -> list[str]:
    return ["D3", "D4", "D14", "D23"] + (["D11"] if cfg["inputs"].get("selector_stage") == "s1" else [])


def run_shard(cfg: dict, editor: str, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, approvals(cfg))
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    items = E.evaluated(E.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    ident = identity(cfg, editor, dataset, items, smoke)
    d = out_dir(cfg, editor, dataset, ident, smoke)
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    sdir, bdir, _ = base_paths(cfg, editor, dataset, items, smoke)
    name = f"{shard:05d}.parquet"
    required(bdir / "shards" / name, "D14 edit shard (run scripts/e_edit.py with this config first)")
    base_sel = pd.read_parquet(required(sdir / "shards" / name, "D14 selection shard"))
    sel2 = pd.read_parquet(required(ES.shard_path(cfg, dataset, items, shard, smoke), "D23 selection shard "
                                    "(run scripts/e_select.py first)"))
    mine = items[shard::num_shards]
    ids = {s.sample_id for s in mine}
    if set(base_sel.sample_id) != ids or set(sel2.sample_id) != ids:
        raise ProtocolError(f"selection caches do not match shard {shard}")
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    stage = ES.X2.S2T.s2_stage(cfg)
    run = start_run(f"e_{editor}_{dataset}_{stage}", cfg, smoke, {"shard": shard, "num_shards": num_shards,
                                                                 "cache": str(d), "base_cache": str(bdir)})
    t0 = time.time()
    sel = base_sel.merge(sel2[["sample_id"] + [f"{k}_rle" for k in SOURCES]], on="sample_id")
    ecfg = {**cfg, "edit": {**cfg["edit"], "editors": {**cfg["edit"]["editors"], editor: {
        **cfg["edit"]["editors"][editor], "conditions": new_conditions(cfg, editor)}}}}
    classes = dict(E.EDITOR_CLASSES)
    if editor == "qwen_image_edit":
        E.EDITOR_CLASSES[editor] = lambda _: RawFromBase(bdir / "images")
    try:
        edits = E.run_edits(ecfg, editor, mine, sel, d / "images")
    finally:
        E.EDITOR_CLASSES.update(classes)
    rows, note = E.score(cfg, mine, sel, edits, d / "images")
    rows = rows.merge(base_sel.drop(columns=["ours_rle", "sam3_rle"]), on="sample_id", how="left")
    tmp = target.with_suffix(".tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"samples": len(mine), "seconds": time.time() - t0, "lpips_note": note})


def shards(d: Path, n: int, what: str) -> pd.DataFrame:
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {what} shards missing under {d}, e.g. {missing[:5]}")
    return pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)


def merge(cfg: dict, editor: str, dataset: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D14", "D23"])
    items = E.evaluated(E.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    ident = identity(cfg, editor, dataset, items, smoke)
    _, bdir, _ = base_paths(cfg, editor, dataset, items, smoke)
    d = out_dir(cfg, editor, dataset, ident, smoke)
    sel_ident = ES.identity(cfg, dataset, [s.sample_id for s in items], smoke)
    sel2 = shards(ES.cache_dir(cfg, dataset, sel_ident, smoke), num_shards, "D23 selection")
    df = pd.concat([shards(bdir, num_shards, "D14 edit"), shards(d, num_shards, "D23 edit")], ignore_index=True)
    conds = E.conditions(cfg, editor) + new_conditions(cfg, editor)
    expected = {(s.sample_id, c) for s in items for c in conds}
    if set(zip(df.sample_id, df.condition)) != expected or df.duplicated(["sample_id", "condition"]).any():
        raise ProtocolError(f"{editor}/{dataset}: merged rows do not match the evaluation set")
    stage = sel_ident["s2_stage"]
    run = start_run(f"e_{editor}_{dataset}_{stage}_merge", cfg, smoke, {"cache": str(d), "base_cache": str(bdir)})
    bs = cfg["edit"]["metrics"]["bootstrap"]
    metrics, table = E.summarize(df, conds, bs)
    metrics.update({"deviation": ["D14", "D23"], "label": "dev" if dataset == "paco_edit" else "edit benchmark",
                    "editor": editor, "dataset": dataset, "smoke": smoke, "samples": len(items),
                    "images": int(df.image.nunique()), "model": cfg["models"][editor], "clip": cfg["models"]["clip"],
                    "settings": cfg["edit"]["editors"][editor], "mask": cfg["edit"]["mask"],
                    "dataset_settings": cfg["edit"]["datasets"][dataset],
                    "selector_stage": cfg["inputs"].get("selector_stage", "v2"), "s2_stage": stage,
                    "s2_run": sel_ident["s2_run"], "s2_eval_run": sel_ident["s2_eval_run"],
                    "localization_s2": ES.localization(sel2, bs),
                    "lpips": "not computed" if df.lpips_out.isna().all() else
                             f"lpips {cfg['edit']['metrics']['lpips_net']}, spatial map averaged over the region",
                    "images_dir": {"d14": str(bdir / "images"), "d23": str(d / "images")}})
    df.to_csv(run / "samples.csv", index=False)
    table.to_csv(run / "table.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(table.to_string(index=False))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", required=True, choices=E.EDITORS)
    ap.add_argument("--dataset", required=True, choices=E.DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_deviation_log(cfg)
    num_shards = 1 if args.smoke else cfg["edit"]["num_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.editor, args.dataset, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.editor, args.dataset, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
