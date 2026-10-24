import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import edit_metrics as EM  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.data import cocoedit_bench as CB  # noqa: E402
from treeground.run import (REPO, ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# Region-grounded editing under the CoCoEdit masked-benchmark protocol (deviation D25) on all 606 GEdit-Bench-EN and
# 737 ImgEdit-Bench samples. Caches (each with its own identity, so a new mask source only adds its own work):
# select/<source>/<dataset>: one mask per localized sample; raw/<editor>/<dataset>: the instruction-only Qwen edit
# of every sample, or the Fill of the add/style/tone/extract samples; edit/<editor>/<dataset>/<condition>: the
# condition's output PNG at source resolution. --merge writes the manifest the judge and the evaluation read.

EDITORS = ("qwen_image_edit", "flux_fill")
BUILTIN = ("gt", "bbox")


@lru_cache(maxsize=None)
def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edit_script():
    return load_script("e_edit")


def code_hash(*rels: str) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()


class SelectorSource:
    @staticmethod
    def stage_cfg(cfg: dict, spec: dict) -> dict:
        return dict(cfg, inputs={**cfg["inputs"], "selector_stage": spec["stage"]})

    @staticmethod
    def requires(spec: dict) -> list[str]:
        return ["D3", "D4", "D14"] + (["D11"] if spec["stage"] == "s1" else [])

    @classmethod
    def identity(cls, cfg: dict, spec: dict) -> dict:
        p, m, E = cfg["proposals"], cfg["models"], edit_script()
        X = E.load_script("x_transfer")
        return {"spec": spec, "selector_run": X.selector_run(cls.stage_cfg(cfg, spec)).name,
                "sam3": m["sam3"], "qwen": m["qwen"], "template": E.REFERENT_TEMPLATE, "node_cap": X.node_cap(),
                "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                                "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
                "code": code_hash("scripts/e_edit.py", "scripts/x_transfer.py")}

    def __init__(self, cfg: dict, spec: dict):
        E = edit_script()
        self.cfg, self.E = self.stage_cfg(cfg, spec), E
        self.X = E.load_script("x_transfer")
        self.models, self.tau, _ = self.X.load_models(self.cfg, self.X.selector_run(self.cfg))
        self.cap = self.X.node_cap()

    def __call__(self, image: np.ndarray, referent: str) -> tuple:
        rle, sigma, nodes, n = self.E.select_ours(self.models, self.cfg, image, referent, self.cap, self.X)
        mask = M.decode(rle).astype(bool) if rle is not None else None
        return mask, sigma, {"nodes": n, "nodes_capped": len(nodes), "above_tau_rej": bool(sigma >= self.tau),
                             "tau_rej": self.tau}


class BaselineSource:
    method = ""
    models: tuple = ()

    @staticmethod
    def requires(spec: dict) -> list[str]:
        return ["D13"]

    @classmethod
    def identity(cls, cfg: dict, spec: dict) -> dict:
        return {"spec": spec, "models": {k: cfg["models"][k] for k in cls.models},
                "settings": cfg["baselines"][cls.method], "proposals_dtype": cfg["proposals"]["dtype"],
                "code": code_hash("scripts/b_baselines.py")}

    def __init__(self, cfg: dict, spec: dict):
        self.pred = getattr(edit_script().load_script("b_baselines"), self.cls_name)(cfg)

    def __call__(self, image: np.ndarray, referent: str) -> tuple:
        mask, score, raw, accepted = self.pred(image, [referent])[0]
        return mask, score, {"raw": raw, "accepted": bool(accepted)}


class Sam3TextSource(BaselineSource):
    method, models, cls_name = "sam3_text", ("sam3",), "Sam3Text"


class QwenBoxSamSource(BaselineSource):
    method, models, cls_name = "qwen_box_sam", ("qwen", "sam3"), "QwenBoxSam"


SOURCE_KINDS = {"selector": SelectorSource, "sam3_text": Sam3TextSource, "qwen_box_sam": QwenBoxSamSource}


def source_class(spec: dict):
    if spec["kind"] == "script":
        return getattr(load_script(spec["script"]), spec["attr"])
    if spec["kind"] not in SOURCE_KINDS:
        raise ProtocolError(f"unknown mask source kind {spec['kind']}")
    return SOURCE_KINDS[spec["kind"]]


def mask_source(cond: str) -> str | None:
    return None if cond == "free" else cond.rsplit("_", 1)[0]


def conditions(cfg: dict, editor: str) -> list[str]:
    conds = list(cfg["cocoedit"]["editors"][editor]["conditions"])
    mode = "blend" if editor == "qwen_image_edit" else "fill"
    for c in conds:
        src = mask_source(c)
        if (c == "free" and editor != "qwen_image_edit") or (src and (not c.endswith(f"_{mode}") or
                                                               src not in (*BUILTIN, *cfg["cocoedit"]["sources"]))):
            raise ProtocolError(f"condition {c} is not valid for {editor}")
    return conds


def used_sources(cfg: dict, editor: str) -> list[str]:
    srcs = {mask_source(c) for c in conditions(cfg, editor)}
    return [s for s in cfg["cocoedit"]["sources"] if s in srcs]


def samples(cfg: dict, dataset: str) -> list:
    if dataset not in CB.DATASETS:
        raise ProtocolError(f"unknown dataset {dataset}")
    return CB.load(dataset, assert_real_path(cfg["data"]["gedit_imgedit"]))


def evaluated(items: list, smoke: bool, per_category: int) -> list:
    items = sorted(items, key=lambda s: s.sample_id)
    if len({s.sample_id for s in items}) != len(items):
        raise ProtocolError("duplicate sample ids")
    if not smoke:
        return items
    taken: dict[str, int] = {}
    out = []
    for s in items:
        if taken.get(s.category, 0) < per_category:
            taken[s.category] = taken.get(s.category, 0) + 1
            out.append(s)
    return out


def ids_hash(items: list) -> str:
    return hashlib.sha256("\n".join(s.sample_id for s in items).encode()).hexdigest()


def root_dir(cfg: dict, smoke: bool) -> Path:
    return Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "cocoedit"


def select_identity(cfg: dict, source: str, dataset: str, items: list, smoke: bool) -> dict:
    spec = cfg["cocoedit"]["sources"][source]
    return {"kind": "cocoedit_select", "source": source, "dataset": dataset, "smoke": smoke, "samples": ids_hash(items),
            "source_identity": source_class(spec).identity(cfg, spec),
            "data": code_hash("treeground/data/cocoedit_bench.py", "treeground/data/edit_bench.py")}


def raw_identity(cfg: dict, editor: str, dataset: str, items: list, smoke: bool) -> dict:
    e = cfg["cocoedit"]["editors"][editor]
    return {"kind": "cocoedit_raw", "editor": editor, "dataset": dataset, "smoke": smoke, "samples": ids_hash(items),
            "model": cfg["models"][editor], "settings": cfg["edit"]["editors"][editor], "seed": cfg["cocoedit"]["seed"],
            "free_type_region": e.get("free_type_region"), "mask": cfg["cocoedit"]["mask"],
            "code": code_hash("scripts/e_cocoedit.py", "treeground/data/cocoedit_bench.py",
                              "treeground/data/edit_bench.py", "treeground/edit_metrics.py")}


def cond_identity(cfg: dict, editor: str, dataset: str, cond: str, raw_hash: str, sel_hashes: dict) -> dict:
    src = mask_source(cond)
    return {"kind": "cocoedit_edit", "editor": editor, "dataset": dataset, "condition": cond, "raw": raw_hash,
            "select": sel_hashes.get(src, src), "mask": cfg["cocoedit"]["mask"],
            "paste_back": cfg["edit"]["editors"][editor].get("paste_back", True)}


def cache(base: Path, ident: dict) -> Path:
    d = base / canonical_hash(ident)[:16]
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    return d


def shard_file(d: Path, shard: int, num_shards: int) -> Path:
    return d / "shards" / f"{shard:05d}of{num_shards:05d}.parquet"


def save_parquet(rows: list[dict], target: Path) -> None:
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)


def free_cuda() -> None:
    edit_script().free_cuda()


def sample_seed(seed: int, sid: str) -> int:
    return edit_script().sample_seed(seed, sid)


def safe_name(sid: str) -> str:
    return edit_script().safe_name(sid)


def run_select(cfg: dict, source: str, items: list, target: Path) -> None:
    spec = cfg["cocoedit"]["sources"][source]
    pred = source_class(spec)(cfg, spec)
    rows = []
    for n, s in enumerate(items, 1):
        t0 = time.time()
        image, gt = s.load()
        mask, score, extra = pred(image, s.referent)
        rle = M.encode(mask)[0] if mask is not None and mask.any() else None
        rows.append({"sample_id": s.sample_id, "rle": json.dumps(rle), "score": float(score),
                     "iou": EM.mask_iou(mask, gt) if rle is not None else 0.0, "extra": json.dumps(extra, default=str),
                     "seconds": time.time() - t0})
        print(f"[select {source} {n}/{len(items)}] {s.sample_id} {time.time() - t0:.1f}s", flush=True)
    del pred
    free_cuda()
    save_parquet(rows, target)


def to_source(img: Image.Image, w: int, h: int) -> np.ndarray:
    # the released metric scripts bring edited images to the source size with PIL bilinear
    img = img.convert("RGB")
    return np.asarray(img if img.size == (w, h) else img.resize((w, h), Image.BILINEAR))


class Editor:
    def __init__(self, cfg: dict, editor: str):
        from diffusers import FluxFillPipeline, QwenImageEditPlusPipeline
        cls = QwenImageEditPlusPipeline if editor == "qwen_image_edit" else FluxFillPipeline
        self.editor = editor
        self.pipe, self.s = edit_script().load_pipe(cls, cfg, editor)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, s, mask: np.ndarray | None, seed: int) -> np.ndarray:
        h, w = image.shape[:2]
        g = torch.Generator("cpu").manual_seed(seed)
        if self.editor == "qwen_image_edit":
            out = self.pipe(image=[Image.fromarray(image)], prompt=s.instruction,
                            negative_prompt=self.s["negative_prompt"], true_cfg_scale=self.s["true_cfg_scale"],
                            guidance_scale=self.s["guidance_scale"], num_inference_steps=self.s["num_inference_steps"],
                            num_images_per_prompt=1, generator=g).images[0]
        else:
            W, H = edit_script().fit_dims(w, h, self.s["target_area"], 16)
            out = self.pipe(prompt=s.fill_prompt, image=Image.fromarray(image),
                            mask_image=Image.fromarray(mask.astype(np.uint8) * 255), height=H, width=W,
                            guidance_scale=self.s["guidance_scale"], num_inference_steps=self.s["num_inference_steps"],
                            max_sequence_length=self.s["max_sequence_length"], generator=g).images[0]
        return to_source(out, w, h)


def bbox_mask(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m, bool)
    if m.any():
        ys, xs = np.nonzero(m)
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def free_type_region(cfg: dict, gt: np.ndarray) -> np.ndarray:
    # non-localized samples (add/style/tone/extract) are shared by every FLUX condition, so they must not see the GT:
    # 'whole' fills the whole image (default); 'gt' (oracle region for all conditions) is kept only for comparison
    mode = cfg["cocoedit"]["editors"]["flux_fill"].get("free_type_region", "whole")
    if mode == "whole":
        return np.ones_like(gt, bool)
    if mode != "gt":
        raise ProtocolError("free_type_region must be 'whole' or 'gt'")
    return gt if gt.any() else np.ones_like(gt, bool)


def needs_raw(editor: str, s) -> bool:
    return editor == "qwen_image_edit" or not s.localized


def save_png(arr: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(arr).save(tmp, format="PNG")
    tmp.replace(path)


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def run_raw(cfg: dict, editor: str, items: list, model: Editor, d: Path) -> list[dict]:
    rows = []
    for n, s in enumerate(items, 1):
        if not needs_raw(editor, s):
            continue
        t0, mk = time.time(), cfg["cocoedit"]["mask"]
        image, gt = s.load()
        region = None if editor == "qwen_image_edit" else free_type_region(cfg, gt)
        mask = None if region is None else EM.dilate(region, mk["dilate_px"])
        out = model(image, s, mask, sample_seed(cfg["cocoedit"]["seed"], s.sample_id))
        if region is not None and cfg["edit"]["editors"][editor].get("paste_back", True):
            out = EM.composite(image, out, EM.blend_alpha(region, mk["dilate_px"], mk["feather_px"]))
        save_png(out, d / "images" / f"{safe_name(s.sample_id)}.png")
        rows.append({"sample_id": s.sample_id, "seconds": time.time() - t0})
        print(f"[raw {editor} {n}/{len(items)}] {s.sample_id} {time.time() - t0:.1f}s", flush=True)
    return rows


def sample_masks(s, gt: np.ndarray, sel: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    out = {"gt": gt, "bbox": bbox_mask(gt)}
    for src, df in sel.items():
        r = json.loads(df.at[s.sample_id, "rle"])
        out[src] = M.decode(r).astype(bool) if r is not None else np.zeros_like(gt, bool)
    return out


def run_condition(cfg: dict, editor: str, cond: str, items: list, sel: dict, raw_dir: Path, model, d: Path
                  ) -> list[dict]:
    mk, src = cfg["cocoedit"]["mask"], mask_source(cond)
    paste = cfg["edit"]["editors"][editor].get("paste_back", True)
    rows = []
    for n, s in enumerate(items, 1):
        t0 = time.time()
        image, gt = s.load()
        m = sample_masks(s, gt, sel)[src] if src and s.localized else None
        raw_png = raw_dir / "images" / f"{safe_name(s.sample_id)}.png"
        if not s.localized or src is None:
            out, mode = load_png(raw_png), "raw"
        elif not m.any():
            out, mode = image, "empty_mask"
        elif editor == "qwen_image_edit":
            alpha = EM.blend_alpha(m, mk["dilate_px"], mk["feather_px"])
            out, mode = EM.composite(image, load_png(raw_png), alpha), "blend"
        else:
            out = model(image, s, EM.dilate(m, mk["dilate_px"]), sample_seed(cfg["cocoedit"]["seed"], s.sample_id))
            if paste:
                out = EM.composite(image, out, EM.blend_alpha(m, mk["dilate_px"], mk["feather_px"]))
            mode = "fill"
        if out.shape != image.shape:
            raise ProtocolError(f"{s.sample_id}/{cond}: output {out.shape} differs from the source {image.shape}")
        save_png(out, d / "images" / f"{safe_name(s.sample_id)}.png")
        rows.append({"sample_id": s.sample_id, "condition": cond, "mode": mode,
                     "mask_area_frac": float(m.mean()) if m is not None else float("nan"),
                     "mask_iou": EM.mask_iou(m, gt) if m is not None else float("nan"),
                     "gt_area_frac": float(gt.mean()), "seconds": time.time() - t0})
        if mode == "fill":
            print(f"[{editor} {cond} {n}/{len(items)}] {s.sample_id} {time.time() - t0:.1f}s", flush=True)
    return rows


def plan(cfg: dict, editor: str, dataset: str, smoke: bool) -> dict:
    items = evaluated(samples(cfg, dataset), smoke, cfg["cocoedit"]["smoke_per_category"])
    root = root_dir(cfg, smoke)
    sel = {}
    for src in used_sources(cfg, editor):
        ident = select_identity(cfg, src, dataset, items, smoke)
        sel[src] = (cache(root / "select" / src / dataset, ident), canonical_hash(ident))
    r_ident = raw_identity(cfg, editor, dataset, items, smoke)
    raw = cache(root / "raw" / editor / dataset, r_ident)
    (raw / "images").mkdir(exist_ok=True)
    conds = {}
    for c in conditions(cfg, editor):
        ident = cond_identity(cfg, editor, dataset, c, canonical_hash(r_ident), {k: v[1] for k, v in sel.items()})
        conds[c] = cache(root / "edit" / editor / dataset / c, ident)
        (conds[c] / "images").mkdir(exist_ok=True)
    return {"items": items, "select": {k: v[0] for k, v in sel.items()}, "raw": raw, "conditions": conds}


def required(cfg: dict, editor: str) -> list[str]:
    srcs = cfg["cocoedit"]["sources"]
    return sorted({"D14", "D25", *[d for s in used_sources(cfg, editor) for d in source_class(srcs[s]).requires(srcs[s])]})


def run_shard(cfg: dict, editor: str, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, required(cfg, editor))
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    p = plan(cfg, editor, dataset, smoke)
    mine = p["items"][shard::num_shards]
    local = [s for s in mine if s.localized]
    pending = [c for c, d in p["conditions"].items() if not shard_file(d, shard, num_shards).exists()]
    if not pending:
        print(f"shard {shard} already done")
        return
    run = start_run(f"cocoedit_{editor}_{dataset}", cfg, smoke, {"shard": shard, "num_shards": num_shards,
                    "select": {k: str(v) for k, v in p["select"].items()}, "raw": str(p["raw"]),
                    "conditions": {c: str(d) for c, d in p["conditions"].items()}})
    t0 = time.time()
    sel = {}
    for src, d in p["select"].items():
        f = shard_file(d, shard, num_shards)
        if not f.exists():
            run_select(cfg, src, local, f)
        sel[src] = pd.read_parquet(f).set_index("sample_id")
        if set(sel[src].index) != {s.sample_id for s in local}:
            raise ProtocolError(f"selection cache {f} does not match shard {shard}")
    model = None
    raw_target = shard_file(p["raw"], shard, num_shards)
    if not raw_target.exists():
        model = Editor(cfg, editor)
        save_parquet(run_raw(cfg, editor, mine, model, p["raw"]), raw_target)
    for c in pending:
        if editor == "flux_fill" and model is None:
            model = Editor(cfg, editor)
        save_parquet(run_condition(cfg, editor, c, mine, sel, p["raw"], model, p["conditions"][c]),
                     shard_file(p["conditions"][c], shard, num_shards))
    del model
    free_cuda()
    write_json(run / "metrics.json", {"samples": len(mine), "localized": len(local), "conditions": pending,
                                      "seconds": time.time() - t0})


def run_select_only(cfg: dict, editor: str, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, required(cfg, editor))
    p = plan(cfg, editor, dataset, smoke)
    local = [s for s in p["items"][shard::num_shards] if s.localized]
    for src, d in p["select"].items():
        f = shard_file(d, shard, num_shards)
        if not f.exists():
            run_select(cfg, src, local, f)


def read_shards(d: Path, num_shards: int, what: str) -> pd.DataFrame:
    missing = [k for k in range(num_shards) if not shard_file(d, k, num_shards).exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {what} shards missing under {d}, e.g. {missing[:5]}")
    return pd.concat([pd.read_parquet(shard_file(d, k, num_shards)) for k in range(num_shards)], ignore_index=True)


def merge(cfg: dict, editor: str, dataset: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, required(cfg, editor))
    p = plan(cfg, editor, dataset, smoke)
    items = p["items"]
    meta = pd.DataFrame([{"sample_id": s.sample_id, "key": s.key, "image": s.image_key, "category": s.category,
                          "scored": s.scored, "mllm": s.mllm, "invert": s.invert, "localized": s.localized}
                         for s in items])
    df = pd.concat([read_shards(d, num_shards, f"{editor}/{dataset}/{c}") for c, d in p["conditions"].items()],
                   ignore_index=True)
    conds = list(p["conditions"])
    if set(zip(df.sample_id, df.condition)) != {(s.sample_id, c) for s in items for c in conds} or \
            df.duplicated(["sample_id", "condition"]).any():
        raise ProtocolError(f"{editor}/{dataset}: merged rows do not match the evaluation set")
    df = df.merge(meta, on="sample_id", how="left", validate="many_to_one")
    loc = {}
    for src, d in p["select"].items():
        s = read_shards(d, num_shards, f"select {src}")
        loc[src] = {"n": int(len(s)), "mask_iou": float(s.iou.mean()), "success@0.5": float((s.iou >= 0.5).mean()),
                    "empty_rate": float((s.rle == "null").mean())}
    run = start_run(f"cocoedit_{editor}_{dataset}_merge", cfg, smoke, {})
    df.to_csv(run / "samples.csv", index=False)
    modes = df.groupby(["condition", "mode"]).size().unstack(fill_value=0)
    write_json(run / "metrics.json", {
        "deviation": "D25", "editor": editor, "dataset": dataset, "smoke": smoke, "samples": len(items),
        "scored": int(meta.scored.sum()), "mllm": int(meta.mllm.sum()), "model": cfg["models"][editor],
        "settings": cfg["edit"]["editors"][editor], "cocoedit": cfg["cocoedit"], "localization": loc,
        "modes": {c: {k: int(v) for k, v in r.items()} for c, r in modes.iterrows()},
        "images_dirs": {c: str(d / "images") for c, d in p["conditions"].items()}, "raw_dir": str(p["raw"]),
        "select_dirs": {k: str(v) for k, v in p["select"].items()}})
    print(json.dumps(loc, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", choices=EDITORS, required=True)
    ap.add_argument("--dataset", choices=CB.DATASETS, required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--select-only", action="store_true", help="only the mask sources of this editor's conditions")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--then-merge", action="store_true", help="merge after the shard (single-shard smoke runs)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = 1 if args.smoke else cfg["cocoedit"]["num_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.editor, args.dataset, num_shards, args.smoke)
        return
    if not 0 <= args.shard < num_shards:
        raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
    if args.select_only:
        run_select_only(cfg, args.editor, args.dataset, args.shard, num_shards, args.smoke)
        return
    run_shard(cfg, args.editor, args.dataset, args.shard, num_shards, args.smoke)
    if args.then_merge:
        merge(cfg, args.editor, args.dataset, num_shards, args.smoke)


if __name__ == "__main__":
    main()
