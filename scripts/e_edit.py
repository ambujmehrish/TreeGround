import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import edit_metrics as E  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data import edit_bench  # noqa: E402
from treeground.features import image_features  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (REPO, ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# Region-grounded editing (deviation D14). One shard: (1) masks for every sample — our selector's top node for the
# referent (I1 template, no rejection) and SAM 3 text prompting (D13 route) — cached per dataset and shared by the
# editors; (2) the editor's outputs per condition, saved as PNG; (3) per-sample metrics against the GT edit mask.
# --merge pools the shards and writes one metrics.json per (editor, dataset).

DATASETS = ("partedit", "gedit", "imgedit", "paco_edit")
EDITORS = ("qwen_image_edit", "flux_fill")
REFERENT_TEMPLATE = "Locate {referent}. <SEL>"  # = paco.PART_TEMPLATE for "the {part} of the {object}"
MASKS = ("ours", "gt", "sam3text")
THR = 0.5


@lru_cache(maxsize=None)
def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def samples(cfg: dict, dataset: str) -> list[edit_bench.EditSample]:
    d, s = cfg["data"], cfg["edit"]["datasets"][dataset]
    if dataset == "partedit":
        return edit_bench.partedit(assert_real_path(d["partedit"]))
    if dataset == "gedit":
        return edit_bench.gedit(assert_real_path(d["gedit_imgedit"]), s["groups"], s["max_edit_frac"])
    if dataset == "imgedit":
        return edit_bench.imgedit(assert_real_path(d["gedit_imgedit"]), s["edit_types"], s["max_edit_frac"])
    if dataset == "paco_edit":
        return edit_bench.paco_edit(assert_real_path(d["paco_lvis"]), assert_real_path(d["coco_images"]), s["n"],
                                    s["seed"], s["min_part_area"], s["per_image"], assert_real_path)
    raise ProtocolError(f"unknown edit dataset {dataset}")


def evaluated(items: list, smoke: bool, n_smoke: int) -> list:
    items = sorted(items, key=lambda s: s.sample_id)
    if len({s.sample_id for s in items}) != len(items):
        raise ProtocolError("duplicate sample ids")
    return items[:n_smoke] if smoke else items


def conditions(cfg: dict, editor: str) -> list[str]:
    return list(cfg["edit"]["editors"][editor]["conditions"])


def mask_source(cond: str) -> str | None:
    if cond == "free":
        return None
    src = cond.rsplit("_", 1)[0]
    if src not in MASKS:
        raise ProtocolError(f"unknown condition {cond}")
    return src


def sample_seed(seed: int, sid: str) -> int:
    return int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16)


def safe_name(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sid)


def fit_dims(w: int, h: int, area: int, multiple: int) -> tuple[int, int]:
    s = (area / (w * h)) ** 0.5
    return max(multiple, round(w * s / multiple) * multiple), max(multiple, round(h * s / multiple) * multiple)


def select_identity(cfg: dict, dataset: str, sel_run: Path, ids: list[str], smoke: bool) -> dict:
    p, m = cfg["proposals"], cfg["models"]
    h = hashlib.sha256()
    for rel in ("scripts/e_edit.py", "treeground/data/edit_bench.py", "treeground/data/paco.py"):
        h.update((REPO / rel).read_bytes())
    return {"kind": "edit_select", "dataset": dataset, "smoke": smoke, "settings": cfg["edit"]["datasets"][dataset],
            "samples": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "sam3": [m["sam3"]["repo"], m["sam3"]["revision"]], "qwen": [m["qwen"]["repo"], m["qwen"]["revision"]],
            "proposals": {k: p[k] for k in ("points_per_side", "masks_per_point", "stability_thresh",
                                            "stability_offset", "mask_threshold", "min_area", "node_dedup", "dtype")},
            "node_cap": load_script("x_transfer").node_cap(), "selector_run": sel_run.name,
            "selector_stage": cfg["inputs"].get("selector_stage", "v2"), "template": REFERENT_TEMPLATE,
            "sam3_text": cfg["baselines"]["sam3_text"], "code": h.hexdigest()}


def edit_identity(cfg: dict, editor: str, sel_hash: str) -> dict:
    return {"kind": "edit", "editor": editor, "select": sel_hash, "model": cfg["models"][editor],
            "settings": cfg["edit"]["editors"][editor], "mask": cfg["edit"]["mask"], "seed": cfg["edit"]["seed"],
            "clip": cfg["models"]["clip"], "metrics": cfg["edit"]["metrics"]}


def cache_dir(cfg: dict, smoke: bool) -> Path:
    return Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "edit"


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def select_ours(models, cfg: dict, image: np.ndarray, referent: str, cap: dict, X) -> tuple:
    sam, sproc, qwen, qproc, sel, sel_id, layer, device = models
    p = cfg["proposals"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                            p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    out = P.generate(sam, sproc, image, pcfg)
    kept = np.flatnonzero(P.dedup(out["rles"], out["areas"], out["stability"], out["src_index"], p["node_dedup"]))
    rles = [out["rles"][i] for i in kept]
    c_idx = X.cap_nodes(rles, out["boxes"][kept], out["stability"][kept], out["src_index"][kept], cap)
    rc, bc = [rles[i] for i in c_idx], out["boxes"][kept][c_idx]
    if not rc:
        return None, float("-inf"), [], len(rles)
    feats = image_features(qwen, qproc, image, rc, bc, [layer], ["hidden"])
    e = sel.nodes(feats[f"hidden_L{layer}"].to(device))
    pr = mllm.probe_states(qwen, qproc, Image.fromarray(image), [REFERENT_TEMPLATE.format(referent=referent)],
                           [layer], sel_id, cfg["edit"]["probe_batch"])[0]
    q = sel.queries(torch.from_numpy(np.ascontiguousarray(pr[:, 0])).to(device))
    sig = torch.sigmoid(sel.scores(q, e))[0].cpu().numpy()
    a = int(np.argmax(sig))
    return rc[a], float(sig[a]), rc, len(rles)


def run_select(cfg: dict, items: list, target: Path, sel_run: Path) -> None:
    X, B = load_script("x_transfer"), load_script("b_baselines")
    models, tau, _ = X.load_models(cfg, sel_run)
    sam3text = B.Sam3Text(cfg)
    cap = X.node_cap()
    rows = []
    for n, s in enumerate(items, 1):
        t0 = time.time()
        image, gt = s.load()
        g = M.encode(gt)[0]
        rle, sigma, nodes, n_nodes = select_ours(models, cfg, image, s.referent, cap, X)
        best = float(M.iou(nodes, [g]).max()) if nodes else 0.0
        mask, score, _, accepted = sam3text(image, [s.referent])[0]
        srle = M.encode(mask)[0] if mask is not None and mask.any() else None
        rows.append({"sample_id": s.sample_id, "ours_rle": json.dumps(rle), "ours_sigma": sigma,
                     "ours_iou": float(M.iou([rle], [g])[0, 0]) if rle is not None else 0.0,
                     "sam3_iou": float(M.iou([srle], [g])[0, 0]) if srle is not None else 0.0,
                     "ours_above_tau_rej": bool(sigma >= tau), "tau_rej": tau, "nodes": n_nodes, "nodes_capped": len(nodes),
                     "best_node_iou": best, "sam3_rle": json.dumps(srle), "sam3_score": score,
                     "sam3_above_threshold": bool(accepted), "gt_area": int(gt.sum()), "select_seconds": time.time() - t0})
        print(f"[select {n}/{len(items)}] {s.sample_id} nodes {n_nodes} sigma {sigma:.3f} {time.time() - t0:.1f}s",
              flush=True)
    del models, sam3text
    free_cuda()
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)


def load_pipe(cls, cfg: dict, editor: str):
    m, s = cfg["models"][editor], cfg["edit"]["editors"][editor]
    pipe = cls.from_pretrained(m["repo"], revision=m["revision"], torch_dtype=getattr(torch, s["dtype"]),
                               cache_dir=cfg.get("hf_home"))
    if s["offload"] == "none":
        pipe.to(cfg["device"])
    elif s["offload"] == "model":
        pipe.enable_model_cpu_offload(device=cfg["device"])
    elif s["offload"] == "sequential":
        pipe.enable_sequential_cpu_offload(device=cfg["device"])
    else:
        raise ProtocolError(f"unknown offload mode {s['offload']}")
    pipe.set_progress_bar_config(disable=True)
    return pipe, s


def back_to(img: Image.Image, w: int, h: int) -> np.ndarray:
    img = img.convert("RGB")
    return np.asarray(img if img.size == (w, h) else img.resize((w, h), Image.LANCZOS))


class QwenImageEdit:
    def __init__(self, cfg: dict):
        from diffusers import QwenImageEditPlusPipeline
        self.pipe, self.s = load_pipe(QwenImageEditPlusPipeline, cfg, "qwen_image_edit")

    @torch.no_grad()
    def __call__(self, image: np.ndarray, s: edit_bench.EditSample, mask, seed: int) -> np.ndarray:
        h, w = image.shape[:2]
        out = self.pipe(image=[Image.fromarray(image)], prompt=s.instruction, negative_prompt=self.s["negative_prompt"],
                        true_cfg_scale=self.s["true_cfg_scale"], guidance_scale=self.s["guidance_scale"],
                        num_inference_steps=self.s["num_inference_steps"], num_images_per_prompt=1,
                        generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return back_to(out, w, h)


class FluxFill:
    def __init__(self, cfg: dict):
        from diffusers import FluxFillPipeline
        self.pipe, self.s = load_pipe(FluxFillPipeline, cfg, "flux_fill")

    @torch.no_grad()
    def __call__(self, image: np.ndarray, s: edit_bench.EditSample, mask: np.ndarray, seed: int) -> np.ndarray:
        h, w = image.shape[:2]
        W, H = fit_dims(w, h, self.s["target_area"], 16)
        out = self.pipe(prompt=s.fill_prompt, image=Image.fromarray(image),
                        mask_image=Image.fromarray(mask.astype(np.uint8) * 255), height=H, width=W,
                        guidance_scale=self.s["guidance_scale"], num_inference_steps=self.s["num_inference_steps"],
                        max_sequence_length=self.s["max_sequence_length"],
                        generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return back_to(out, w, h)


EDITOR_CLASSES = {"qwen_image_edit": QwenImageEdit, "flux_fill": FluxFill}


def sample_masks(row, gt: np.ndarray) -> dict[str, np.ndarray]:
    h, w = gt.shape

    def dec(js):
        r = json.loads(js)
        return M.decode(r).astype(bool) if r is not None else np.zeros((h, w), bool)
    return {"ours": dec(row.ours_rle), "gt": gt, "sam3text": dec(row.sam3_rle)}


def run_edits(cfg: dict, editor: str, items: list, sel: pd.DataFrame, img_dir: Path) -> pd.DataFrame:
    model = EDITOR_CLASSES[editor](cfg)
    mk, conds = cfg["edit"]["mask"], conditions(cfg, editor)
    paste = cfg["edit"]["editors"][editor].get("paste_back", True)
    by_id = sel.set_index("sample_id")
    rows = []
    for n, s in enumerate(items, 1):
        d = img_dir / safe_name(s.sample_id)
        d.mkdir(parents=True, exist_ok=True)
        image, gt = s.load()
        masks = sample_masks(by_id.loc[s.sample_id], gt)
        seed = sample_seed(cfg["edit"]["seed"], s.sample_id)
        t0 = time.time()
        raw = model(image, s, None, seed) if editor == "qwen_image_edit" else None
        t_raw = time.time() - t0
        for c in conds:
            src = mask_source(c)
            m = masks[src] if src else None
            t1 = time.time()
            if m is not None and not m.any():
                out = image  # empty selection: nothing to edit
            elif editor == "qwen_image_edit":
                out = raw if m is None else E.composite(image, raw, E.blend_alpha(m, mk["dilate_px"], mk["feather_px"]))
            else:
                filled = model(image, s, E.dilate(m, mk["dilate_px"]), seed)
                out = E.composite(image, filled, E.blend_alpha(m, mk["dilate_px"], mk["feather_px"])) if paste else filled
            Image.fromarray(out).save(d / f"{c}.png")
            # instruction editors run once per sample; their time is shared by the conditions derived from it
            rows.append({"sample_id": s.sample_id, "condition": c, "empty_mask": bool(m is not None and not m.any()),
                         "edit_seconds": t_raw + time.time() - t1})
        if raw is not None:
            Image.fromarray(raw).save(d / "raw.png")
        print(f"[edit {n}/{len(items)}] {s.sample_id} {time.time() - t0:.1f}s", flush=True)
    del model
    free_cuda()
    return pd.DataFrame(rows)


class Clip:
    def __init__(self, cfg: dict):
        from transformers import CLIPModel, CLIPProcessor
        m = cfg["models"]["clip"]
        self.device = cfg["device"]
        self.model = CLIPModel.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"))
        self.model = self.model.to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"))

    @staticmethod
    def _emb(out) -> np.ndarray:
        return (out if isinstance(out, torch.Tensor) else out.pooler_output)[0].float().cpu().numpy()

    @torch.no_grad()
    def image(self, img: np.ndarray) -> np.ndarray:
        x = self.proc(images=Image.fromarray(img), return_tensors="pt").to(self.device)
        return self._emb(self.model.get_image_features(pixel_values=x.pixel_values))

    @torch.no_grad()
    def text(self, t: str) -> np.ndarray:
        x = self.proc(text=[t], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        return self._emb(self.model.get_text_features(input_ids=x.input_ids, attention_mask=x.attention_mask))


def lpips_model(cfg: dict):
    try:
        return E.Lpips(cfg["edit"]["metrics"]["lpips_net"], cfg["device"]), None
    except ImportError:
        return None, "lpips package not installed; LPIPS columns are NaN"


def score(cfg: dict, items: list, sel: pd.DataFrame, edits: pd.DataFrame, img_dir: Path) -> tuple[pd.DataFrame, str | None]:
    clip = Clip(cfg)
    lp, note = lpips_model(cfg)
    by_id = sel.set_index("sample_id")
    rows = []
    for s in items:
        image, gt = s.load()
        masks = sample_masks(by_id.loc[s.sample_id], gt)
        out_region = ~gt
        e_src, t_src, t_tgt = clip.image(image), clip.text(s.src_caption), clip.text(s.tgt_caption)
        for r in edits[edits.sample_id == s.sample_id].itertuples():
            with Image.open(img_dir / safe_name(s.sample_id) / f"{r.condition}.png") as im:
                out = np.asarray(im.convert("RGB"))
            src = mask_source(r.condition)
            lmap = lp(out, image) if lp is not None else None
            e_out = clip.image(out)
            rows.append({"sample_id": s.sample_id, "image": s.image_key, "subset": s.subset, "condition": r.condition,
                         "mask": src or "none", "empty_mask": r.empty_mask, "edit_seconds": r.edit_seconds,
                         "psnr_out": E.masked_psnr(out, image, out_region),
                         "ssim_out": E.masked_mean(E.ssim_map(out, image), out_region),
                         "lpips_out": E.masked_mean(lmap, out_region) if lmap is not None else float("nan"),
                         "l1_in": E.masked_mean(E.abs_change(out, image), gt),
                         "lpips_in": E.masked_mean(lmap, gt) if lmap is not None else float("nan"),
                         "clip_dir": E.directional(e_src, e_out, t_src, t_tgt),
                         "clip_t": E.cosine(e_out, t_tgt),
                         "mask_iou": E.mask_iou(masks[src], gt) if src else float("nan"),
                         "mask_area_frac": float(masks[src].mean()) if src else float("nan"),
                         "gt_area_frac": float(gt.mean())})
    del clip, lp
    free_cuda()
    return pd.DataFrame(rows), note


def run_shard(cfg: dict, editor: str, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D3", "D4", "D14"] + (["D11"] if cfg["inputs"].get("selector_stage") == "s1" else []))
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    X = load_script("x_transfer")
    sel_run = X.selector_run(cfg)
    items = evaluated(samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    s_ident = select_identity(cfg, dataset, sel_run, [s.sample_id for s in items], smoke)
    s_hash = canonical_hash(s_ident)
    e_ident = edit_identity(cfg, editor, s_hash)
    root = cache_dir(cfg, smoke)
    sdir = root / "select" / dataset / s_hash[:16]
    edir = root / editor / dataset / canonical_hash(e_ident)[:16]
    target = edir / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    for d, ident in ((sdir, s_ident), (edir, e_ident)):
        (d / "shards").mkdir(parents=True, exist_ok=True)
        if not (d / "identity.json").exists():
            write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    mine = items[shard::num_shards]
    run = start_run(f"e_{editor}_{dataset}", cfg, smoke, {"shard": shard, "num_shards": num_shards,
                                                          "cache": str(edir), "select_cache": str(sdir),
                                                          "selector_run": str(sel_run)})
    t0 = time.time()
    s_target = sdir / "shards" / f"{shard:05d}.parquet"
    if not s_target.exists():
        run_select(cfg, mine, s_target, sel_run)
    sel = pd.read_parquet(s_target)
    if set(sel.sample_id) != {s.sample_id for s in mine}:
        raise ProtocolError(f"selection cache {s_target} does not match shard {shard}")
    edits = run_edits(cfg, editor, mine, sel, edir / "images")
    rows, note = score(cfg, mine, sel, edits, edir / "images")
    rows = rows.merge(sel.drop(columns=["ours_rle", "sam3_rle"]), on="sample_id", how="left")
    tmp = target.with_suffix(".tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"samples": len(mine), "seconds": time.time() - t0, "lpips_note": note})


METRICS = ("psnr_out", "ssim_out", "lpips_out", "l1_in", "lpips_in", "clip_dir", "clip_t")


def condition_block(g: pd.DataFrame, bs: dict) -> dict:
    out = {"n": int(len(g)), "empty_mask_rate": float(g.empty_mask.mean())}
    for k in METRICS:
        v = g[k].to_numpy(np.float64)
        ok = ~np.isnan(v)
        out[k] = mean_ci(v[ok], g.image.to_numpy()[ok], bs["n"], bs["seed"]) if ok.any() else None
    if g["mask"].iloc[0] in ("ours", "sam3text"):
        out["mask_iou"] = mean_ci(g.mask_iou.to_numpy(), g.image.to_numpy(), bs["n"], bs["seed"])
        out["mask_success@0.5"] = mean_ci((g.mask_iou >= THR).to_numpy(float), g.image.to_numpy(), bs["n"], bs["seed"])
        out["mask_area_over_gt"] = float(np.median(g.mask_area_frac / g.gt_area_frac.clip(lower=1e-12)))
    return out


def summarize(df: pd.DataFrame, conds: list[str], bs: dict) -> tuple[dict, pd.DataFrame]:
    out, table = {}, []
    for c in conds:
        g = df[df.condition == c]
        out[c] = {"all": condition_block(g, bs),
                  "by_subset": {k: condition_block(h, bs) for k, h in g.groupby("subset")} if g.subset.nunique() > 1
                  else None}
        b = out[c]["all"]
        table.append({"condition": c, "n": b["n"], **{k: b[k]["mean"] if b[k] else None for k in METRICS},
                      "mask_iou": b.get("mask_iou", {}).get("mean"), "empty_mask_rate": b["empty_mask_rate"]})
    first = df[df.condition == conds[0]]
    img = first.image.to_numpy()

    def ci(v):
        return mean_ci(np.asarray(v, np.float64), img, bs["n"], bs["seed"])
    loc = {"ours_mask_iou": ci(first.ours_iou), "ours_success@0.5": ci(first.ours_iou >= THR),
           "ours_best_node_iou": ci(first.best_node_iou), "ours_best_node_success@0.5": ci(first.best_node_iou >= THR),
           "ours_above_tau_rej_rate": float(first.ours_above_tau_rej.mean()),
           "sam3text_mask_iou": ci(first.sam3_iou), "sam3text_success@0.5": ci(first.sam3_iou >= THR),
           "sam3text_above_threshold_rate": float(first.sam3_above_threshold.mean())}
    return {"conditions": out, "localization": loc}, pd.DataFrame(table)


def merge(cfg: dict, editor: str, dataset: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D14"])
    X = load_script("x_transfer")
    sel_run = X.selector_run(cfg)
    items = evaluated(samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    s_hash = canonical_hash(select_identity(cfg, dataset, sel_run, [s.sample_id for s in items], smoke))
    e_ident = edit_identity(cfg, editor, s_hash)
    edir = cache_dir(cfg, smoke) / editor / dataset / canonical_hash(e_ident)[:16]
    missing = [k for k in range(num_shards) if not (edir / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {editor}/{dataset} shards missing under {edir}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(edir / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    conds = conditions(cfg, editor)
    expected = {(s.sample_id, c) for s in items for c in conds}
    if set(zip(df.sample_id, df.condition)) != expected or df.duplicated(["sample_id", "condition"]).any():
        raise ProtocolError(f"{editor}/{dataset}: merged rows do not match the evaluation set")
    run = start_run(f"e_{editor}_{dataset}_merge", cfg, smoke, {"cache": str(edir), "selector_run": str(sel_run)})
    metrics, table = summarize(df, conds, cfg["edit"]["metrics"]["bootstrap"])
    lp = df.lpips_out.isna().all()
    metrics.update({"deviation": "D14", "label": "dev" if dataset == "paco_edit" else "edit benchmark",
                    "editor": editor, "dataset": dataset, "smoke": smoke, "samples": len(items),
                    "images": int(df.image.nunique()), "model": cfg["models"][editor], "clip": cfg["models"]["clip"],
                    "settings": cfg["edit"]["editors"][editor], "mask": cfg["edit"]["mask"],
                    "dataset_settings": cfg["edit"]["datasets"][dataset], "selector_run": sel_run.name,
                    "lpips": "not computed: lpips package missing in the shard jobs" if lp else
                             f"lpips {cfg['edit']['metrics']['lpips_net']}, spatial map averaged over the region",
                    "images_dir": str(edir / "images")})
    df.to_csv(run / "samples.csv", index=False)
    table.to_csv(run / "table.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(table.to_string(index=False))
    print(run)


def fetch_lpips(cfg: dict) -> None:
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    E.Lpips(cfg["edit"]["metrics"]["lpips_net"], "cpu")
    print(f"LPIPS {cfg['edit']['metrics']['lpips_net']} backbone cached under {cfg['torch_home']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", choices=EDITORS)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fetch-lpips", action="store_true", help="login node: cache the LPIPS backbone weights")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.fetch_lpips:
        fetch_lpips(cfg)
        return
    if not (args.editor and args.dataset):
        ap.error("--editor and --dataset are required")
    num_shards = 1 if args.smoke else cfg["edit"]["num_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.editor, args.dataset, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.editor, args.dataset, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
