import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (REPO, ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, resolve_run, start_run, write_json)

# Same-protocol baselines (deviation D13) on the V2 evaluation set: PACO-LVIS val I1 object and part queries
# (chains are scored from the same object and part predictions) plus gRefCOCO val no-target queries.
# Every model returns at most one mask per query; an empty mask is an abstention and scores IoU 0 when a target
# is present.

METHODS = ("sam3_text", "qwen_box_sam", "sa2va")
THR = 0.5
SINGLE_LEVEL = 0.5  # as V4 (D5): |GT part| >= 0.5 |GT object|
QWEN_GRID = 1000.0  # Qwen3-VL grounding boxes are relative, integers in [0, 1000]
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[([^\[\]]*)\]')
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def phrase(text: str) -> str:
    t = text.replace(mllm.SEL, "").strip()
    t = t[len("Locate "):] if t.startswith("Locate ") else t
    return t.rstrip(". ").strip()


def concept(p: str) -> str:
    return p[4:] if p.lower().startswith("the ") else p


def parse_box(text: str) -> list[float] | None:
    m = _BOX.search(text)
    if m is None:
        m = re.search(r"\[([^\[\]]*)\]", text)
    if m is None:
        return None
    vals = [float(v) for v in _NUM.findall(m.group(1))]
    if len(vals) != 4:
        return None
    x0, y0, x1, y1 = (min(max(v, 0.0), QWEN_GRID) for v in vals)
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def box_to_pixels(box: list[float], w: int, h: int) -> np.ndarray:
    return np.asarray(box, dtype=np.float64) / QWEN_GRID * np.array([w, h, w, h], dtype=np.float64)


def shard_keys(keys, shard: int, num_shards: int) -> list[str]:
    return sorted(set(keys))[shard::num_shards]


def overlap(pred: M.Rle | None, gt: M.Rle | None) -> dict:
    pa = int(M.areas([pred])[0]) if pred is not None else 0
    ga = int(M.areas([gt])[0]) if gt is not None else 0
    inter = int(M.intersections([pred], [gt])[0, 0]) if pred is not None and gt is not None else 0
    union = pa + ga - inter
    if gt is None:
        iou = float(pa == 0)  # no-target: correct abstention counts 1, as V2's end-to-end IoU
    else:
        iou = inter / union if union else 0.0
    return {"inter": inter, "union": union, "pred_area": pa, "gt_area": ga, "iou": iou}


def ratio_ci(num, den, groups, n: int, seed: int, alpha: float = 0.05) -> list[float]:
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    if not len(uniq):
        return [float("nan"), float("nan")]
    sn = np.bincount(inv, weights=np.asarray(num, np.float64), minlength=len(uniq))
    sd = np.bincount(inv, weights=np.asarray(den, np.float64), minlength=len(uniq))
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        stats[i] = (w * sn).sum() / max((w * sd).sum(), 1.0)
    return [float(v) for v in np.quantile(stats, [alpha / 2, 1 - alpha / 2])]


def block(g: pd.DataFrame, bs: dict) -> dict:
    out = {"n": int(len(g)),
           "gIoU": mean_ci(g.iou.to_numpy(), g.image.to_numpy(), bs["n"], bs["seed"]),
           "cIoU": {"value": float(g.inter.sum() / max(g.union.sum(), 1)),
                    "ci95": ratio_ci(g.inter, g.union, g.image, bs["n"], bs["seed"])},
           "success@0.5": mean_ci((g.iou >= THR).to_numpy(float), g.image.to_numpy(), bs["n"], bs["seed"]),
           "empty_prediction_rate": float((g.pred_area == 0).mean())}
    if "iou_any" in g and not np.allclose(g.iou_any, g.iou):
        out["gIoU_without_score_threshold"] = mean_ci(g.iou_any.to_numpy(), g.image.to_numpy(), bs["n"], bs["seed"])
    return out


def chain_table(pres: pd.DataFrame) -> pd.DataFrame:
    by_qid = pres.set_index("qid")
    rows = []
    for q in pres[pres.kind == "part"].itertuples():
        key, _, label = q.qid.split("|", 2)
        o = by_qid.loc[f"{key}|object|{label.split(':', 1)[0]}"]
        rows.append({"qid": f"{key}|chain|{label}", "image": q.image, "step1": o.iou >= THR, "step2": q.iou >= THR,
                     "chain": (o.iou >= THR) and (q.iou >= THR),
                     "single_level": q.gt_area / max(o.gt_area, 1) >= SINGLE_LEVEL})
    return pd.DataFrame(rows, columns=["qid", "image", "step1", "step2", "chain", "single_level"])


def chain_block(ch: pd.DataFrame, bs: dict) -> dict:
    def ci(col):
        return mean_ci(ch[col].to_numpy(float), ch.image.to_numpy(), bs["n"], bs["seed"])
    return {"chains": int(len(ch)), "final_part_success@0.5": ci("step2"), "step1_object_success@0.5": ci("step1"),
            "chain_success_end_to_end": ci("chain")}


def summarize(df: pd.DataFrame, bs: dict) -> tuple[dict, pd.DataFrame]:
    pres, absn = df[df.present], df[~df.present]
    ch = chain_table(pres)
    out = {"present": block(pres, bs), "by_kind": {k: block(g, bs) for k, g in pres.groupby("kind")},
           "no_target": {"n": int(len(absn)),
                         "abstention_rate": float((absn.pred_area == 0).mean()) if len(absn) else None,
                         "gIoU_present_and_no_target": mean_ci(df.iou.to_numpy(), df.image.to_numpy(),
                                                               bs["n"], bs["seed"])},
           "chains": {"all": chain_block(ch, bs), "two_level": chain_block(ch[~ch.single_level], bs),
                      "single_level": chain_block(ch[ch.single_level], bs)}}
    return out, ch


def build_queries(cfg: dict, wl: pd.DataFrame, mrun: Path) -> pd.DataFrame:
    val = wl[wl.role == "paco_val"]
    ids = {int(k.split(":")[1]) for k in val.image_key}
    images = paco.load(cfg["data"]["paco_lvis"], "val", ids)
    paths = dict(zip(wl.image_key, wl.path))
    rows = []
    for key in sorted(val.image_key):
        for q in paco.i1_queries(images[int(key.split(":")[1])])[0]:
            if q.kind in ("object", "part"):
                rows.append({"qid": q.qid, "kind": q.kind, "phrase": phrase(q.text), "image": key,
                             "path": paths[key], "gt": q.gt[0], "present": True})
    gref = pd.read_parquet(mrun / "grefcoco_queries.parquet")
    for g in gref[gref.split == "val"].itertuples():
        rows.append({"qid": g.qid, "kind": "absent", "phrase": g.text.strip(), "image": g.image_key,
                     "path": paths[g.image_key], "gt": -1, "present": False})
    df = pd.DataFrame(rows)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate query ids")
    return df


class Sam3Text:
    def __init__(self, cfg: dict):
        from transformers import Sam3Model, Sam3Processor
        m, self.s = cfg["models"]["sam3"], cfg["baselines"]["sam3_text"]
        self.device, self.dtype = cfg["device"], getattr(torch, cfg["proposals"]["dtype"])
        self.model = checked_load(Sam3Model, m["repo"], m["revision"], cfg.get("hf_home"), dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.proc = Sam3Processor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"))

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrases: list[str]) -> list[tuple]:
        h, w = image.shape[:2]
        pix = self.proc(images=Image.fromarray(image), return_tensors="pt").pixel_values
        ve = self.model.get_vision_features(pixel_values=pix.to(self.device, self.dtype))
        out = []
        for p in phrases:
            t = self.proc(text=concept(p), return_tensors="pt").to(self.device)
            r = self.model(vision_embeds=ve, input_ids=t.input_ids, attention_mask=t.attention_mask)
            s = r.pred_logits[0].float().sigmoid()
            if r.presence_logits is not None:
                s = s * r.presence_logits[0].float().sigmoid()
            i = int(s.argmax())
            # same order as Sam3ImageProcessorFast.post_process_instance_segmentation: sigmoid, resize, threshold
            m = F.interpolate(r.pred_masks[0, i][None, None].float().sigmoid(), size=(h, w), mode="bilinear",
                              align_corners=False)[0, 0] > self.s["mask_threshold"]
            score = float(s[i])
            out.append((m.cpu().numpy(), score, "", score >= self.s["score_threshold"]))
        return out


class QwenBoxSam:
    def __init__(self, cfg: dict):
        q, sm, self.s = cfg["models"]["qwen"], cfg["models"]["sam3"], cfg["baselines"]["qwen_box_sam"]
        self.device, self.dtype = cfg["device"], getattr(torch, cfg["proposals"]["dtype"])
        self.qwen, self.qproc = mllm.load_qwen(q["repo"], q["revision"], self.device, cfg["features"]["dtype"],
                                               cfg.get("hf_home"))
        self.sam, self.sproc = P.load_sam3(sm["repo"], sm["revision"], self.device, cfg["proposals"]["dtype"],
                                           cfg.get("hf_home"))

    def chat(self, p: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                                                  "text": self.s["prompt"].format(phrase=p)}]}]
        return self.qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrases: list[str]) -> list[tuple]:
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        tok = self.qproc.tokenizer
        tok.padding_side = "left"
        raws = []
        for i in range(0, len(phrases), self.s["batch_size"]):
            chunk = phrases[i:i + self.s["batch_size"]]
            inp = self.qproc(text=[self.chat(p) for p in chunk], images=[pil] * len(chunk), padding=True,
                             return_tensors="pt").to(self.device)
            gen = self.qwen.generate(**inp, max_new_tokens=self.s["max_new_tokens"], do_sample=False)
            raws += self.qproc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        boxes = [parse_box(r) for r in raws]
        valid = [i for i, b in enumerate(boxes) if b is not None]
        masks: dict[int, np.ndarray] = {}
        if valid:
            pix = self.sproc(images=image, return_tensors="pt").pixel_values.to(self.device, self.dtype)
            emb = self.sam.get_image_embeddings(pix)
            t = self.sproc.target_size
            for j in range(0, len(valid), self.s["sam_batch"]):
                idx = valid[j:j + self.s["sam_batch"]]
                px = np.stack([box_to_pixels(boxes[i], w, h) for i in idx]) * np.array([t / w, t / h, t / w, t / h])
                r = self.sam(image_embeddings=emb, input_boxes=torch.tensor(px, dtype=self.dtype,
                                                                             device=self.device)[None],
                             multimask_output=False)
                logits = self.sproc.post_process_masks([r.pred_masks[0].float()], [[h, w]], binarize=False)[0]
                for i, m in zip(idx, (logits[:, 0] > self.s["mask_threshold"]).cpu().numpy()):
                    masks[i] = m
        return [(masks.get(i), float("nan"), raws[i], i in masks) for i in range(len(phrases))]


class Sa2VA:
    # follows the model card's predict_forward usage at the pinned revision (remote code)
    def __init__(self, cfg: dict):
        from transformers import AutoModel, AutoProcessor
        m, self.s = cfg["models"]["sa2va"], cfg["baselines"]["sa2va"]
        self.model = checked_load(AutoModel, m["repo"], m["revision"], cfg.get("hf_home"),
                                  dtype=getattr(torch, self.s["dtype"]), trust_remote_code=True)
        self.model = self.model.to(cfg["device"]).eval()
        self.proc = AutoProcessor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"),
                                                  trust_remote_code=True)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrases: list[str]) -> list[tuple]:
        pil = Image.fromarray(image)
        out = []
        for p in phrases:
            r = self.model.predict_forward(image=pil, text=self.s["prompt"].format(phrase=p), past_text="",
                                           mask_prompts=None, processor=self.proc)
            ms = [np.asarray(x)[0] for x in r["prediction_masks"]]
            # the official RefCOCO evaluation takes the union of the masks of all [SEG] tokens
            m = np.logical_or.reduce(ms) if ms else None
            out.append((m, float("nan"), r["prediction"], m is not None))
        return out


def checked_load(cls, repo: str, revision: str, cache_dir: str | None, **kw):
    model, info = cls.from_pretrained(repo, revision=revision, cache_dir=cache_dir, output_loading_info=True, **kw)
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "mismatched_keys") and v}
    if bad:
        raise ProtocolError(f"{repo}@{revision[:8]} did not load cleanly: "
                            f"{ {k: list(v)[:5] for k, v in bad.items()} }")
    return model


PREDICTORS = {"sam3_text": Sam3Text, "qwen_box_sam": QwenBoxSam, "sa2va": Sa2VA}


def identity(cfg: dict, method: str, mhash: str, smoke: bool) -> dict:
    used = {"sam3_text": ["sam3"], "qwen_box_sam": ["qwen", "sam3"], "sa2va": ["sa2va"]}[method]
    return {"kind": "baseline", "method": method, "smoke": smoke, "manifest": mhash,
            "settings": cfg["baselines"][method], "models": {k: cfg["models"][k] for k in used},
            "dtype": cfg["proposals"]["dtype"], "templates": paco.TEMPLATES,
            "code": hashlib.sha256((REPO / "scripts" / "b_baselines.py").read_bytes()).hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if ident["smoke"] else "full") / "baselines" / ident["method"]
            / canonical_hash(ident)[:16])


def select_images(queries: pd.DataFrame, smoke: bool, n_smoke: int) -> list[str]:
    if not smoke:
        return sorted(set(queries.image))
    pres = sorted(set(queries[queries.present].image))[:n_smoke]
    absn = sorted(set(queries[~queries.present].image))[:max(1, n_smoke // 2)]
    return sorted(set(pres + absn))


def run_shard(cfg: dict, method: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D13"])
    mrun, _, mhash, wl = manifest_inputs(cfg, False)
    ident = identity(cfg, method, mhash, smoke)
    d = out_dir(cfg, ident)
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    queries = build_queries(cfg, wl, mrun)
    keys = shard_keys(select_images(queries, smoke, cfg["baselines"]["smoke_images"]), shard, num_shards)
    run = start_run(f"b_{method}", cfg, smoke, {"shard": shard, "num_shards": num_shards, "cache": str(d),
                                                 "manifest_run": str(mrun)})
    ids = {int(k.split(":")[1]) for k in keys if k.startswith("paco_lvis:")}
    annos = paco.load(cfg["data"]["paco_lvis"], "val", ids)
    predict = PREDICTORS[method](cfg)
    rows, t0 = [], time.time()
    for n, (key, g) in enumerate(queries[queries.image.isin(keys)].groupby("image", sort=True), 1):
        t1 = time.time()
        image = load_rgb(assert_real_path(g.path.iloc[0]))
        gts = {}
        if g.present.any():
            img = annos[int(key.split(":")[1])]
            if image.shape[:2] != (img.h, img.w):
                raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {(img.h, img.w)}")
            gts = paco.gt_rles(img)
        for q, (mask, score, raw, accepted) in zip(g.itertuples(), predict(image, g.phrase.tolist())):
            if mask is not None and mask.shape != image.shape[:2]:
                raise ProtocolError(f"{q.qid}: mask {mask.shape} differs from the image {image.shape[:2]}")
            rle = M.encode(mask)[0] if mask is not None and mask.any() else None
            gt = gts[q.gt] if q.present else None
            acc = overlap(rle if accepted else None, gt)
            rows.append({"qid": q.qid, "kind": q.kind, "image": key, "present": q.present, "phrase": q.phrase,
                         "score": score, "accepted": bool(accepted), "raw": raw, **acc,
                         "iou_any": overlap(rle, gt)["iou"], "pred_rle": json.dumps(rle if accepted else None)})
        print(f"[{n}/{len(keys)}] {key} queries {len(g)} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(keys), "queries": len(rows), "seconds": time.time() - t0})


def merge(cfg: dict, method: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D13"])
    mrun, _, mhash, wl = manifest_inputs(cfg, False)
    ident = identity(cfg, method, mhash, smoke)
    d = out_dir(cfg, ident)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {method} shards missing under {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    queries = build_queries(cfg, wl, mrun)
    expected = set(queries[queries.image.isin(select_images(queries, smoke, cfg["baselines"]["smoke_images"]))].qid)
    if set(df.qid) != expected or df.qid.duplicated().any():
        raise ProtocolError(f"{method}: merged queries do not match the evaluation set ({len(df)} vs {len(expected)})")
    run = start_run(f"b_{method}_merge", cfg, smoke, {"cache": str(d), "manifest_run": str(mrun)})
    metrics, ch = summarize(df, cfg["v2"]["bootstrap"])
    metrics.update({"method": method, "deviation": "D13", "label": "dev", "smoke": smoke,
                    "split": "PACO-LVIS val I1 object/part queries (+ gRefCOCO val no-target)",
                    "models": ident["models"], "settings": ident["settings"], "images": int(df.image.nunique()),
                    "v2_reference": v2_reference(cfg, df, smoke)})
    df.drop(columns=["pred_rle"]).to_csv(run / "queries.csv", index=False)
    ch.to_csv(run / "chains.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(json.dumps({"gIoU": metrics["present"]["gIoU"], "cIoU": metrics["present"]["cIoU"]}, indent=1))
    print(run)


def v2_reference(cfg: dict, df: pd.DataFrame, smoke: bool) -> dict | None:
    try:
        run = resolve_run(cfg, "v2", cfg.get("inputs", {}).get("v2", "latest"), False)
    except ProtocolError:
        return None
    m = json.loads((run / "metrics.json").read_text())
    v2q = pd.read_csv(run / "eval_queries.csv")
    same = set(v2q[v2q.present].qid) == set(df[df.present].qid)
    return {"v2_run": run.name, "same_present_queries": bool(same) if not smoke else None,
            "e2e_gIoU_present": m["iii_end_to_end_gIoU"]["present"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = 1 if args.smoke else cfg["baselines"]["num_shards"]
    if args.merge:
        merge(cfg, args.method, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.method, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
