import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, node_cap  # noqa: E402
from treeground.metrics import area_bin, mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            start_run, write_json)

# Go / no-go diagnostic for the box-then-select method (dev label, PACO-LVIS val, nothing tuned, not a result).
# Per PACO val I1 object / part query, the best IoU with the ground truth reachable inside each candidate set:
#   auto     the capped D3/D4 SAM 3 nodes the current tree-attention models select from (s2_nodes.node_set)
#   box1     Qwen box -> SAM 3 single-mask output of the first box (the D13 qwen_box_sam prediction)
#   box_top  first box, the highest-scored of SAM 3's 3 masks
#   box3     every Qwen box of the query x SAM 3's 3 masks
#   obj3     the parent object's Qwen boxes x 3 masks (part queries; equals box3 for object queries)
#   boxes    box3 + obj3
#   union    auto + boxes
#   text1    SAM 3 text, top instance for the query phrase (the D13 sam3_text prediction)
#   text3    SAM 3 text, top-3 instances for the query phrase (s3_pool.TextInstances)
#   objtext3 top-3 instances for the parent object's phrase (equals text3 for object queries)
#   union_text  auto + text3 + objtext3 (the query-aware tree of the proposed method)
#   union_all   auto + boxes + text3 + objtext3
# box1 and text1 are real predictions; every other column is an oracle (best candidate) ceiling for a selector.
# Qwen: 'zero' = Qwen3-VL-4B unchanged (b_parity qwen_box_all), 'ft' = its D40 Group A LoRA (qwenbox_ft).

FILES = ("scripts/box_coverage.py", "scripts/b_parity.py", "scripts/b_baselines.py", "scripts/s2_nodes.py",
         "scripts/s3_pool.py", "treeground/proposals.py", "treeground/data/paco.py")
SETS = ("auto", "box1", "box_top", "box3", "obj3", "boxes", "union", "text1", "text3", "objtext3", "union_text",
        "union_all")
BOX_MODELS = {"zero": "qwen_box_all", "ft": "qwenbox_ft"}
THR = 0.5


def identity(cfg: dict, box: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    c = cfg["box_coverage"]
    ident = {"kind": "box_coverage", "box": box, "code": h.hexdigest(), "models": {k: cfg["models"][k]
             for k in ("qwen", "sam3")}, "proposals": cfg["proposals"], "cap": node_cap(cfg, False),
             "prompt": cfg["baselines"]["qwen_box_sam"], "parity": cfg["parity"][BOX_MODELS[box]],
             "sam3_text": cfg["baselines"]["sam3_text"], "instances_per_phrase": c["instances_per_phrase"],
             "max_images": c["max_images"], "num_shards": c["num_shards"]}
    if box == "ft":
        import qwenbox_train as QT
        ident["qwenbox_checkpoint"] = str(QT.checkpoint(cfg)[0])
    return ident


def out_dir(cfg: dict, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "box_coverage" / ident["box"] / canonical_hash(ident)[:16]


def val_images(cfg: dict) -> pd.DataFrame:
    _, _, _, wl = manifest_inputs(cfg, False)
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key").reset_index(drop=True)
    n = cfg["box_coverage"]["max_images"]
    if n and n < len(rows):
        rows = rows.iloc[np.sort(np.random.default_rng(0).permutation(len(rows))[:n])].reset_index(drop=True)
    return rows


def image_queries(img: paco.PacoImage) -> list[dict]:
    # object / part queries of the I1 set; a part's object phrase is its own object's I1 query phrase
    import b_baselines as B
    parts = {p.ann_id: p for p in img.parts}
    out = []
    for q in paco.i1_queries(img)[0]:
        if q.kind not in ("object", "part"):
            continue
        obj = parts[q.gt[0]].obj_name if q.kind == "part" else None
        obj_phrase = B.phrase(paco.OBJECT_TEMPLATE.format(object=paco.display(obj))) if obj else None
        out.append({"qid": q.qid, "kind": q.kind, "phrase": B.phrase(q.text), "obj_phrase": obj_phrase,
                    "gt": q.gt[0]})
    return out


def best(cands: np.ndarray, gt: np.ndarray) -> float:
    # cands (n, h, w) bool, gt (h, w) bool -> best IoU (0 for no candidates)
    if not len(cands):
        return 0.0
    inter = (cands & gt).reshape(len(cands), -1).sum(1)
    union = cands.reshape(len(cands), -1).sum(1) + gt.sum() - inter
    return float((inter / np.maximum(union, 1)).max())


def query_rows(key: str, queries: list[dict], gts: dict, auto: list, prompted: dict, text: dict) -> list[dict]:
    # prompted: phrase -> list over its boxes of (masks (3, h, w) bool, scores (3,), single (h, w) bool)
    # text: phrase -> list of (mask (h, w) bool, score), highest score first
    auto_iou = M.iou(auto, [gts[q["gt"]] for q in queries]).max(0) if auto and queries else np.zeros(len(queries))
    rows = []
    for k, q in enumerate(queries):
        gt_rle = gts[q["gt"]]
        gt = M.decode(gt_rle).astype(bool)
        own = prompted.get(q["phrase"], [])
        par = prompted.get(q["obj_phrase"], []) if q["obj_phrase"] else own
        t_own = text.get(q["phrase"], [])
        t_par = text.get(q["obj_phrase"], []) if q["obj_phrase"] else t_own
        three = lambda boxes: np.concatenate([b[0] for b in boxes]) if boxes else np.zeros((0,) + gt.shape, bool)
        stack = lambda inst: np.stack([m for m, _ in inst]) if inst else np.zeros((0,) + gt.shape, bool)
        r = {"qid": q["qid"], "kind": q["kind"], "image": key, "area_bin": area_bin(float(M.areas([gt_rle])[0])),
             "n_boxes": len(own), "n_obj_boxes": len(par), "auto": float(auto_iou[k]),
             "box1": best(own[0][2][None], gt) if own else 0.0,
             "box_top": best(own[0][0][int(np.argmax(own[0][1]))][None], gt) if own else 0.0,
             "box3": best(three(own), gt), "obj3": best(three(par), gt)}
        r["boxes"] = max(r["box3"], r["obj3"])
        r["union"] = max(r["auto"], r["boxes"])
        r["text1"] = best(t_own[0][0][None], gt) if t_own else 0.0
        r["text3"], r["objtext3"] = best(stack(t_own), gt), best(stack(t_par), gt)
        r["union_text"] = max(r["auto"], r["text3"], r["objtext3"])
        r["union_all"] = max(r["union"], r["union_text"])
        rows.append(r)
    return rows


class Prompter:
    def __init__(self, cfg: dict, box: str):
        import b_parity as BP
        import s2_nodes as S2N
        import s3_pool as S3P
        self.cfg, self.S2N = cfg, S2N
        self.pred = BP.PARITY[BOX_MODELS[box]](cfg)
        self.text = S3P.TextInstances(cfg)
        self.k = int(cfg["box_coverage"]["instances_per_phrase"])
        self.cap = node_cap(cfg, False)

    @torch.no_grad()
    def boxes(self, image: np.ndarray, phrases: list[str]) -> list[list[list[float]]]:
        from PIL import Image
        p = self.pred
        pil = Image.fromarray(image)
        p.qproc.tokenizer.padding_side = "left"
        raws = []
        for i in range(0, len(phrases), p.s["batch_size"]):
            chunk = phrases[i:i + p.s["batch_size"]]
            inp = p.qproc(text=[p.chat(t) for t in chunk], images=[pil] * len(chunk), padding=True,
                          return_tensors="pt").to(p.device)
            gen = p.qwen.generate(**inp, max_new_tokens=p.s["max_new_tokens"], do_sample=False)
            raws += p.qproc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        return [p.boxes(r) for r in raws]

    @torch.no_grad()
    def masks(self, image: np.ndarray, boxes: list[list[list[float]]]) -> list[list[tuple]]:
        import b_baselines as B
        p = self.pred
        h, w = image.shape[:2]
        flat = [(k, b) for k, bs in enumerate(boxes) for b in bs]
        out: list[list[tuple]] = [[] for _ in boxes]
        if not flat:
            return out
        pix = p.sproc(images=image, return_tensors="pt").pixel_values.to(p.device, p.dtype)
        emb = p.sam.get_image_embeddings(pix)
        t = p.sproc.target_size
        scale = np.array([t / w, t / h, t / w, t / h])
        for j in range(0, len(flat), p.s["sam_batch"]):
            part = flat[j:j + p.s["sam_batch"]]
            px = torch.tensor(np.stack([B.box_to_pixels(b, w, h) for _, b in part]) * scale, dtype=p.dtype,
                              device=p.device)[None]
            res = {}
            for multi in (True, False):
                r = p.sam(image_embeddings=emb, input_boxes=px, multimask_output=multi)
                logits = p.sproc.post_process_masks([r.pred_masks[0].float()], [[h, w]], binarize=False)[0]
                res[multi] = ((logits > p.s["mask_threshold"]).cpu().numpy(), r.iou_scores[0].float().cpu().numpy())
            for i, (k, _) in enumerate(part):
                out[k].append((res[True][0][i], res[True][1][i], res[False][0][i, 0]))
        return out

    def __call__(self, key: str, image: np.ndarray, queries: list[dict], gts: dict) -> list[dict]:
        phrases = sorted({q["phrase"] for q in queries} | {q["obj_phrase"] for q in queries if q["obj_phrase"]})
        prompted = dict(zip(phrases, self.masks(image, self.boxes(image, phrases))))
        text = dict(zip(phrases, self.text(image, phrases, self.k)))
        auto = self.S2N.node_set(self.pred.sam, self.pred.sproc, self.cfg, image, self.cap)
        return query_rows(key, queries, gts, auto, prompted, text)


def run_shard(cfg: dict, box: str, shard: int) -> None:
    n = cfg["box_coverage"]["num_shards"]
    if not 0 <= shard < n:
        raise ProtocolError(f"--shard must be in 0..{n - 1}")
    ident = identity(cfg, box)
    d = out_dir(cfg, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    images = val_images(cfg).iloc[shard::n]
    run = start_run(f"box_coverage_{box}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    annos = paco.load(cfg["data"]["paco_lvis"], "val", {int(k.split(":")[1]) for k in images.image_key})
    prompter = Prompter(cfg, box)
    rows, t0 = [], time.time()
    for i, r in enumerate(images.itertuples(), 1):
        img = annos[int(r.image_key.split(":")[1])]
        qs = image_queries(img)
        if not qs:
            continue
        image = load_rgb(assert_real_path(r.path))
        if image.shape[:2] != (img.h, img.w):
            raise ProtocolError(f"{r.image_key}: image {image.shape[:2]} differs from the annotation")
        rows += prompter(r.image_key, image, qs, paco.gt_rles(img))
        if i % 25 == 0:
            print(f"[{i}/{len(images)}] queries {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "queries": len(rows), "seconds": time.time() - t0})


def summarize(df: pd.DataFrame, bs: dict) -> dict:
    def block(g: pd.DataFrame) -> dict:
        groups = g.image.to_numpy()
        return {s: {"hit@0.5": mean_ci(g[s] >= THR, groups, bs["n"], bs["seed"]),
                    "mean_best_iou": mean_ci(g[s], groups, bs["n"], bs["seed"])} for s in SETS} | {
            "no_box": float((g.n_boxes == 0).mean()), "queries": int(len(g))}
    out = {}
    for kind in ("object", "part"):
        g = df[df.kind == kind]
        if len(g):
            out[kind] = block(g)
            out[kind]["by_area_bin"] = {b: block(x) for b, x in g.groupby("area_bin")}
    return out


def merge(cfg: dict, box: str) -> None:
    ident = identity(cfg, box)
    d = out_dir(cfg, ident)
    n = cfg["box_coverage"]["num_shards"]
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate query rows")
    run = start_run(f"box_coverage_{box}", cfg, False, {"cache": str(d)})
    df.to_csv(run / "coverage_queries.csv", index=False)
    res = summarize(df, cfg["v2"]["bootstrap"])
    write_json(run / "metrics.json", {"label": "dev", "diagnostic": True, "split": "PACO-LVIS val", "box": box,
                                      "iou_threshold": THR, "images": int(df.image.nunique()), **res})
    for kind, r in res.items():
        print(kind, r["queries"], "no box", round(r["no_box"], 3),
              {s: (round(r[s]["hit@0.5"]["mean"], 3), round(r[s]["mean_best_iou"]["mean"], 3)) for s in SETS})
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--box", choices=sorted(BOX_MODELS), default="zero")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.box)
    else:
        run_shard(cfg, args.box, args.shard)


if __name__ == "__main__":
    main()
