import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402

# ReasonSeg VAL diagnosis (development only: nothing here is reported as a result, nothing touches a test split).
# Where does ours lose on reasoning questions? Per val query and setting: the IoU of our one mask (scored at the
# image's resolution, as reason_eval scores it) and the best IoU any candidate of the pool reaches (coverage), so a
# miss is either the pool lacking the target or the selector passing it over.
#   cap1536   ours as evaluated on test (D50): long side capped at 1536 px, SAM 3 phrase = the question (s3_pool rule)
#   full      the same at the image's own resolution (does the cap cost us?)
#   rewrite   cap1536, but the pool's SAM 3 phrase is the target named by the base Qwen3-VL-4B (the backbone of ours
#             and of Box + SAM 3) from the question; our model still reads the original question
# Shards over the val queries; --merge writes the per-setting summary (by target size) to a 'val' metrics run.

SETTINGS = ("cap1536", "full", "rewrite")
REWRITE_PROMPT = ("Read the request about this image and name the one object or object part it refers to. "
                  "Answer with a short noun phrase only, such as 'the red cup' or 'the handle of the door'.\n"
                  "Request: {text}")
SIZE_BINS = (0.0, 0.05, 0.2, 0.5, 1.0001)


def val_records(cfg: dict) -> list[tuple]:
    import reason_eval as RE
    return RE.reasonseg_records({**cfg, "data": {**cfg["data"], "reasonseg": cfg["data"]["reasonseg_val"]}})


def noun_phrase(raw: str) -> str:
    # the model's first line, without quotes, a trailing period or a leading 'Answer:'
    s = raw.strip().splitlines()[0] if raw.strip() else ""
    s = s.split(":", 1)[1] if s.lower().startswith("answer:") else s
    return s.strip().strip("\"'`").rstrip(".").strip()


def downsize(image: np.ndarray, side: int | None) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    f = 1.0 if side is None else min(1.0, side / max(h, w))
    if f >= 1:
        return image, 1.0
    return np.asarray(Image.fromarray(image).resize((round(w * f), round(h * f)), Image.BILINEAR)), f


def resize_mask(m: np.ndarray, h: int, w: int) -> np.ndarray:
    if m.shape == (h, w):
        return m
    return np.asarray(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BILINEAR)) >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


class Rewriter:
    def __init__(self, cfg: dict):
        from treeground import mllm
        q = cfg["models"]["qwen"]
        self.device = cfg["device"]
        self.qwen, self.proc = mllm.load_qwen(q["repo"], q["revision"], self.device, cfg["features"]["dtype"],
                                              cfg.get("hf_home"))

    @torch.no_grad()
    def __call__(self, image: np.ndarray, text: str) -> tuple[str, str]:
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": REWRITE_PROMPT.format(text=text)}]}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[chat], images=[Image.fromarray(image)], return_tensors="pt").to(self.device)
        gen = self.qwen.generate(**inp, max_new_tokens=24, do_sample=False)
        raw = self.proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return noun_phrase(raw), raw


class Diag:
    # reason_eval.Ours' query and pick, with the pool kept so its coverage can be measured
    def __init__(self, cfg: dict):
        import reason_eval as RE
        import ta_test_eval as TTE
        from treeground import tree as T
        from treeground import tree_attn as TA
        self.TTE, self.T, self.TA = TTE, T, TA
        self.s = TTE.Scorer(cfg)
        if not self.s.pool:
            raise ProtocolError("the diagnosis needs a pool checkpoint")
        self.s.text.clip = RE.safe_text_clip(self.s.text.tok)

    def pool(self, image: np.ndarray, phrases: list[str]):
        # ta_test_eval.Scorer.pools for one query with its SAM 3 phrases given explicitly
        s = self.s
        rles, _ = s.nodes(image)
        inst = s.text(image, phrases, s.ps["instances_per_phrase"])
        pool, imeta = s.S3P.augment(rles, inst, s.ps["max_instances"], s.ps["dup_iou"])
        return self.TTE.Pool(pool, self.T.build_tree(pool, s.device).parent if pool else None, len(rles),
                             self.TA.pool_features(len(rles), imeta))

    def __call__(self, image: np.ndarray, gt: np.ndarray, key: str, text: str, phrases: list[str]) -> dict:
        from treeground import mllm
        from treeground import s2 as S2
        q = self.TTE.S2T.Query("q0", "object", S2.user_text(f"{text} {mllm.SEL}"), f"{S2.user_text(text)} {mllm.SEL}", [])
        p = self.pool(image, phrases)
        if not p.n:
            return {"pick_iou": 0.0, "best_iou": 0.0, "pool": 0, "pick_kind": "none"}
        lg = self.s.logits(image, key, p, [q])[0][0]
        a = int(np.argmax(lg))
        g_small = resize_mask(gt, *image.shape[:2])
        ious = M.iou(p.rles, [M.encode(g_small[None])[0]])[:, 0]
        h, w = gt.shape
        return {"pick_iou": iou(resize_mask(M.decode(p.rles[a]).astype(bool), h, w), gt),
                "best_iou": float(ious.max()), "pick_small_iou": float(ious[a]), "pool": int(p.n),
                "pick_kind": "node" if a < p.n_grid else "inst"}


def run_shard(cfg: dict, shard: int) -> None:
    import s3_pool as S3P
    from treeground import mllm
    n = cfg["reason_diag"]["num_shards"]
    out = Path(cfg["cache_root"]) / "full" / "reason_diag_val" / cfg["reason_diag"]["tag"]
    (out / "shards").mkdir(parents=True, exist_ok=True)
    target = out / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run("reason_diag_val_shard", cfg, False, {"shard": shard, "cache": str(out)})
    recs = val_records(cfg)[shard::n]
    diag, rw = Diag(cfg), Rewriter(cfg)
    rows, t0 = [], time.time()
    for i, (qid, key, text, load_image, load_gt) in enumerate(recs, 1):
        image, gt = load_image(), load_gt()
        if gt.shape != image.shape[:2]:
            raise ProtocolError(f"{qid}: GT {gt.shape} differs from the image {image.shape[:2]}")
        row = {"qid": qid, "text": text, "gt_frac": float(gt.mean())}
        small, _ = downsize(image, cfg["reason_eval"]["ours_max_side"])
        phrase, raw = rw(small, text)
        row.update(rewrite_phrase=phrase, rewrite_raw=raw)
        plain = S3P.phrases(f"{text} {mllm.SEL}")   # exactly the phrases Scorer.pools derives for this query
        for name, img, ph in (("cap1536", small, plain), ("full", image, plain), ("rewrite", small, [phrase] if phrase else plain)):
            row.update({f"{name}_{k}": v for k, v in diag(img, gt, key, text, ph).items()})
        rows.append(row)
        print(f"[{i}/{len(recs)}] {time.time() - t0:.0f}s  {qid}  '{phrase}'  "
              + "  ".join(f"{s} {row[f'{s}_pick_iou']:.2f}/{row[f'{s}_best_iou']:.2f}" for s in SETTINGS), flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"queries": len(rows), "seconds": time.time() - t0})


def summarize(df: pd.DataFrame) -> dict:
    bins = pd.cut(df.gt_frac, SIZE_BINS, right=False, labels=[f"{a:.0%}-{b:.0%}" for a, b in zip(SIZE_BINS, SIZE_BINS[1:])])
    out = {"queries": int(len(df)), "settings": {}}
    for s in SETTINGS:
        g = {"pick_gIoU": float(df[f"{s}_pick_iou"].mean()), "best_in_pool": float(df[f"{s}_best_iou"].mean()),
             "pool_has_target@0.5": float((df[f"{s}_best_iou"] >= 0.5).mean()),
             "picked_given_covered@0.5": float((df[f"{s}_pick_iou"][df[f"{s}_best_iou"] >= 0.5] >= 0.5).mean()),
             "by_size": {}}
        for b, sub in df.groupby(bins, observed=True):
            g["by_size"][str(b)] = {"n": int(len(sub)), "pick_gIoU": float(sub[f"{s}_pick_iou"].mean()),
                                    "best_in_pool": float(sub[f"{s}_best_iou"].mean())}
        out["settings"][s] = g
    return out


def merge(cfg: dict) -> None:
    n = cfg["reason_diag"]["num_shards"]
    d = Path(cfg["cache_root"]) / "full" / "reason_diag_val" / cfg["reason_diag"]["tag"] / "shards"
    miss = [k for k in range(n) if not (d / f"{k:05d}.parquet").exists()]
    if miss:
        raise ProtocolError(f"{len(miss)} of {n} shards missing in {d}")
    df = pd.concat([pd.read_parquet(d / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if df.qid.duplicated().any() or len(df) != len(val_records(cfg)):
        raise ProtocolError(f"{len(df)} rows for {len(val_records(cfg))} val queries")
    run = start_run("reason_diag_val", cfg, False, {"cache": str(d.parent)})
    df.to_csv(run / "eval_queries.csv", index=False)
    res = summarize(df)
    write_json(run / "metrics.json", {"label": "val", "what": "ReasonSeg val diagnosis (development only)",
                                      "rewrite_prompt": REWRITE_PROMPT, "results": res})
    print(json.dumps(res, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reason_diag.yaml")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(ROOT / args.config)
    merge(cfg) if args.merge else run_shard(cfg, args.shard)


if __name__ == "__main__":
    main()
