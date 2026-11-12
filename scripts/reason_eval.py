import argparse
import functools
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, load_config, require_approved, start_run,  # noqa: E402
                            write_json)

# Deviation D50 (supplementary): reasoning queries, test splits only, nothing tuned.
#   instructpart_human / instructpart_gpt  InstructPart test (Wan et al. 2025, Task Reasoning Part Segmentation):
#       test/data_test.json, one query per (image, object, part) = instruction[0] (human-written) or [1] (GPT-4
#       rewritten); image test/images/<base>-<object>-<part>.jpg, GT test/masks/<same>.png > 0 (the D38 files)
#   reasonseg  ReasonSeg test (LISA; the Seg-Zero / VisionReasoner release Ricky06662/ReasonSeg_test, 779 rows):
#       query = text, GT = mask
# Every method answers each query with one mask: ours the top-1 candidate of its pool for the query given as free
# text (as ta_test_eval.GresOurs on gRefCOCO); baselines the D13 / D31 single-mask predictors with the same text
# (SAM 3 text clipped to its 32-token encoder, the D27 rule). gIoU = mean per-query IoU, cIoU = sum I / sum U.
# Ours sees images resized to a long side of reason_eval.ours_max_side (ReasonSeg has 4K photos; the full-resolution
# pool took ~150 s per query); its chosen mask is resized back to the image (bilinear, >= 0.5) before scoring.

BENCHMARKS = ("instructpart_human", "instructpart_gpt", "reasonseg")
FILES = ("scripts/reason_eval.py", "scripts/ta_test_eval.py", "scripts/s3_pool.py", "scripts/b_baselines.py",
         "scripts/b_baselines2.py")


# ---- records: (qid, image key, text, image loader, GT loader) ----

def instructpart_records(cfg: dict, which: int) -> list[tuple]:
    d = Path(cfg["data"]["instructpart"]) / "test"
    out = []
    for entry in json.loads((d / "data_test.json").read_text()):
        base = os.path.splitext(entry["image_path"])[0]
        for item in entry["part_list"]:
            name = f"{base}-{item['object'].replace(' ', '_')}-{item['part'].replace(' ', '_')}"
            img, gt = d / "images" / f"{name}.jpg", d / "masks" / f"{name}.png"
            out.append((f"instructpart:{name}", f"instructpart:{name}", item["instruction"][which],
                        lambda p=img: np.asarray(Image.open(p).convert("RGB")),
                        lambda p=gt: (lambda a: a if a.ndim == 2 else a.any(-1))(np.asarray(Image.open(p)) > 0)))
    return out


@functools.lru_cache(maxsize=4)
def parquet_file(path: str) -> pd.DataFrame:
    # the 4 test files (~1.2 GB) are read once per process
    return pd.read_parquet(path)


def reasonseg_records(cfg: dict) -> list[tuple]:
    files = sorted((Path(cfg["data"]["reasonseg"]) / "data").glob("test-*.parquet"))
    if not files:
        raise ProtocolError(f"no ReasonSeg test parquet under {cfg['data']['reasonseg']}/data")
    parts = []
    for f in files:
        m = pd.read_parquet(f, columns=["image_id", "ann_id", "text"])
        parts.append(m.assign(file=str(f), row=np.arange(len(m))))
    meta = pd.concat(parts, ignore_index=True)

    def row(f: str, r: int) -> pd.Series:
        return parquet_file(f).iloc[r]

    out = []
    for m in meta.itertuples():
        out.append((f"reasonseg:{m.image_id}:{m.ann_id}", f"reasonseg:{m.image_id}", m.text,
                    # the JPEGs carry EXIF orientation; the masks are drawn on the upright image
                    lambda f=m.file, r=m.row: np.asarray(
                        ImageOps.exif_transpose(Image.open(io.BytesIO(row(f, r)["image"]["bytes"]))).convert("RGB")),
                    lambda f=m.file, r=m.row: np.asarray([np.asarray(x, bool) for x in row(f, r)["mask"]])))
    return out


def records(cfg: dict, bench: str) -> list[tuple]:
    if bench == "reasonseg":
        return reasonseg_records(cfg)
    return instructpart_records(cfg, 0 if bench == "instructpart_human" else 1)


# ---- predictors: image, key, text -> bool mask or None ----

class Ours:
    def __init__(self, cfg: dict):
        import ta_test_eval as TTE
        self.TTE, self.s = TTE, TTE.Scorer(cfg)
        self.max_side = cfg["reason_eval"]["ours_max_side"]
        if self.s.pool:
            self.s.text.clip = safe_text_clip(self.s.text.tok)

    def __call__(self, image: np.ndarray, key: str, text: str) -> np.ndarray | None:
        h, w = image.shape[:2]
        f = self.max_side / max(h, w)
        if f >= 1:
            return self.select(image, key, text)
        small = np.asarray(Image.fromarray(image).resize((round(w * f), round(h * f)), Image.BILINEAR))
        m = self.select(small, key, text)
        return None if m is None else upsample(m, h, w)

    def select(self, image: np.ndarray, key: str, text: str) -> np.ndarray | None:
        from treeground import masks as M
        from treeground import mllm
        from treeground import s2 as S2
        T = self.TTE
        q = T.S2T.Query("q0", "object", S2.user_text(f"{text} {mllm.SEL}"), f"{S2.user_text(text)} {mllm.SEL}", [])
        pools, lgs = self.s.score(image, key, [f"{text} {mllm.SEL}"], [q])
        if not pools[0].n:
            return None
        return M.decode(pools[0].rles[int(np.argmax(lgs[0][0]))])


# '_na' (no-abstain): the same predictor, but its top mask is kept when the score is below the official threshold,
# as ours and Sa2VA always answer and every reasoning query has a target (as b_parity's sam3_union_na)
NO_ABSTAIN = "_na"


def safe_text_clip(tok):
    # s3_pool.TextInstances.clip (SAM 3's 32-token text encoder) drops trailing words only, so a single word longer
    # than the limit reaches the encoder and fails (one ReasonSeg query); the baselines' rule adds the token-level
    # cut. Identical output whenever the word rule suffices.
    import b_baselines as B
    clip = Baseline._clipper(tok, lambda s: s)
    return lambda p: clip(B.concept(p))


def is_sentence(text: str) -> bool:
    # Sa2VA's own evaluation switches to its question prompt on '?'; instructions end with '.' or '!'
    return text.rstrip().endswith(("?", ".", "!"))


def upsample(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    m = Image.fromarray(mask.astype(np.float32)).resize((w, h), Image.BILINEAR)
    return np.asarray(m) >= 0.5


class Baseline:
    def __init__(self, cfg: dict, method: str):
        import b_baselines as B
        self.abstain = not method.endswith(NO_ABSTAIN)
        method = method.removesuffix(NO_ABSTAIN)
        if method not in B.PREDICTORS:
            import b_baselines2  # noqa: F401  (registers the D31 predictors)
        self.pred = B.PREDICTORS[method](cfg)
        # reasoning queries are sentences or instructions; the phrase prompt ("Please segment {phrase} in this
        # image.") would garble them, so a sentence gets the model's own sentence prompt (reason_prompts)
        sent = cfg.get("reason_prompts", {}).get(method)
        self.prompts = None if sent is None else {False: self.pred.s["prompt"], True: sent}
        tok = getattr(getattr(self.pred, "proc", None), "tokenizer", None) if method.startswith("sam3") else None
        self.clip = (lambda p: p) if tok is None else self._clipper(tok, B.concept)

    @staticmethod
    def _clipper(tok, concept):
        limit = min(tok.model_max_length, 32)

        def n(s: str) -> int:
            return len(tok(concept(s)).input_ids)

        def clip(p: str) -> str:
            words = p.split()
            while len(words) > 1 and n(" ".join(words)) > limit:
                words.pop()
            s = " ".join(words)
            if n(s) > limit:   # a single over-long word: keep its leading tokens
                s = tok.decode(tok(concept(s), truncation=True, max_length=limit).input_ids, skip_special_tokens=True)
            return s
        return clip

    def __call__(self, image: np.ndarray, key: str, text: str) -> np.ndarray | None:
        if self.prompts is not None:
            self.pred.s = {**self.pred.s, "prompt": self.prompts[is_sentence(text)]}
        mask, _, _, acc = self.pred(image, [self.clip(text)])[0]
        return mask.astype(bool) if (mask is not None and (acc or not self.abstain)) else None


# ---- bookkeeping ----

def is_ours(method: str) -> bool:
    return method == "ours"


def stage_name(cfg: dict, method: str) -> str:
    if not is_ours(method):
        return method
    import ta_test_eval as TTE
    return TTE.TAT.stage(cfg)


def identity(cfg: dict, method: str, bench: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "reason_eval", "benchmark": bench, "method": method, "stage": stage_name(cfg, method),
             "num_shards": cfg["reason_eval"]["num_shards"], "code": h.hexdigest(),
             "data": cfg["datasets"]["reasonseg"]["revision"] if bench == "reasonseg" else "instructpart test"}
    if is_ours(method):
        import ta_test_eval as TTE
        ckpt, run = TTE.TAE.load_checkpoint(cfg, False)
        ident.update(run=run.name, checkpoint=cfg["ta_eval"]["checkpoint"], step=ckpt["meta"]["step"],
                     layer=ckpt["meta"]["layer"], pool_settings=ckpt["meta"].get("pool_settings"),
                     sam3_text=cfg["baselines"]["sam3_text"], max_side=cfg["reason_eval"]["ours_max_side"])
    else:
        ident.update(models=cfg["models"], baselines=cfg["baselines"], reason_prompts=cfg.get("reason_prompts", {}))
    return ident


def approvals(cfg: dict, method: str) -> list[str]:
    if is_ours(method):
        import ta_test_eval as TTE
        return ["D50"] + TTE.TAT.run_deviations(cfg["ta"])
    return ["D50", "D13", "D31"]


def out_dir(cfg: dict, method: str, bench: str, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / "full" / "reason_eval" / stage_name(cfg, method) / bench /
            canonical_hash(ident)[:16])


def run_shard(cfg: dict, method: str, bench: str, shard: int) -> None:
    require_approved(cfg, approvals(cfg, method))
    n = cfg["reason_eval"]["num_shards"]
    ident = identity(cfg, method, bench)
    d = out_dir(cfg, method, bench, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"reason_{bench}_{stage_name(cfg, method)}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    recs = records(cfg, bench)[shard::n]
    predict = Ours(cfg) if is_ours(method) else Baseline(cfg, method)
    rows, t0 = [], time.time()
    for i, (qid, key, text, load_image, load_gt) in enumerate(recs, 1):
        image, gt = load_image(), load_gt()
        if gt.shape != image.shape[:2]:
            raise ProtocolError(f"{qid}: GT {gt.shape} differs from the image {image.shape[:2]}")
        pred = predict(image, key, text)
        pred = np.zeros(gt.shape, bool) if pred is None else pred
        rows.append({"qid": qid, "image": key, "inter": int((pred & gt).sum()), "union": int((pred | gt).sum()),
                     "empty": bool(not pred.any())})
        if i % 25 == 0:
            print(f"[{i}/{len(recs)}] {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"queries": len(rows), "seconds": time.time() - t0})


def merge(cfg: dict, method: str, bench: str) -> None:
    require_approved(cfg, approvals(cfg, method))
    n = cfg["reason_eval"]["num_shards"]
    d = out_dir(cfg, method, bench, identity(cfg, method, bench))
    miss = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if miss:
        raise ProtocolError(f"{len(miss)} of {n} shards missing in {d}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if df.qid.duplicated().any() or len(df) != len(records(cfg, bench)):
        raise ProtocolError(f"{len(df)} rows do not match the benchmark's queries")
    bs = cfg["v2"]["bootstrap"]
    run = start_run(f"reason_{bench}_{stage_name(cfg, method)}", cfg, False, {"cache": str(d)})
    df.to_csv(run / "eval_queries.csv", index=False)
    iou = np.where(df.union > 0, df.inter / df.union.clip(lower=1), 0.0)
    res = {"queries": int(len(df)), "gIoU": mean_ci(iou, df.image.to_numpy(), bs["n"], bs["seed"]),
           "cIoU": float(df.inter.sum() / max(1, df.union.sum())), "empty_rate": float(df["empty"].mean())}
    write_json(run / "metrics.json", {"label": "test", "deviation": "D50", "benchmark": bench, "method": method,
                                      "stage": stage_name(cfg, method), "results": res})
    print(json.dumps({"gIoU": round(res["gIoU"]["mean"] * 100, 2), "cIoU": round(res["cIoU"] * 100, 2),
                      "empty_rate": round(res["empty_rate"], 3)}))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method, args.benchmark)
    else:
        if not 0 <= args.shard < cfg["reason_eval"]["num_shards"]:
            raise ProtocolError(f"shard {args.shard} outside 0..{cfg['reason_eval']['num_shards'] - 1}")
        run_shard(cfg, args.method, args.benchmark, args.shard)


if __name__ == "__main__":
    main()
