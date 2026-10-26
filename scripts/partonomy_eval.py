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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import node_cap  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# Deviation D32: PARTONOMY-Core (arXiv 2505.20759) segmentation in its "(gt)" setting, exactly as the official
# evaluation (AnselBlume/partonomy: validate_partonomy.py, evaluation/evaluators/masks.py): for every question of
# the Identification / Intersection (positive) / Difference (negative) types, the parts of the correct answer
# choice are given; each part is segmented separately and paired with its GT mask (union of that part's instance
# masks); IoU with empty-union = 1; micro-gIoU = mean over all (question, part) pairs, macro-gIoU = mean over
# questions of the per-question mean. A part shared by several questions of an image is predicted once.
# Query per part: 'the {part} of the {object}' ({object} = the image label). Ours: single step, no rejection, top-1
# over the capped D3/D4 nodes ('nodes') and over nodes + the D30 SAM 3 instances of the part phrase and of
# 'the {object}' ('aug'); baselines: the D13/D31 predictors on the same phrase (an abstention is an empty mask).

QTYPES = {"identification_with_label": "identification", "positive_with_label": "intersection",
          "negative_with_label": "difference"}
FILES = ("scripts/partonomy_eval.py", "scripts/s3_pool.py", "scripts/s3_eval.py", "scripts/s2_train.py",
         "scripts/b_baselines.py", "scripts/b_baselines2.py")


def questions(root: Path) -> list[dict]:
    # a correct-answer part without a mask of that name (142 Intersection parts, e.g. 'landing gear' vs the stored
    # 'fixed landing gear') is skipped and counted; Identification and Difference have none
    qs = [r for r in json.loads((root / "test-qa_pairs.json").read_text()) if r["question_type"] in QTYPES]
    for r in qs:
        gt = r["answer_parts"][r["answer_types"].index("correct")]
        r["parts"] = [p for p in gt if p in (r["segmentations"] or {})]
        r["skipped_parts"] = len(gt) - len(r["parts"])
    return [r for r in qs if r["parts"]]


def phrase(part: str, obj: str) -> str:
    return f"the {part} of the {obj}"


def gt_mask(segs: list[dict], hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(hw, bool)
    for s in segs:
        s = s.get("segmentation", s)
        m = M.decode({"size": list(s["size"]), "counts": s["counts"]}).astype(bool)
        if m.shape != hw:  # the official loader resizes mismatching masks to the image size
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.NEAREST)) > 127
        out |= m
    return out


def iou(pred: np.ndarray | None, gt: np.ndarray) -> float:
    p = np.zeros_like(gt) if pred is None else pred.astype(bool)
    u = np.logical_or(p, gt).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, gt).sum() / u)


def items(qs: list[dict]) -> pd.DataFrame:
    rows = {}
    for r in qs:
        for p in r["parts"]:
            rows.setdefault((r["image_path"], p), {"image": r["image_path"], "label": r["image_label"], "part": p})
    return pd.DataFrame(sorted(rows.values(), key=lambda x: (x["image"], x["part"])))


class Ours:
    def __init__(self, cfg: dict):
        import s3_eval as S3E
        import s3_pool as S3P
        import s2_nodes as S2N
        from treeground import proposals as P
        self.S3P, self.S2N, self.cfg = S3P, S2N, cfg
        from treeground.identity import qwen_static
        dev = cfg["device"]
        self.model, self.proc, self.extra, self.sel, self.meta, self.run = S3E.load_model(cfg, qwen_static(cfg), dev)
        m, p = cfg["models"], cfg["proposals"]
        self.sam, self.sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], dev, p["dtype"], cfg.get("hf_home"))
        self.text = S3P.TextInstances(cfg)
        self.cap = node_cap(cfg, False)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, label: str, parts: list[str]) -> list[dict]:
        import s2_train as S2T
        import x_transfer_s2 as X2
        from treeground import s2 as S2
        nodes = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        k = self.cfg["s3_pool"]["instances_per_phrase"]
        plist = [phrase(p, label) for p in parts] + [f"the {label}"]
        pool, owner = list(nodes), {}
        for ph, lst in zip(plist, self.text(image, plist, k)):
            owner[ph] = []
            for msk, _ in lst:
                if msk.any():
                    owner[ph].append(len(pool))
                    pool.append(M.encode(msk)[0])
        qs = [S2T.Query(f"partonomy|part|{label}:{p}", "part", S2.user_text(f"Locate {phrase(p, label)}. {mllm.SEL}"),
                        f"{p} of the {label} {mllm.SEL}", []) for p in parts]
        out = []
        if not pool:
            return [{"nodes": None, "aug": None} for _ in parts]
        z, probes = S2T.recompute(self.model, X2.ImageSource(self.proc, self.extra["sel_id"], image, pool), None, qs,
                                  self.meta["layer"], self.cfg["device"], self.cfg["s2_eval"]["batch_size"])
        e = self.sel.nodes(z.half().to(self.cfg["device"]))
        n = len(nodes)
        for p, pr in zip(parts, probes):
            s = self.sel.scores(self.sel.queries(pr.half()), e)[0].float().cpu().numpy()
            idx = np.array(list(range(n)) + owner[phrase(p, label)] + owner[f"the {label}"], np.int64)
            a_nodes = int(np.argmax(s[:n])) if n else -1
            a_aug = int(idx[int(np.argmax(s[idx]))])
            out.append({"nodes": M.decode(pool[a_nodes]).astype(bool) if a_nodes >= 0 else None,
                        "aug": M.decode(pool[a_aug]).astype(bool), "aug_kind": "node" if a_aug < n else "inst"})
        return out


class Baseline:
    def __init__(self, cfg: dict, method: str):
        import b_baselines as B
        if method not in B.PREDICTORS:
            import b_baselines2  # noqa: F401  (registers the D31 predictors)
        self.pred = B.PREDICTORS[method](cfg)

    def __call__(self, image: np.ndarray, label: str, parts: list[str]) -> list[dict]:
        res = self.pred(image, [phrase(p, label) for p in parts])
        return [{"pred": (m if acc else None)} for m, _, _, acc in res]


def stage_name(cfg: dict, method: str) -> str:
    if method != "ours":
        return method
    import s2_train as S2T
    return S2T.s2_stage(cfg)


def identity(cfg: dict, method: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "partonomy_eval", "method": method, "stage": stage_name(cfg, method), "code": h.hexdigest(),
             "num_shards": cfg["partonomy_eval"]["num_shards"]}
    if method == "ours":
        import s2_eval as S2E
        run = S2E.load_checkpoint(cfg, False)[1]
        ident.update(run=run.name, checkpoint=cfg["s2_eval"]["checkpoint"], pool=cfg["s3_pool"]["instances_per_phrase"])
    else:
        ident.update(models={k: v for k, v in cfg["models"].items()}, baselines=cfg["baselines"])
    return ident


def out_dir(cfg: dict, method: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "partonomy_eval" / stage_name(cfg, method) / canonical_hash(ident)[:16]


def approvals(method: str) -> list[str]:
    return ["D32"] + (["D10", "D21", "D30"] if method == "ours" else ["D13"])


def run_shard(cfg: dict, method: str, shard: int) -> None:
    require_approved(cfg, approvals(method))
    root = Path(cfg["data"]["partonomy"])
    df = items(questions(root))
    n = cfg["partonomy_eval"]["num_shards"]
    images = sorted(df.image.unique())[shard::n]
    ident = identity(cfg, method)
    d = out_dir(cfg, method, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"partonomy_{stage_name(cfg, method)}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    predict = Ours(cfg) if method == "ours" else Baseline(cfg, method)
    segs = {}
    for r in questions(root):
        segs.setdefault(r["image_path"], {}).update(r["segmentations"])
    rows, t0 = [], time.time()
    for i, img in enumerate(images, 1):
        g = df[df.image == img]
        image = load_rgb(assert_real_path(str(root / img)))
        hw = image.shape[:2]
        parts = g.part.tolist()
        preds = predict(image, g.label.iloc[0], parts)
        for p, pr in zip(parts, preds):
            gt = gt_mask(segs[img][p], hw)
            row = {"image": img, "part": p, "gt_area": int(gt.sum())}
            for k, v in pr.items():
                row[f"iou_{k}" if k != "aug_kind" else k] = v if k == "aug_kind" else iou(v, gt)
            rows.append(row)
        print(f"[{i}/{len(images)}] {img} parts {len(parts)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "parts": len(rows), "seconds": time.time() - t0})


def summarize(qs: list[dict], df: pd.DataFrame, col: str, bs: dict) -> dict:
    by = {(r.image, r.part): getattr(r, col) for r in df.itertuples()}
    out = {}
    for qt, name in QTYPES.items():
        pairs, per_q, groups = [], [], []
        for r in qs:
            if r["question_type"] != qt:
                continue
            v = [by[(r["image_path"], p)] for p in r["parts"]]
            pairs += v
            groups += [r["image_path"]] * len(v)
            per_q.append(float(np.mean(v)))
        qimg = [r["image_path"] for r in qs if r["question_type"] == qt]
        skipped = sum(r["skipped_parts"] for r in qs if r["question_type"] == qt)
        out[name] = {"skipped_parts_without_mask": skipped, "micro_gIoU": mean_ci(np.array(pairs), np.array(groups), bs["n"], bs["seed"]),
                     "macro_gIoU": mean_ci(np.array(per_q), np.array(qimg), bs["n"], bs["seed"]),
                     "questions": len(per_q), "parts": len(pairs)}
    return out


def merge(cfg: dict, method: str) -> None:
    require_approved(cfg, approvals(method))
    root = Path(cfg["data"]["partonomy"])
    qs = questions(root)
    ident = identity(cfg, method)
    d = out_dir(cfg, method, ident)
    n = cfg["partonomy_eval"]["num_shards"]
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if len(df) != len(items(qs)) or df.duplicated(["image", "part"]).any():
        raise ProtocolError(f"{len(df)} part predictions for {len(items(qs))} (image, part) pairs")
    bs = cfg["v2"]["bootstrap"]
    cols = [c for c in df.columns if c.startswith("iou_")]
    run = start_run(f"partonomy_{stage_name(cfg, method)}", cfg, False, {"cache": str(d)})
    metrics = {"deviation": "D32", "label": "PARTONOMY-Core (gt) setting", "method": method,
               "stage": stage_name(cfg, method), "results": {c[4:]: summarize(qs, df, c, bs) for c in cols},
               "published_reference": "arXiv 2505.20759 Table 2 (gt rows: X-Decoder, SEEM, Grounded SAM 2)"}
    if "aug_kind" in df:
        metrics["aug_pick_from_instances"] = float((df.aug_kind == "inst").mean())
    df.to_csv(run / "parts.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(json.dumps({k: {q: round(v[q]["micro_gIoU"]["mean"], 3) for q in v} for k, v in metrics["results"].items()}))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method)
    else:
        with torch.no_grad():
            run_shard(cfg, args.method, args.shard)


if __name__ == "__main__":
    main()
