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
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config,  # noqa: E402
                            require_approved, start_run, write_json)

# Deviation D35: PACO-LVIS test (paco_lvis_v1_test.json), the held-out eval split of the manifest (its images win
# precedence over every train copy, so no test image is in any training set; checked again here by sha1). Every test
# image, the same I1 queries as val (paco.i1_queries: object, part, chain). Nothing is tuned on test: ours answers
# top-1 over the query's D30 pool (capped nodes computed on the fly + its phrases' top-3 SAM 3 text instances), with
# no rejection threshold, exactly the s3_eval val rule (s3_eval.image_rows); baselines are the D13 / D31 predictors
# with their fixed settings on the same phrase as on val. Run once per method; numbers are reported as they come.

FILES = ("scripts/paco_test_eval.py", "scripts/s3_eval.py", "scripts/s3_pool.py", "scripts/s2_train.py",
         "scripts/s2_nodes.py", "scripts/b_baselines.py", "scripts/b_baselines2.py")


def stage_name(cfg: dict, method: str) -> str:
    if method != "ours":
        return method
    import s2_train as S2T
    return S2T.s2_stage(cfg)


def test_images(cfg: dict) -> pd.DataFrame:
    _, manifest, _, wl = manifest_inputs(cfg, False)
    test = manifest[(manifest.source == "paco_lvis") & (manifest.split == "test")]
    train = set(wl[wl.role.str.endswith("_train") | wl.role.str.endswith("_train_nt")].sha1)
    if test.sha1.isin(train).any():
        raise ProtocolError(f"{int(test.sha1.isin(train).sum())} PACO test images are in a training worklist")
    return test.sort_values("image_key").reset_index(drop=True)


def queries(cfg: dict, images: pd.DataFrame, annos: dict) -> pd.DataFrame:
    rows = []
    for r in images.itertuples():
        img = annos.get(int(r.image_key.split(":")[1]))
        if img is None:
            continue
        for q in paco.i1_queries(img)[0]:
            rows.append({"qid": q.qid, "kind": q.kind, "text": q.text, "n_sel": q.n_sel, "gt": json.dumps(q.gt),
                         "image_key": r.image_key, "sha1": r.sha1, "path": r.path})
    df = pd.DataFrame(rows)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate query ids")
    return df


def identity(cfg: dict, method: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "paco_test_eval", "method": method, "stage": stage_name(cfg, method),
             "num_shards": cfg["paco_test"]["num_shards"], "code": h.hexdigest()}
    if method == "ours":
        import s2_eval as S2E
        ident.update(run=S2E.load_checkpoint(cfg, False)[1].name, checkpoint=cfg["s2_eval"]["checkpoint"],
                     pool=cfg["s3_pool"]["instances_per_phrase"], sam3_text=cfg["baselines"]["sam3_text"])
    else:
        ident.update(models=cfg["models"], baselines=cfg["baselines"])
    return ident


def out_dir(cfg: dict, method: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "paco_test_eval" / stage_name(cfg, method) / canonical_hash(ident)[:16]


def approvals(cfg: dict, method: str) -> list[str]:
    if method != "ours":
        return ["D35", "D13", "D31"]
    import s2_train as S2T
    return ["D35", "D30"] + S2T.deviations(cfg)


class Ours:
    def __init__(self, cfg: dict):
        import s2_nodes as S2N
        import s3_eval as S3E
        import s3_pool as S3P
        from treeground import proposals as P
        from treeground.identity import node_cap, qwen_static
        self.cfg, self.S2N, self.S3E = cfg, S2N, S3E
        dev = cfg["device"]
        self.model, self.proc, self.extra, self.sel, self.meta, _ = S3E.load_model(cfg, qwen_static(cfg), dev)
        m, p = cfg["models"], cfg["proposals"]
        self.sam, self.sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], dev, p["dtype"], cfg.get("hf_home"))
        self.text = S3P.TextInstances(cfg)
        self.cap = node_cap(cfg, False)

    def __call__(self, image: np.ndarray, g: pd.DataFrame, gts: dict) -> list[dict]:
        nodes = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        return self.S3E.image_rows(self.model, self.proc, self.extra, self.sel, self.meta["layer"], self.cfg["device"],
                                   self.text, self.cfg, g, image, nodes, gts)


class Baseline:
    def __init__(self, cfg: dict, method: str):
        import b_baselines as B
        if method not in B.PREDICTORS:
            import b_baselines2  # noqa: F401
        self.B, self.pred = B, B.PREDICTORS[method](cfg)

    def __call__(self, image: np.ndarray, g: pd.DataFrame, gts: dict) -> list[dict]:
        g = g[g.kind.isin(["object", "part"])]
        rows = []
        for q, (mask, score, raw, acc) in zip(g.itertuples(), self.pred(image, [self.B.phrase(t) for t in g.text])):
            if mask is not None and mask.shape != image.shape[:2]:
                raise ProtocolError(f"{q.qid}: mask {mask.shape} differs from the image {image.shape[:2]}")
            rle = M.encode(mask)[0] if mask is not None and mask.any() else None
            gt = gts[json.loads(q.gt)[0]]
            rows.append({"qid": q.qid, "kind": q.kind, "image": q.image_key, "present": True, "score": score,
                         "accepted": bool(acc), **self.B.overlap(rle if acc else None, gt),
                         "iou_any": self.B.overlap(rle, gt)["iou"]})
        return rows


def run_shard(cfg: dict, method: str, shard: int) -> None:
    require_approved(cfg, approvals(cfg, method))
    n = cfg["paco_test"]["num_shards"]
    images = test_images(cfg).iloc[shard::n]
    ident = identity(cfg, method)
    d = out_dir(cfg, method, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"pacotest_{stage_name(cfg, method)}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    annos = paco.load(cfg["data"]["paco_lvis"], "test", {int(k.split(":")[1]) for k in images.image_key})
    qs = queries(cfg, images, annos)
    predict = Ours(cfg) if method == "ours" else Baseline(cfg, method)
    rows, t0 = [], time.time()
    for i, (key, g) in enumerate(qs.groupby("image_key", sort=True), 1):
        img = annos[int(key.split(":")[1])]
        image = load_rgb(assert_real_path(g.path.iloc[0]))
        if image.shape[:2] != (img.h, img.w):
            raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {(img.h, img.w)}")
        rows += predict(image, g.reset_index(drop=True), paco.gt_rles(img))
        if i % 25 == 0:
            print(f"[{i}] queries {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "queries": len(rows), "seconds": time.time() - t0})


def summarize_ours(df: pd.DataFrame, bs: dict) -> dict:
    def ci(v, frame):
        return mean_ci(np.asarray(v, float), frame.image.to_numpy(), bs["n"], bs["seed"])
    pres = df[df.kind.isin(["object", "part"])].reset_index(drop=True)
    union = np.where(pres.sel_union > 0, pres.sel_union, pres.gt_area)
    out = {"present_queries": int(len(pres)), "gIoU": ci(pres.sel_iou, pres),
           "success@0.5": ci(pres.sel_iou >= 0.5, pres),
           "cIoU": float(pres.sel_inter.sum() / max(1, union.sum())),
           "by_kind": {k: ci(pres.sel_iou[pres.kind == k], pres[pres.kind == k]) for k in ("object", "part")},
           "pick_from_instances": float((pres.pick_kind == "inst").mean()),
           "oracle_best_pool": float(pres.best_pool_iou.mean())}
    ch = df[df.kind == "chain"].reset_index(drop=True)
    if len(ch):
        out["chain"] = {"chains": int(len(ch)), "part_gIoU_flat": ci(ch.part_flat_iou, ch),
                        "part_gIoU_tree": ci(ch.part_tree_iou, ch),
                        "part_success_flat": ci(ch.part_flat_iou >= 0.5, ch),
                        "part_success_tree": ci(ch.part_tree_iou >= 0.5, ch)}
    return out


def merge(cfg: dict, method: str) -> None:
    require_approved(cfg, approvals(cfg, method))
    d = out_dir(cfg, method, identity(cfg, method))
    n = cfg["paco_test"]["num_shards"]
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate query rows")
    bs = cfg["v2"]["bootstrap"]
    run = start_run(f"pacotest_{stage_name(cfg, method)}", cfg, False, {"cache": str(d)})
    df.to_csv(run / "eval_queries.csv", index=False)
    if method == "ours":
        res = summarize_ours(df, bs)
    else:
        import b_baselines as B
        res = B.summarize(df, bs)[0]
    write_json(run / "metrics.json", {"label": "test", "deviation": "D35", "split": "PACO-LVIS test",
                                      "method": method, "stage": stage_name(cfg, method),
                                      "images": int(df.image.nunique()), **res})
    print(json.dumps(res.get("gIoU") or res["present"]["gIoU"]))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method)
    else:
        if not 0 <= args.shard < cfg["paco_test"]["num_shards"]:
            raise ProtocolError(f"--shard must be in 0..{cfg['paco_test']['num_shards'] - 1}")
        with torch.no_grad():
            run_shard(cfg, args.method, args.shard)


if __name__ == "__main__":
    main()
