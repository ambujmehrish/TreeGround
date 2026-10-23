import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs  # noqa: E402
from treeground.run import (REPO, ProtocolError, canonical_hash, load_config, require_approved,  # noqa: E402
                            resolve_run, start_run, write_json)

# D13 baselines on the D7/D8 transfer sets (deviation D16): images, chains and GT from x_transfer.records, one
# object query per chain object and one part query per chain, scored as x_transfer scores nodes (masks by mask IoU,
# OpenImages boxes by the box IoU of the predicted mask's bounding box). Chains are scored from the independent
# object and part predictions; the common eligible subset is read from the x_transfer merge run's chains_strict.csv.


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("b_baselines")
XT = _load("x_transfer")
COLUMNS = ["qid", "kind", "image", "phrase", "object_qid", "chain_qid", "part_ratio", "box_matched", "score",
           "accepted", "raw", "inter", "union", "pred_area", "gt_area", "iou", "iou_any", "pred_rle"]
CODE = ["scripts/b_transfer.py", "scripts/b_baselines.py", "scripts/x_transfer.py", "treeground/data/pascal_part.py",
        "treeground/data/ade20k.py", "treeground/data/lvis_hier.py", "treeground/data/openimages_hier.py",
        "treeground/data/paco.py"]


def same_gt(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    return list(a["size"]) == list(b["size"]) and a["counts"] == b["counts"]


def image_queries(key: str, chains) -> list[dict]:
    # phrases as the D13 PACO queries: I1 object / part templates on the names x_transfer puts in CHAIN_TEMPLATE
    objects, rows = {}, []
    for qid, obj, part, ogt, pgt in chains:
        oq = f"{key}|object|{obj}"
        if oq in objects:
            if not same_gt(objects[oq], ogt):
                raise ProtocolError(f"{oq}: one object phrase names two different GT regions")
        else:
            objects[oq] = ogt
            rows.append({"qid": oq, "kind": "object", "phrase": B.phrase(paco.OBJECT_TEMPLATE.format(object=obj)),
                         "gt": ogt, "object_qid": oq, "chain_qid": None, "part_ratio": float("nan")})
        rows.append({"qid": qid.replace("|chain|", "|part|", 1), "kind": "part",
                     "phrase": B.phrase(paco.PART_TEMPLATE.format(object=obj, part=part)), "gt": pgt,
                     "object_qid": oq, "chain_qid": qid,
                     "part_ratio": XT.gt_area(pgt) / max(1.0, XT.gt_area(ogt))})
    return rows


def score(pred: M.Rle | None, gt) -> dict:
    if not isinstance(gt, np.ndarray):
        return B.overlap(pred, gt)
    ga = XT.gt_area(gt)
    if pred is None:
        return {"inter": 0.0, "union": ga, "pred_area": 0, "gt_area": ga, "iou": 0.0}
    pb = M.boxes_xyxy([pred])[0]
    wh = np.clip(np.minimum(pb[2:], gt[2:]) - np.maximum(pb[:2], gt[:2]), 0, None)
    inter = float(wh.prod())
    return {"inter": inter, "union": ga + XT.gt_area(pb) - inter, "pred_area": int(M.areas([pred])[0]),
            "gt_area": ga, "iou": float(M.box_iou(gt, pb)[0, 0])}


def chain_table(df: pd.DataFrame) -> pd.DataFrame:
    obj = df[df.kind == "object"].set_index("qid").iou
    parts = df[df.kind == "part"]
    o = obj.reindex(parts.object_qid).to_numpy()
    if np.isnan(o).any():
        raise ProtocolError("part queries without their object query")
    return pd.DataFrame({"qid": parts.chain_qid.to_numpy(), "image": parts.image.to_numpy(), "object_iou": o,
                         "part_iou": parts.iou.to_numpy(), "step1": o >= B.THR, "step2": parts.iou.to_numpy() >= B.THR,
                         "chain": (o >= B.THR) & (parts.iou.to_numpy() >= B.THR),
                         "single_level": parts.part_ratio.to_numpy() >= B.SINGLE_LEVEL})


def summarize(df: pd.DataFrame, ch: pd.DataFrame, bs: dict) -> dict:
    return {"by_kind": {k: B.block(df[df.kind == k], bs) for k in ("object", "part")},
            "chains": {"all": B.chain_block(ch, bs), "two_level": B.chain_block(ch[~ch.single_level], bs),
                       "single_level": B.chain_block(ch[ch.single_level], bs)}}


def common_chains(path: Path) -> tuple[set, set]:
    df = pd.read_csv(path)
    wide = df.pivot(index="qid", columns="structure", values="feasible").astype(bool)
    return set(wide.index[wide.all(axis=1)]), set(wide.index)


def restrict(df: pd.DataFrame, chains: set) -> pd.DataFrame:
    parts = df[(df.kind == "part") & df.chain_qid.isin(chains)]
    return pd.concat([df[(df.kind == "object") & df.qid.isin(set(parts.object_qid))], parts])


def common_block(cfg: dict, dataset: str, df: pd.DataFrame, ch: pd.DataFrame, bs: dict, smoke: bool):
    stage = cfg["b_transfer"]["common_stage"].format(dataset=dataset)
    try:
        run = resolve_run(cfg, stage, cfg.get("inputs", {}).get(stage, "latest"), False)
    except ProtocolError as e:
        return f"not available: {e}"
    path = run / "chains_strict.csv"
    if not path.exists():
        return f"not available: {path} missing"
    common, listed = common_chains(path)
    ours = set(ch.qid)
    out = {"source_run": f"{stage}/{run.name}", "x_transfer_chains": len(listed), "baseline_chains": len(ours),
           "shared_chains": len(ours & listed), "chains_match_x_transfer": None if smoke else ours == listed,
           "size": len(common & ours), "fraction": len(common & ours) / max(1, len(ours & listed))}
    out.update(summarize(restrict(df, common & ours), ch[ch.qid.isin(common)], bs))
    return out


def identity(cfg: dict, method: str, dataset: str, smoke: bool) -> dict:
    used = {"sam3_text": ["sam3"], "qwen_box_sam": ["qwen", "sam3"], "sa2va": ["sa2va"]}[method]
    h = hashlib.sha256()
    for rel in CODE:
        h.update((REPO / rel).read_bytes())
    return {"kind": "baseline_transfer", "method": method, "dataset": dataset, "smoke": smoke,
            "smoke_images": cfg["b_transfer"]["smoke_images"] if smoke else None,
            "manifest": manifest_inputs(cfg, False)[2] if dataset == "lvis_hier" else None,
            "transfer": cfg["transfer"][dataset], "openimages_max_side": XT.OPENIMAGES_MAX_SIDE,
            "settings": cfg["baselines"][method], "models": {k: cfg["models"][k] for k in used},
            "dtype": cfg["proposals"]["dtype"], "templates": paco.TEMPLATES, "code": h.hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if ident["smoke"] else "full") / "baselines_transfer"
            / ident["dataset"] / ident["method"] / canonical_hash(ident)[:16])


def deviations(dataset: str) -> list[str]:
    return ["D7", "D13", "D16"] + (["D8"] if dataset in ("lvis_hier", "openimages_hier") else []) \
        + (["D9"] if dataset == "openimages_hier" else [])


def images(cfg: dict, dataset: str, shard: int, num_shards: int, smoke: bool):
    n = 0
    for rec in XT.records(cfg, dataset, shard, num_shards):
        if not rec[3]:
            continue
        yield rec
        n += 1
        if smoke and n >= cfg["b_transfer"]["smoke_images"]:
            return


def run_shard(cfg: dict, method: str, dataset: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, deviations(dataset))
    ident = identity(cfg, method, dataset, smoke)
    d = out_dir(cfg, ident)
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    run = start_run(f"b_transfer_{method}_{dataset}", cfg, smoke,
                    {"shard": shard, "num_shards": num_shards, "cache": str(d)})
    predict = B.PREDICTORS[method](cfg)
    rows, t0, n = [], time.time(), 0
    for key, image, hw, chains, _ in images(cfg, dataset, shard, num_shards, smoke):
        t1 = time.time()
        if image.shape[:2] != hw:
            raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation {hw}")
        qs = image_queries(key, chains)
        for q, (mask, sc, raw, accepted) in zip(qs, predict(image, [q["phrase"] for q in qs])):
            if mask is not None and mask.shape != image.shape[:2]:
                raise ProtocolError(f"{q['qid']}: mask {mask.shape} differs from the image {image.shape[:2]}")
            rle = M.encode(mask)[0] if mask is not None and mask.any() else None
            gt = q.pop("gt")
            rows.append({**q, "image": key, "box_matched": isinstance(gt, np.ndarray), "score": sc,
                         "accepted": bool(accepted), "raw": raw, **score(rle if accepted else None, gt),
                         "iou_any": score(rle, gt)["iou"], "pred_rle": json.dumps(rle if accepted else None)})
        n += 1
        print(f"[{n}] {key} chains {len(chains)} queries {len(qs)} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": n, "queries": len(rows), "seconds": time.time() - t0})


def merge(cfg: dict, method: str, dataset: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, deviations(dataset))
    ident = identity(cfg, method, dataset, smoke)
    d = out_dir(cfg, ident)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {method} {dataset} shards missing under {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    if df.qid.duplicated().any():
        raise ProtocolError(f"{method} {dataset}: duplicate query ids across shards")
    run = start_run(f"b_transfer_{method}_{dataset}_merge", cfg, smoke, {"cache": str(d)})
    bs = cfg["v4"]["bootstrap"]
    ch = chain_table(df)
    metrics = {"method": method, "dataset": dataset, "label": f"transfer:{dataset}", "deviation": "D16",
               "smoke": smoke, "box_matched": bool(df.box_matched.any()), "images": int(df.image.nunique()),
               "queries": {k: int((df.kind == k).sum()) for k in ("object", "part")},
               "models": ident["models"], "settings": ident["settings"], **summarize(df, ch, bs),
               "common_eligible_subset": common_block(cfg, dataset, df, ch, bs, smoke)}
    df.drop(columns=["pred_rle"]).to_csv(run / "queries.csv", index=False)
    ch.to_csv(run / "chains.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(json.dumps({"part_gIoU": metrics["by_kind"]["part"]["gIoU"],
                      "chain": metrics["chains"]["all"]["chain_success_end_to_end"]}, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=B.METHODS)
    ap.add_argument("--dataset", required=True, choices=XT.DATASETS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = 1 if args.smoke else cfg["b_transfer"]["num_shards"]
    if args.merge:
        merge(cfg, args.method, args.dataset, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.method, args.dataset, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
