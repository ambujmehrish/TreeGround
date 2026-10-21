import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import mllm  # noqa: E402
from treeground.cache import (cache_dir, load_tensors, mark_complete, save_tensors, write_identity)  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.features import image_features, load_siglip  # noqa: E402
from treeground.identity import feature_identity, manifest_inputs, open_proposals, qwen_static  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            start_run, write_json)

COS_MIN = 0.999
REL_FRO_MAX = 1e-2


def feature_worklist(cfg: dict, wl: pd.DataFrame) -> pd.DataFrame:
    return wl[wl.role.isin(cfg["features"]["roles"])]


def shard_items(wl: pd.DataFrame, shard: int, num_shards: int) -> pd.DataFrame:
    unique = wl.drop_duplicates("sha1").sort_values("sha1").reset_index(drop=True)
    return unique.iloc[shard::num_shards]


def load_models(cfg: dict):
    q = cfg["models"]["qwen"]
    model, processor = mllm.load_qwen(q["repo"], q["revision"], cfg["device"], cfg["features"]["dtype"],
                                      cfg.get("hf_home"))
    layers = mllm.candidate_layers(model.config.text_config.num_hidden_layers)
    mllm.check_layers(model, layers)
    siglip = None
    if "siglip" in cfg["features"]["feature_types"]:
        s = cfg["models"]["siglip"]
        siglip = load_siglip(s["repo"], s["revision"], cfg["device"], cfg["features"]["dtype"], cfg.get("hf_home"))
    return model, processor, layers, siglip


def invariance_check(cfg: dict, smoke: bool) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    static = qwen_static(cfg)
    ident = feature_identity(cfg, smoke, mhash, static)
    out_dir = cache_dir(cfg, "features", ident)
    write_identity(out_dir, ident)
    run = start_run("s0_features_check", cfg, smoke, {"manifest_run": str(mrun), "cache": str(out_dir)})
    model, processor, layers, _ = load_models(cfg)
    val = paco.load(cfg["data"]["paco_lvis"], "val")
    keys = wl[wl.role == "paco_val"].sort_values("image_key")
    results = []
    for row in keys.itertuples():
        img = val[int(row.image_key.split(":")[1])]
        queries, _ = paco.i1_queries(img)
        if not queries:
            continue
        pil = Image.fromarray(load_rgb(assert_real_path(row.path)))
        a = mllm.image_pass(model, processor, pil, layers, False)
        prompt = mllm.query_prompt(processor, queries[0].text, True)
        b = mllm.image_pass(model, processor, pil, layers, False, text=prompt)
        for k in layers:
            x, y = a["hidden"][k].float(), b["hidden"][k].float()
            cos = torch.nn.functional.cosine_similarity(x, y, dim=-1).min().item()
            rel = ((x - y).norm() / x.norm()).item()
            results.append({"image_key": row.image_key, "layer": k, "min_cos": cos, "rel_fro": rel,
                            "max_abs": (x - y).abs().max().item(), "passed": cos >= COS_MIN and rel <= REL_FRO_MAX})
        if len({r["image_key"] for r in results}) >= cfg["features"]["invariance_check_images"]:
            break
    df = pd.DataFrame(results)
    df.to_csv(run / "invariance.csv", index=False)
    passed = bool(len(df)) and bool(df["passed"].all())
    summary = {"passed": passed, "images": int(df["image_key"].nunique()) if len(df) else 0,
               "min_cos": float(df["min_cos"].min()) if len(df) else None,
               "max_rel_fro": float(df["rel_fro"].max()) if len(df) else None,
               "tolerance": {"min_cos": COS_MIN, "max_rel_fro": REL_FRO_MAX}, "run": str(run)}
    write_json(run / "metrics.json", summary)
    write_json(out_dir / "invariance.json", summary)
    if not passed:
        raise ProtocolError(f"image-token invariance check failed: {summary}")


def run_shard(cfg: dict, smoke: bool, shard: int, num_shards: int) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    static = qwen_static(cfg)
    ident = feature_identity(cfg, smoke, mhash, static)
    out_dir = cache_dir(cfg, "features", ident)
    check = out_dir / "invariance.json"
    if not check.exists() or not read_json(check)["passed"]:
        raise ProtocolError("run the image-token invariance check (--check) before extracting features")
    props = open_proposals(cfg, smoke, manifest, mhash, capped=True)
    run = start_run("s0_features", cfg, smoke, {"shard": shard, "num_shards": num_shards, "cache": str(out_dir),
                                                 "manifest_run": str(mrun), "manifest_sha256": mhash})
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    model, processor, layers, siglip = load_models(cfg)
    if layers != static["layers"]:
        raise ProtocolError("candidate layers differ between config and loaded model")
    ihash = canonical_hash(ident)
    done = skipped = 0
    t0 = time.time()
    items = shard_items(feature_worklist(cfg, wl), shard, num_shards)
    print(f"shard {shard}/{num_shards}: {len(items)} images -> {out_dir}", flush=True)
    for i, row in enumerate(items.itertuples(), 1):
        target = out_dir / "images" / f"{row.sha1}.safetensors"
        if target.exists() and load_tensors(target, [])[1].get("identity") == ihash:
            skipped += 1
            continue
        image = load_rgb(assert_real_path(row.path))
        nodes = props.nodes(row.sha1, cfg["proposals"]["node_dedup"])
        if (nodes["h"], nodes["w"]) != image.shape[:2]:
            raise ProtocolError(f"{row.path}: proposal masks do not match the image size")
        tensors = image_features(model, processor, image, nodes["rles"], nodes["boxes"], layers,
                                 cfg["features"]["feature_types"], siglip, cfg["features"]["siglip_batch"])
        save_tensors(target, tensors, {"image_sha1": row.sha1, "identity": ihash, "n_nodes": str(len(nodes["rles"]))})
        done += 1
        print(f"[{i}/{len(items)}] {row.image_key} nodes={len(nodes['rles'])} grid={tuple(tensors['grid'].tolist())} "
              f"{time.time() - t0:.0f}s elapsed", flush=True)
    write_json(run / "metrics.json", {"images": done, "skipped": skipped, "seconds": time.time() - t0})


def merge(cfg: dict, smoke: bool) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    static = qwen_static(cfg)
    ident = feature_identity(cfg, smoke, mhash, static)
    out_dir = cache_dir(cfg, "features", ident)
    run = start_run("s0_features_merge", cfg, smoke, {"cache": str(out_dir)})
    ihash = canonical_hash(ident)
    missing, nodes = [], []
    todo = sorted(feature_worklist(cfg, wl)["sha1"].unique())
    print(f"checking {len(todo)} feature files in {out_dir}", flush=True)
    for sha1 in todo:
        f = out_dir / "images" / f"{sha1}.safetensors"
        if not f.exists():
            missing.append(sha1)
            continue
        meta = load_tensors(f, [])[1]
        if meta.get("identity") != ihash or meta.get("image_sha1") != sha1:
            raise ProtocolError(f"{f} has a foreign identity")
        nodes.append(int(meta["n_nodes"]))
    if missing:
        raise ProtocolError(f"{len(missing)} images lack features, e.g. {missing[:3]}")
    summary = {"images": len(nodes), "nodes_total": int(sum(nodes)), "layers": static["layers"],
               "feature_types": cfg["features"]["feature_types"]}
    mark_complete(out_dir, summary)
    write_json(run / "metrics.json", summary)
    print(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = args.num_shards or (1 if args.smoke else cfg["features"]["num_shards"])
    if args.check:
        invariance_check(cfg, args.smoke)
    elif args.merge:
        merge(cfg, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.smoke, args.shard, num_shards)


if __name__ == "__main__":
    main()
