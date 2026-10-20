import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import mllm  # noqa: E402
from treeground.cache import load_tensors, mark_complete, save_tensors, write_identity  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, query_identity, qwen_static  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            start_run, write_json)

TEXT_ONLY = "textonly"


def build_queries(cfg: dict, wl: pd.DataFrame, mrun: Path) -> pd.DataFrame:
    rows = []
    labels = set()
    for split, role in (("train", "paco_train"), ("val", "paco_val")):
        keys = wl[wl.role == role]
        if not len(keys):
            continue
        images = paco.load(cfg["data"]["paco_lvis"], split)
        for row in keys.sort_values("image_key").itertuples():
            img = images[int(row.image_key.split(":")[1])]
            labels.update(p.label for p in img.parts)
            for q in paco.i1_queries(img)[0]:
                rows.append({"qid": q.qid, "kind": q.kind, "text": q.text, "n_sel": q.n_sel, "gt": json.dumps(q.gt),
                             "label": None, "image_key": row.image_key, "sha1": row.sha1, "split": split,
                             "provenance": f"paco_lvis/{split}/I1"})
    for label in sorted(labels):
        q = paco.text_query(label)
        rows.append({"qid": q.qid, "kind": "text", "text": q.text, "n_sel": 1, "gt": "[]", "label": label,
                     "image_key": None, "sha1": TEXT_ONLY, "split": "any", "provenance": "paco_lvis/text_only"})
    gref = pd.read_parquet(mrun / "grefcoco_queries.parquet")
    for g in gref.itertuples():
        if pd.isna(g.sha1):
            raise ProtocolError(f"gRefCOCO query {g.qid} has no image in the manifest")
        rows.append({"qid": g.qid, "kind": "absent", "text": f"{g.text} {mllm.SEL}", "n_sel": 1, "gt": "[]",
                     "label": None, "image_key": g.image_key, "sha1": g.sha1, "split": g.split,
                     "provenance": f"{g.provenance}/sent_id={g.sent_id}"})
    df = pd.DataFrame(rows).sort_values(["sha1", "qid"]).reset_index(drop=True)
    if df["qid"].duplicated().any():
        raise ProtocolError("duplicate query ids")
    df["offset"] = df.groupby("sha1")["n_sel"].cumsum() - df["n_sel"]
    return df


def query_set_hash(df: pd.DataFrame) -> str:
    return canonical_hash(df[["qid", "text", "sha1", "split", "provenance"]].values.tolist())


def shard_groups(df: pd.DataFrame, shard: int, num_shards: int) -> list[str]:
    groups = sorted(df["sha1"].unique())
    return [g for i, g in enumerate(groups) if i % num_shards == shard]


def find_cache(cfg: dict, smoke: bool, mhash: str, qhash: str) -> Path | None:
    root = Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "queries"
    for d in sorted(root.glob("*")) if root.exists() else []:
        f = d / "identity.json"
        if f.exists():
            ident = read_json(f)["identity"]
            if ident["manifest"] == mhash and ident["query_set"] == qhash:
                return d
    return None


def run_shard(cfg: dict, smoke: bool, shard: int, num_shards: int) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    df = build_queries(cfg, wl, mrun)
    qhash = query_set_hash(df)
    q = cfg["models"]["qwen"]
    model, processor = mllm.load_qwen(q["repo"], q["revision"], cfg["device"], cfg["features"]["dtype"],
                                      cfg.get("hf_home"))
    sel_id, sel_hash = mllm.add_sel_token(model, processor)
    static = qwen_static(cfg)
    layers = static["layers"]
    mllm.check_layers(model, layers)
    if mllm.chat_template_hash(processor) != static["chat_template"]:
        raise ProtocolError("chat template differs from the pinned processor")
    ident = query_identity(cfg, smoke, mhash, static, sel_hash, qhash)
    out_dir = Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "queries" / canonical_hash(ident)[:16]
    write_identity(out_dir, ident)
    df.to_parquet(out_dir / f"queries.shard{shard:05d}.parquet", index=False)
    run = start_run("s0_queries", cfg, smoke, {"shard": shard, "num_shards": num_shards, "cache": str(out_dir),
                                                "manifest_run": str(mrun), "sel_token_id": sel_id})
    (out_dir / "probes").mkdir(parents=True, exist_ok=True)
    ihash = canonical_hash(ident)
    paths = dict(zip(wl["sha1"], wl["path"]))
    t0, done = time.time(), 0
    groups = shard_groups(df, shard, num_shards)
    print(f"shard {shard}/{num_shards}: {len(groups)} image groups -> {out_dir}", flush=True)
    for i, sha1 in enumerate(groups, 1):
        target = out_dir / "probes" / f"{sha1}.safetensors"
        if target.exists() and load_tensors(target, [])[1].get("identity") == ihash:
            continue
        sub = df[df.sha1 == sha1]
        image = None if sha1 == TEXT_ONLY else Image.fromarray(load_rgb(assert_real_path(paths[sha1])))
        states = mllm.probe_states(model, processor, image, sub["text"].tolist(), layers, sel_id,
                                   cfg["queries"]["batch_size"])
        probes = np.concatenate(states, 0).astype(np.float16)
        if probes.shape[0] != int(sub["n_sel"].sum()):
            raise ProtocolError(f"probe count mismatch for {sha1}")
        save_tensors(target, {"probes": probes}, {"image_sha1": sha1, "identity": ihash,
                                                  "layers": json.dumps(layers)})
        done += 1
        print(f"[{i}/{len(groups)}] {sha1[:12]} queries={len(sub)} probes={probes.shape[0]} "
              f"{time.time() - t0:.0f}s elapsed", flush=True)
    write_json(run / "metrics.json", {"groups": done, "seconds": time.time() - t0})


def merge(cfg: dict, smoke: bool) -> None:
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    df = build_queries(cfg, wl, mrun)
    qhash = query_set_hash(df)
    out_dir = find_cache(cfg, smoke, mhash, qhash)
    if out_dir is None:
        raise ProtocolError("no query cache matches the current manifest and query set")
    ident = read_json(out_dir / "identity.json")["identity"]
    expected = query_identity(cfg, smoke, mhash, qwen_static(cfg), ident["sel_init"], qhash)
    write_identity(out_dir, expected)
    run = start_run("s0_queries_merge", cfg, smoke, {"cache": str(out_dir)})
    ihash = canonical_hash(expected)
    for sha1, sub in df.groupby("sha1"):
        f = out_dir / "probes" / f"{sha1}.safetensors"
        if not f.exists():
            raise ProtocolError(f"missing probe states for {sha1}")
        tensors, meta = load_tensors(f)
        if meta.get("identity") != ihash or tensors["probes"].shape[0] != int(sub["n_sel"].sum()):
            raise ProtocolError(f"{f} does not match its queries")
    df.to_parquet(out_dir / "queries.parquet", index=False)
    for f in out_dir.glob("queries.shard*.parquet"):
        f.unlink()
    summary = {"queries": len(df), "by_kind": df.groupby("kind").size().to_dict(),
               "by_split": df.groupby("split").size().to_dict(), "query_set": qhash}
    mark_complete(out_dir, summary)
    write_json(run / "metrics.json", summary)
    print(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = args.num_shards or (1 if args.smoke else cfg["queries"]["num_shards"])
    if args.merge:
        merge(cfg, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.smoke, args.shard, num_shards)


if __name__ == "__main__":
    main()
