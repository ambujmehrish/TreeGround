import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import cocoedit_metrics as CM  # noqa: E402
from treeground.data import cocoedit_bench as CB  # noqa: E402
from treeground.run import (REPO, ProtocolError, canonical_hash, load_config, read_json, require_approved,  # noqa: E402
                            resolve_run, start_run, write_json)

# MLLM judge for the D25 edit caches. GEdit-Bench: VIEScore (e_judge.py, D18 prompts verbatim; SC/PQ/O, ~512x512
# inputs, images before the text). ImgEdit-Bench: the per-edit-type 1-5 prompts of PKU-YuanGroup/ImgEdit
# Benchmark/Basic/prompts.json verbatim (text before [source, edited] as in basic_bench.py), sample score = mean of
# the "<name>: <int>" lines rounded to 2 (step1_get_avgscore.py). Identical output images of one sample are judged once.

PROMPTS_FILE = REPO / "treeground" / "data" / "imgedit_prompts.json"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imgedit_prompts() -> dict:
    return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def protocol(dataset: str) -> str:
    return "viescore" if dataset == "gedit_full" else "imgedit"


def gedit_requests(J, instruction: str, image: np.ndarray, out: np.ndarray, area: int, tag: str) -> list[dict]:
    src, edt = J.judge_view(image, area), J.judge_view(out, area)
    return [{"condition": tag, "variant": "full", "kind": k, "box": None, "images": [src, edt] if k == "sc" else [edt],
             "text": J.prompt(k, instruction), "text_first": False} for k in J.KINDS]


def imgedit_request(prompts: dict, s, image: np.ndarray, out: np.ndarray, tag: str) -> dict:
    return {"condition": tag, "images": [Image.fromarray(image), Image.fromarray(out)],
            "text": prompts[s.category].replace("<edit_prompt>", s.instruction), "text_first": True}


def chat(r: dict) -> list[dict]:
    imgs = [{"type": "image"}] * len(r["images"])
    txt = [{"type": "text", "text": r["text"]}]
    return [{"role": "user", "content": txt + imgs if r["text_first"] else imgs + txt}]


def make_judge(cfg: dict, key: str, dataset: str):
    J = load_script("e_judge")

    class Judge(J.Judge):
        def __init__(self):
            super().__init__(cfg, key)
            self.max_new = self.s["max_new_tokens"]
            if dataset == "imgedit_full":
                from transformers import AutoProcessor
                m, ji = cfg["models"][key], cfg["cocoedit"]["judge"]["imgedit"]
                self.proc = AutoProcessor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"),
                                                          min_pixels=ji["min_pixels"], max_pixels=ji["max_pixels"])
                self.proc.tokenizer.padding_side = "left"
                self.max_new = ji["max_new_tokens"]

        @torch.no_grad()
        def __call__(self, reqs: list[dict]) -> list[str]:
            texts = [self.proc.apply_chat_template(chat(r), tokenize=False, add_generation_prompt=True) for r in reqs]
            inp = self.proc(text=texts, images=[im for r in reqs for im in r["images"]], padding=True,
                            return_tensors="pt").to(self.model.device)
            gen = self.model.generate(**inp, max_new_tokens=self.max_new, do_sample=False, num_beams=1)
            return self.proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
    return Judge(), J


def judge_key(cfg: dict, override: str | None) -> str:
    key = override or cfg["cocoedit"]["judge"]["model"]
    if key not in cfg["judge"]["models"]:
        raise ProtocolError(f"unknown judge {key}")
    return key


def stage(cfg: dict, editor: str, dataset: str, key: str) -> str:
    suffix = "" if key == cfg["cocoedit"]["judge"]["model"] else f"_{key}"
    return f"cocoedit_judge_{editor}_{dataset}{suffix}"


def edit_merge(cfg: dict, editor: str, dataset: str, smoke: bool) -> Path:
    return resolve_run(cfg, f"cocoedit_{editor}_{dataset}_merge", cfg["inputs"].get("cocoedit_merge", "latest"), smoke)


def identity(cfg: dict, key: str, dataset: str, merge_run: Path) -> dict:
    J = load_script("e_judge")
    prompts = (J.SC_PROMPT + "\0" + J.PQ_PROMPT) if dataset == "gedit_full" else PROMPTS_FILE.read_text(encoding="utf-8")
    return {"kind": "cocoedit_judge", "judge": key, "model": cfg["models"][key], "dataset": dataset,
            "merge_run": merge_run.name, "protocol": protocol(dataset),
            "settings": {k: v for k, v in cfg["judge"].items() if k not in J.NOT_IN_IDENTITY},
            "cocoedit_judge": {k: v for k, v in cfg["cocoedit"]["judge"].items() if k != "model"},
            "prompts": hashlib.sha256(prompts.encode()).hexdigest(),
            "code": hashlib.sha256(Path(__file__).read_bytes() + (REPO / "scripts/e_judge.py").read_bytes()).hexdigest()}


def judge_dir(cfg: dict, smoke: bool, key: str, editor: str, dataset: str, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "cocoedit_judge" / key / editor / dataset
            / canonical_hash(ident)[:16])


def shard_file(d: Path, shard: int, num_shards: int) -> Path:
    return d / "shards" / f"{shard:05d}of{num_shards:05d}.parquet"


def unique_outputs(paths: dict[str, Path]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c, p in paths.items():
        groups.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(c)
    return {cs[0]: cs for cs in groups.values()}


def judge_sample(judge, J, cfg: dict, dataset: str, s, image: np.ndarray, outs: dict[str, np.ndarray], prompts
                 ) -> list[dict]:
    bs = cfg["judge"]["batch_size"]
    if dataset == "gedit_full":
        reqs = [r for tag, o in outs.items()
                for r in gedit_requests(J, s.instruction, image, o, cfg["cocoedit"]["judge"]["gedit"]["target_area"], tag)]
    else:
        reqs = [imgedit_request(prompts, s, image, o, tag) for tag, o in outs.items()]
    raws = []
    for i in range(0, len(reqs), bs):
        raws += judge(reqs[i:i + bs])
    if dataset == "gedit_full":
        return [{k: r[k] for k in ("condition", "sc", "pq", "overall", "parse_failed", "sc_raw", "pq_raw")}
                for r in J.score_rows(reqs, raws)]
    return [{"condition": r["condition"], "overall": CM.imgedit_score(raw), "raw": raw,
             "parse_failed": math.isnan(CM.imgedit_score(raw))} for r, raw in zip(reqs, raws, strict=True)]


def items_by_id(cfg: dict, dataset: str, smoke: bool) -> dict:
    C = load_script("e_cocoedit")
    items = C.evaluated(C.samples(cfg, dataset), smoke, cfg["cocoedit"]["smoke_per_category"])
    return {s.sample_id: s for s in items}


def run_shard(cfg: dict, editor: str, dataset: str, key: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D25"])
    merge_run = edit_merge(cfg, editor, dataset, smoke)
    ident = identity(cfg, key, dataset, merge_run)
    jdir = judge_dir(cfg, smoke, key, editor, dataset, ident)
    target = shard_file(jdir, shard, num_shards)
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (jdir / "shards").mkdir(parents=True, exist_ok=True)
    if not (jdir / "identity.json").exists():
        write_json(jdir / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    run = start_run(f"{stage(cfg, editor, dataset, key)}_shard", cfg, smoke,
                    {"shard": shard, "num_shards": num_shards, "cache": str(jdir), "merge_run": str(merge_run)})
    m = read_json(merge_run / "metrics.json")
    dirs = {c: Path(d) for c, d in m["images_dirs"].items()}
    C = load_script("e_cocoedit")
    items = sorted(items_by_id(cfg, dataset, smoke).values(), key=lambda s: s.sample_id)
    if len(items) != m["samples"]:
        raise ProtocolError(f"{merge_run} covers {m['samples']} samples, the dataset gives {len(items)}")
    prompts = imgedit_prompts() if dataset == "imgedit_full" else None
    judge, J = make_judge(cfg, key, dataset)
    rows, t0 = [], time.time()
    mine = items[shard::num_shards]
    for n, s in enumerate(mine, 1):
        t1 = time.time()
        image, _ = s.load()
        paths = {c: d / f"{C.safe_name(s.sample_id)}.png" for c, d in dirs.items()}
        groups = unique_outputs(paths)
        outs = {c: C.load_png(paths[c]) for c in groups}
        for r in judge_sample(judge, J, cfg, dataset, s, image, outs, prompts):
            rows += [{"sample_id": s.sample_id, **r, "condition": c, "judged_as": r["condition"]}
                     for c in groups[r["condition"]]]
        print(f"[judge {n}/{len(mine)}] {s.sample_id} {len(groups)}/{len(paths)} unique {time.time() - t1:.1f}s",
              flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"samples": len(mine), "seconds": time.time() - t0})


def summarize(df: pd.DataFrame, dataset: str, conds: list[str]) -> tuple[dict, pd.DataFrame]:
    cats = CB.MLLM_CATEGORIES[dataset]
    ndigits = 2 if dataset == "imgedit_full" else None  # ImgEdit step2_typescore.py rounds category means to 2
    keys = ("sc", "pq", "overall") if dataset == "gedit_full" else ("overall",)
    out, table = {}, []
    for c in conds:
        g = df[df.condition == c]
        block = {"n": int(len(g)), "parse_failed_rate": float(g.parse_failed.mean())}
        for k in keys:
            per_cat = {cat: list(h[k]) for cat, h in g.groupby("category")}
            overall, per = CM.category_overall(per_cat, cats, ndigits)
            block[k] = {"overall": overall, "by_category": per,
                        "all_categories_mean": CM.nanmean(g[k]),
                        "mllm_samples_mean": CM.nanmean(g[g.mllm][k])}
        out[c] = block
        table.append({"condition": c, "n": block["n"], "parse_failed_rate": block["parse_failed_rate"],
                      **{f"{k}_overall": block[k]["overall"] for k in keys}})
    return out, pd.DataFrame(table)


def merge(cfg: dict, editor: str, dataset: str, key: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, ["D25"])
    merge_run = edit_merge(cfg, editor, dataset, smoke)
    jdir = judge_dir(cfg, smoke, key, editor, dataset, identity(cfg, key, dataset, merge_run))
    missing = [k for k in range(num_shards) if not shard_file(jdir, k, num_shards).exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} judge shards missing under {jdir}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(shard_file(jdir, k, num_shards)) for k in range(num_shards)], ignore_index=True)
    meta = pd.read_csv(merge_run / "samples.csv")[["sample_id", "condition", "category", "mllm", "scored", "image"]]
    conds = list(read_json(merge_run / "metrics.json")["images_dirs"])
    if set(zip(df.sample_id, df.condition)) != set(zip(meta.sample_id, meta.condition)) or \
            df.duplicated(["sample_id", "condition"]).any():
        raise ProtocolError(f"{editor}/{dataset}: judged rows do not match {merge_run}")
    df = df.merge(meta, on=["sample_id", "condition"], how="left", validate="one_to_one")
    run = start_run(stage(cfg, editor, dataset, key), cfg, smoke, {"cache": str(jdir), "merge_run": str(merge_run)})
    metrics, table = summarize(df, dataset, conds)
    raw_cols = [c for c in ("raw", "sc_raw", "pq_raw") if c in df]
    df.drop(columns=raw_cols).to_csv(run / "samples.csv", index=False)
    df[["sample_id", "condition", *raw_cols]].to_parquet(run / "raw_outputs.parquet", index=False)
    table.to_csv(run / "table.csv", index=False)
    write_json(run / "metrics.json", {
        "deviation": "D25", "editor": editor, "dataset": dataset, "smoke": smoke, "judge": key,
        "model": cfg["models"][key], "protocol": protocol(dataset), "merge_run": merge_run.name, "conditions": metrics,
        "mllm_categories": list(CB.MLLM_CATEGORIES[dataset]),
        "prompts_source": "e_judge.VIESCORE_SOURCE" if dataset == "gedit_full"
        else cfg["cocoedit"]["judge"]["imgedit_prompts_source"],
        "overall_rule": "unweighted mean of the per-category means over mllm_categories (unparseable outputs "
                        "dropped); ImgEdit category means rounded to 2 decimals as step2_typescore.py"})
    print(table.to_string(index=False))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", required=True)
    ap.add_argument("--dataset", choices=CB.DATASETS, required=True)
    ap.add_argument("--judge", help="judge model key (default: cocoedit.judge.model)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--then-merge", action="store_true", help="merge after the shard (single-shard smoke runs)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    key = judge_key(cfg, args.judge)
    num_shards = 1 if args.smoke else cfg["cocoedit"]["judge_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.editor, args.dataset, key, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.editor, args.dataset, key, args.shard, num_shards, args.smoke)
        if args.then_merge:
            merge(cfg, args.editor, args.dataset, key, num_shards, args.smoke)


if __name__ == "__main__":
    main()
