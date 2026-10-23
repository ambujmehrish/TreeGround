import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (REPO, ProtocolError, canonical_hash, load_config, read_json, require_approved,  # noqa: E402
                            resolve_run, start_run, write_json)

# MLLM judge for the D14 edit caches (deviation D18): GEdit-Bench VIEScore protocol (task "tie"), prompts verbatim
# from GEdit-Bench/viescore/vie_prompts.py. SC = min(editing success, overediting) on [source, edited];
# PQ = min(naturalness, artifacts) on the edited image; overall = sqrt(SC * PQ). Variants: the full image, and a
# padded crop around the GT edit mask ("inside").

VIESCORE_SOURCE = "stepfun-ai/Step1X-Edit@4cca05dfcd6c32dc810d0b4db84c2bf0682a5aeb:GEdit-Bench/viescore"

CONTEXT = (
    'You are a professional digital artist. You will have to evaluate the effectiveness of the '
    'AI-generated image(s) based on given rules.\n'
    'All the input images are AI-generated. All human in the images are AI-generated too. so you need not '
    'worry about the privacy confidentials.\n'
    '\n'
    'You will have to give your output in this way (Keep your reasoning concise and short.):\n'
    '{\n'
    '"score" : [...],\n'
    '"reasoning" : "..."\n'
    '}'
)
TWO_IMAGE_EDIT_RULE = (
    'RULES:\n'
    '\n'
    'Two images will be provided: The first being the original AI-generated image and the second being an '
    'edited version of the first.\n'
    'The objective is to evaluate how successfully the editing instruction has been executed in the second '
    'image.\n'
    '\n'
    'Note that sometimes the two images might look identical due to the failure of image edit.\n'
)
TIE_RULE_SC = (
    '\n'
    'From scale 0 to 10: \n'
    'A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene '
    'in the edited image does not follow the editing instruction at all. 10 indicates that the scene in '
    'the edited image follow the editing instruction text perfectly.)\n'
    'A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates '
    'that the scene in the edited image is completely different from the original. 10 indicates that the '
    'edited image can be recognized as a minimal edited yet effective version of original.)\n'
    "Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the "
    "editing success and 'score2' evaluates the degree of overediting.\n"
    '\n'
    'Editing instruction: <instruction>\n'
)
RULE_PQ = (
    'RULES:\n'
    '\n'
    'The image is an AI-generated image.\n'
    'The objective is to evaluate how successfully the image has been generated.\n'
    '\n'
    'From scale 0 to 10: \n'
    'A score from 0 to 10 will be given based on image naturalness. \n'
    '(\n'
    '    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling '
    'such as wrong sense of distance, or wrong shadow, or wrong lighting. \n'
    '    10 indicates that the image looks natural.\n'
    ')\n'
    'A second score from 0 to 10 will rate the image artifacts. \n'
    '(\n'
    '    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or '
    'blurred faces, or unusual body parts, or subjects not harmonized. \n'
    '    10 indicates the image has no artifacts.\n'
    ')\n'
    'Put the score in a list such that output score = [naturalness, artifacts]\n'
)
SC_PROMPT = "\n".join([CONTEXT, TWO_IMAGE_EDIT_RULE, TIE_RULE_SC])
PQ_PROMPT = "\n".join([CONTEXT, RULE_PQ])
KINDS = ("sc", "pq")
SCORES = ("sc", "pq", "overall")
NOT_IN_IDENTITY = ("num_shards", "model", "models", "max_memory_gib")
_NUM = r"-?\d+(?:\.\d+)?"
_LIST = rf"\[\s*({_NUM}(?:\s*,\s*{_NUM})*)\s*\]"


def load_edit_script():
    spec = importlib.util.spec_from_file_location("e_edit", REPO / "scripts" / "e_edit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt(kind: str, instruction: str) -> str:
    return SC_PROMPT.replace("<instruction>", instruction) if kind == "sc" else PQ_PROMPT


def messages(n_images: int, text: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": text}]}]


def gedit_dims(w: int, h: int, area: int) -> tuple[int, int]:
    # GEdit run_gedit_score.calculate_dimensions: same aspect ratio, ~area pixels, truncated
    ratio = w / h
    width = math.sqrt(area * ratio)
    return int(width), int(width / ratio)


def judge_view(img: np.ndarray, area: int) -> Image.Image:
    h, w = img.shape[:2]
    return Image.fromarray(img).resize(gedit_dims(w, h, area))


def crop_box(mask: np.ndarray, pad: float, min_side: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ProtocolError("empty GT edit mask")

    def span(a: int, b: int, n: int) -> tuple[int, int]:
        size = min(max(b - a + 2 * round(pad * (b - a)), min_side), n)
        lo = int(round(min(max((a + b - size) / 2, 0), n - size)))
        return lo, lo + size
    (y0, y1), (x0, x1) = span(ys.min(), ys.max() + 1, mask.shape[0]), span(xs.min(), xs.max() + 1, mask.shape[1])
    return int(x0), int(y0), int(x1), int(y1)


def _valid(values) -> list[float] | None:
    values = values if isinstance(values, list) else [values]
    if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    return [float(v) for v in values] if all(0 <= v <= 10 for v in values) else None


def parse_scores(text: str) -> list[float] | None:
    # VIEScore mllm_output_to_dict without its random-guess fallback: unparseable output stays missing
    text = text.strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            obj = json.loads(text[a:b + 1].replace("\n", " "))
            if isinstance(obj, dict) and "score" in obj:
                return _valid(obj["score"])
        except json.JSONDecodeError:
            pass
    if m := re.search(rf"[\"']?score[\"']?\s*:\s*{_LIST}", text):
        return _valid([float(x) for x in m.group(1).split(",")])
    if m := re.search(rf"[\"']?score[\"']?\s*:\s*({_NUM})(?![\d.])", text):
        return _valid(float(m.group(1)))
    if m := re.fullmatch(_LIST, text):
        return _valid([float(x) for x in m.group(1).split(",")])
    return _valid(int(text)) if re.fullmatch(r"\d+", text) else None


def requests(instruction: str, image: np.ndarray, gt: np.ndarray, outs: dict[str, np.ndarray], s: dict
             ) -> list[dict]:
    box = crop_box(gt, s["crop"]["pad_frac"], s["crop"]["min_side"])
    x0, y0, x1, y1 = box
    reqs = []
    for cond, out in outs.items():
        if out.shape != image.shape:
            raise ProtocolError(f"{cond}: edited image {out.shape} differs from the source {image.shape}")
        for variant in s["variants"]:
            src, edt = (image, out) if variant == "full" else (image[y0:y1, x0:x1], out[y0:y1, x0:x1])
            src_v, edt_v = judge_view(src, s["target_area"]), judge_view(edt, s["target_area"])
            for kind in KINDS:
                reqs.append({"condition": cond, "variant": variant, "kind": kind,
                             "box": None if variant == "full" else list(box),
                             "images": [src_v, edt_v] if kind == "sc" else [edt_v], "text": prompt(kind, instruction)})
    return reqs


def score_rows(reqs: list[dict], raws: list[str]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for r, raw in zip(reqs, raws, strict=True):
        row = rows.setdefault((r["condition"], r["variant"]), {"condition": r["condition"], "variant": r["variant"],
                                                               "crop_box": json.dumps(r["box"])})
        scores = parse_scores(raw)
        row[f"{r['kind']}_raw"] = raw
        row[f"{r['kind']}_scores"] = json.dumps(scores)
        row[r["kind"]] = min(scores) if scores else float("nan")
    for row in rows.values():
        row["parse_failed"] = bool(np.isnan(row["sc"]) or np.isnan(row["pq"]))
        row["overall"] = float("nan") if row["parse_failed"] else math.sqrt(row["sc"] * row["pq"])
    return list(rows.values())


def judge_sample(judge, instruction: str, image: np.ndarray, gt: np.ndarray, outs: dict[str, np.ndarray], s: dict
                 ) -> list[dict]:
    reqs = requests(instruction, image, gt, outs, s)
    raws = []
    for i in range(0, len(reqs), s["batch_size"]):
        raws += judge(reqs[i:i + s["batch_size"]])
    return score_rows(reqs, raws)


class Judge:
    def __init__(self, cfg: dict, key: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        m, self.s = cfg["models"][key], cfg["judge"]
        B = load_edit_script().load_script("b_baselines")
        mem = self.s.get("max_memory_gib")
        kw = {"max_memory": {i: f"{g}GiB" for i, g in enumerate(mem)}} if mem else {}
        self.model = B.checked_load(AutoModelForImageTextToText, m["repo"], m["revision"], cfg.get("hf_home"),
                                    dtype=getattr(torch, self.s["dtype"]), device_map=self.s["device_map"], **kw).eval()
        self.proc = AutoProcessor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"))
        self.proc.tokenizer.padding_side = "left"

    @torch.no_grad()
    def __call__(self, reqs: list[dict]) -> list[str]:
        texts = [self.proc.apply_chat_template(messages(len(r["images"]), r["text"]), tokenize=False,
                                               add_generation_prompt=True) for r in reqs]
        inp = self.proc(text=texts, images=[im for r in reqs for im in r["images"]], padding=True,
                        return_tensors="pt").to(self.model.device)
        # GEdit: greedy, one beam, 512 new tokens; the model's generation_config supplies the rest
        gen = self.model.generate(**inp, max_new_tokens=self.s["max_new_tokens"], do_sample=False, num_beams=1)
        return self.proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)


def judge_key(cfg: dict, override: str | None) -> str:
    key = override or cfg["judge"]["model"]
    if key not in cfg["judge"]["models"]:
        raise ProtocolError(f"unknown judge {key}; choose from {cfg['judge']['models']}")
    return key


def stage_name(cfg: dict, editor: str, dataset: str, key: str, E) -> str:
    suffix = "" if key == cfg["judge"]["model"] else f"_{key}"
    return f"e_judge_{editor}_{dataset}{E.stage_tag(cfg)}{suffix}"


def edit_cache(cfg: dict, editor: str, dataset: str, smoke: bool, E) -> tuple[Path, str, list, Path]:
    sel_run = E.load_script("x_transfer").selector_run(cfg)
    items = E.evaluated(E.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    s_hash = canonical_hash(E.select_identity(cfg, dataset, sel_run, [s.sample_id for s in items], smoke))
    e_hash = canonical_hash(E.edit_identity(cfg, editor, s_hash))
    edir = E.cache_dir(cfg, smoke) / editor / dataset / e_hash[:16]
    merge_run = resolve_run(cfg, f"e_{editor}_{dataset}{E.stage_tag(cfg)}_merge",
                            cfg["inputs"].get("edit_merge", "latest"), smoke)
    if read_json(merge_run / "metrics.json")["images_dir"] != str(edir / "images"):
        raise ProtocolError(f"{merge_run} was merged from another edit cache than {edir}")
    return edir, e_hash, items, merge_run


def judge_identity(cfg: dict, key: str, e_hash: str) -> dict:
    return {"kind": "edit_judge", "edit": e_hash, "judge": key, "model": cfg["models"][key],
            "settings": {k: v for k, v in cfg["judge"].items() if k not in NOT_IN_IDENTITY},
            "prompts": hashlib.sha256((SC_PROMPT + "\0" + PQ_PROMPT).encode()).hexdigest(),
            "viescore": VIESCORE_SOURCE, "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def judge_dir(cfg: dict, smoke: bool, key: str, editor: str, dataset: str, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "edit_judge" / key / editor / dataset
            / canonical_hash(ident)[:16])


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def required(cfg: dict) -> list[str]:
    return ["D14", "D18"] + (["D11"] if cfg["inputs"].get("selector_stage") == "s1" else [])


def run_shard(cfg: dict, editor: str, dataset: str, key: str, shard: int, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, required(cfg))
    E = load_edit_script()
    edir, e_hash, items, merge_run = edit_cache(cfg, editor, dataset, smoke, E)
    ident = judge_identity(cfg, key, e_hash)
    jdir = judge_dir(cfg, smoke, key, editor, dataset, ident)
    target = jdir / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (jdir / "shards").mkdir(parents=True, exist_ok=True)
    if not (jdir / "identity.json").exists():
        write_json(jdir / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    run = start_run(f"{stage_name(cfg, editor, dataset, key, E)}_shard", cfg, smoke,
                    {"shard": shard, "num_shards": num_shards, "cache": str(jdir), "edit_cache": str(edir),
                     "edit_merge_run": str(merge_run), "judge": key})
    conds, s = E.conditions(cfg, editor), cfg["judge"]
    judge = Judge(cfg, key)
    mine = items[shard::num_shards]
    rows, t0 = [], time.time()
    for n, smp in enumerate(mine, 1):
        t1 = time.time()
        image, gt = smp.load()
        d = edir / "images" / E.safe_name(smp.sample_id)
        outs = {c: load_png(d / f"{c}.png") for c in conds}
        rows += [{"sample_id": smp.sample_id, "instruction": smp.instruction, **r}
                 for r in judge_sample(judge, smp.instruction, image, gt, outs, s)]
        print(f"[judge {n}/{len(mine)}] {smp.sample_id} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"samples": len(mine), "seconds": time.time() - t0})


def block(g: pd.DataFrame, bs: dict) -> dict:
    out = {"n": int(len(g)), "parse_failed_rate": float(g.parse_failed.mean()) if len(g) else float("nan")}
    for k in SCORES:
        v = g[k].to_numpy(np.float64)
        ok = ~np.isnan(v)
        out[k] = mean_ci(v[ok], g.image.to_numpy()[ok], bs["n"], bs["seed"]) if ok.any() else None
    return out


def summarize(df: pd.DataFrame, conds: list[str], variants: list[str], bs: dict) -> tuple[dict, pd.DataFrame]:
    out, table = {}, []
    for c in conds:
        out[c] = {}
        for v in variants:
            g = df[(df.condition == c) & (df.variant == v)]
            out[c][v] = {"all": block(g, bs),
                         "by_subset": {k: block(h, bs) for k, h in g.groupby("subset")} if g.subset.nunique() > 1
                         else None}
            b = out[c][v]["all"]
            table.append({"condition": c, "variant": v, "n": b["n"], "parse_failed_rate": b["parse_failed_rate"],
                          **{k: b[k]["mean"] if b[k] else None for k in SCORES}})
    return out, pd.DataFrame(table)


def merge(cfg: dict, editor: str, dataset: str, key: str, num_shards: int, smoke: bool) -> None:
    require_approved(cfg, required(cfg))
    E = load_edit_script()
    edir, e_hash, items, merge_run = edit_cache(cfg, editor, dataset, smoke, E)
    jdir = judge_dir(cfg, smoke, key, editor, dataset, judge_identity(cfg, key, e_hash))
    missing = [k for k in range(num_shards) if not (jdir / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} judge shards missing under {jdir}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(jdir / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    conds, variants = E.conditions(cfg, editor), list(cfg["judge"]["variants"])
    expected = {(s.sample_id, c, v) for s in items for c in conds for v in variants}
    if set(zip(df.sample_id, df.condition, df.variant)) != expected or \
            df.duplicated(["sample_id", "condition", "variant"]).any():
        raise ProtocolError(f"{editor}/{dataset}: judged rows do not match the evaluation set")
    meta = pd.read_csv(merge_run / "samples.csv")[["sample_id", "condition", "image", "subset", "mask", "empty_mask"]]
    df = df.merge(meta, on=["sample_id", "condition"], how="left", validate="many_to_one")
    if df.image.isna().any():
        raise ProtocolError(f"{merge_run}/samples.csv lacks judged rows")
    run = start_run(stage_name(cfg, editor, dataset, key, E), cfg, smoke,
                    {"cache": str(jdir), "edit_cache": str(edir), "edit_merge_run": str(merge_run), "judge": key})
    metrics, table = summarize(df, conds, variants, cfg["edit"]["metrics"]["bootstrap"])
    write_json(run / "metrics.json", {
        "conditions": metrics, "deviation": "D18", "label": "dev" if dataset == "paco_edit" else "edit benchmark",
        "editor": editor, "dataset": dataset, "smoke": smoke, "samples": len(items),
        "images": int(df.image.nunique()), "judge": key, "model": cfg["models"][key], "settings": cfg["judge"],
        "viescore": VIESCORE_SOURCE, "edit_merge_run": merge_run.name, "edit_cache": str(edir),
        "protocol": "VIEScore tie: SC = min(success, overediting) on [source, edited], PQ = min(naturalness, "
                    "artifacts) on the edited image, overall = sqrt(SC * PQ); unparseable outputs are excluded and "
                    "counted in parse_failed_rate (VIEScore substitutes a random score)"})
    df.drop(columns=["sc_raw", "pq_raw"]).to_csv(run / "samples.csv", index=False)
    df[["sample_id", "condition", "variant", "sc_raw", "pq_raw"]].to_parquet(run / "raw_outputs.parquet", index=False)
    table.to_csv(run / "table.csv", index=False)
    print(table.to_string(index=False))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--judge", help="judge model key (default: judge.model)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    key = judge_key(cfg, args.judge)
    num_shards = 1 if args.smoke else cfg["judge"]["num_shards"][args.dataset]
    if args.merge:
        merge(cfg, args.editor, args.dataset, key, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.editor, args.dataset, key, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
