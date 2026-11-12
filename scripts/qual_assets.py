import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json, resolve_run, runs_root  # noqa: E402

# Read-only qualitative assets (no protocol stage, nothing tuned, no metric reported from here). For each family the
# queries are picked from the finished test runs' per-query IoUs of ours and the strongest same-size baseline
# (Box + SAM 3): a fixed-seed uniform sample ('random'), the largest IoU margins for ours ('ours_better') and against
# ours ('ours_worse'). Masks of the picked queries are recomputed with the exact predictors of those runs and the
# recomputed IoU is stored next to the run's, so a figure can be checked against the table it illustrates.
#   part:   OP-HRG part queries (InstructPart part names, PartImageNet, PascalPart-116), Table 1 predictors
#           (ophrg_eval baselines at their released thresholds; ours = ta_test_eval.OphrgOurs' pick)
#   reason: D50 reasoning queries (InstructPart instructions, ReasonSeg), reason_eval predictors
#   edit:   D25 / D47 CoCoEdit samples: stored masks of every source and the stored edits of each condition
# Output: $TG_ROOT/results/figures/qual_<family>.json (images as small JPEGs, masks as RLE at that size).

FAMILIES = ("part", "reason", "edit")
GROUPS = ("random", "ours_better", "ours_worse")
SIDE = 512
PART_BENCHES = ("instructpart", "partimagenet", "pascalpart")
REASON_BENCHES = ("instructpart_human", "reasonseg")
PART_BASELINES = ("qwen_box_sam", "sam3_text", "sa2va")
REASON_BASELINES = ("qwen_box_sam", "sa2va", "sam3_text_na")
EDIT_CONDITIONS = {"qwen_image_edit": ("free", "gt_blend", "ours_softfam_blend", "qwen_box_sam_blend", "sam3text_blend"),
                   "flux_fill": ("gt_fill", "ours_softfam_fill", "qwen_box_sam_fill", "sam3text_fill")}
EDIT_SOURCES = ("ours_softfam", "qwen_box_sam", "sam3text")   # ours = the Tab. 1 checkpoint (edit_cocoedit_softfam)


# ---- selection ----

def run_ious(cfg: dict, stage: str, kind: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(resolve_run(cfg, stage, "latest", False) / "eval_queries.csv")
    if kind is not None:
        df = df[df.kind == kind]
    return pd.DataFrame({"qid": df.qid, "iou": np.where(df.union > 0, df.inter / df.union.clip(lower=1), 0.0)})


def choose(ours: pd.DataFrame, base: pd.DataFrame, k: int, rng) -> list[dict]:
    # k per group, no query twice; the random group is drawn first so it is not biased by the margin groups
    df = ours.merge(base, on="qid", suffixes=("_ours", "_base"), validate="one_to_one")
    if len(df) != len(ours) or len(df) != len(base):
        raise ProtocolError(f"run query sets differ ({len(ours)} ours, {len(base)} baseline, {len(df)} common)")
    df["margin"] = df.iou_ours - df.iou_base
    picked, out = set(), []
    order = {"random": rng.permutation(len(df)), "ours_better": np.argsort(-df.margin.to_numpy(), kind="stable"),
             "ours_worse": np.argsort(df.margin.to_numpy(), kind="stable")}
    for g in GROUPS:
        n = 0
        for i in order[g]:
            r = df.iloc[int(i)]
            if n == k:
                break
            if r.qid in picked:
                continue
            picked.add(r.qid)
            out.append({"qid": r.qid, "group": g, "run_iou": {"ours": float(r.iou_ours), "base": float(r.iou_base)}})
            n += 1
    return out


# ---- encoding ----

def small(image: np.ndarray, side: int = SIDE) -> np.ndarray:
    f = side / max(image.shape[:2])
    if f >= 1:
        return image
    return np.asarray(Image.fromarray(image).resize((round(image.shape[1] * f), round(image.shape[0] * f)),
                                                    Image.BILINEAR))


def small_mask(mask: np.ndarray | None, hw: tuple[int, int]) -> dict | None:
    if mask is None or not mask.any():
        return None
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.BILINEAR)
    import fig_assets as FA
    return FA.rle_json(M.encode(np.asarray(m) >= 128)[0])


def iou(a: np.ndarray | None, b: np.ndarray) -> float:
    if a is None:
        return 0.0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def jpeg(image: np.ndarray) -> str:
    import fig_assets as FA
    return FA.jpeg_b64(small(image), quality=88)


def free() -> None:
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ---- ours' single-mask pick (ta_test_eval.single_rows' rule) ----

def pick_index(kind: str, lg: np.ndarray, parents, anchor) -> int:
    # an anchored part query is decided object-first from its two-row chain logits; everything else is the argmax
    from treeground import tree_attn as TA
    if kind == "part" and anchor is not None:
        return int(TA.anchor(lg[0], lg[1], parents, TA.anchor_mode(anchor))[1])
    return int(np.argmax(lg[0]))


def ours_part_masks(scorer, image: np.ndarray, key: str, qs: list) -> list[np.ndarray | None]:
    # ta_test_eval.OphrgOurs with the picked mask returned instead of its intersection / union
    import ophrg_eval as OE
    import ta_test_eval as TTE
    g = pd.DataFrame([{"qid": q[0], "kind": q[1], "text": OE.text(q[1], q[2], q[3])} for q in qs])
    queries = TTE.queries_of(g)
    if scorer.anchor is not None:
        queries = [TTE.chain_query(q[0], q[2], q[3]) if q[1] == "part" else qq for q, qq in zip(qs, queries)]
    pools, lgs = scorer.score(image, key, g.text.tolist(), queries)
    return [M.decode(p.rles[pick_index(q[1], lg, p.parents, scorer.anchor)]).astype(bool) if p.n else None
            for q, p, lg in zip(qs, pools, lgs)]


# ---- families ----

def part_family(cfg: dict, bcfg: dict, k: int, rng) -> list[dict]:
    import b_baselines as B
    import b_baselines2  # noqa: F401  (registers the D31 predictors)
    import ophrg_eval as OE
    import ta_test_eval as TTE
    stage = TTE.TAT.stage(cfg)
    want = {}
    for bench in PART_BENCHES:
        for c in choose(run_ious(cfg, f"ophrg_{bench}_{stage}", "part"),
                        run_ious(bcfg, f"ophrg_{bench}_qwen_box_sam", "part"), k, rng):
            want[c["qid"]] = {**c, "benchmark": bench}
    keys = {qid.split("|")[0] for qid in want}   # a query id starts with its image key
    items = []   # (record, image key, query, image)
    for bench in PART_BENCHES:
        for key, path, qs in OE.LOADERS[bench](cfg):
            if key not in keys:
                continue
            image = np.asarray(Image.open(path).convert("RGB"))
            for q in OE.resolve(key, qs, *image.shape[:2]):
                if q[0] in want:
                    items.append((want[q[0]], key, q, image))
    if len(items) != len(want):
        raise ProtocolError(f"found {len(items)} of {len(want)} picked queries")
    preds = {}
    s = TTE.Scorer(cfg)
    for rec, key, q, image in items:
        preds[(rec["qid"], "ours")] = (ours_part_masks(s, image, key, [q])[0], True)
    del s
    free()
    for m in PART_BASELINES:
        pred = B.PREDICTORS[m](bcfg)
        for rec, key, q, image in items:
            mask, _, _, acc = pred(image, [B.phrase(OE.text(q[1], q[2], q[3]))])[0]
            preds[(rec["qid"], m)] = (None if mask is None else mask.astype(bool), bool(acc))
        del pred
        free()
    out = []
    for rec, key, q, image in items:
        gt, hw = q[4], small(image).shape[:2]
        meth = {}
        for m in ("ours", *PART_BASELINES):
            mask, acc = preds[(rec["qid"], m)]
            shown = mask if acc else None   # Table 1 scores a mask below the released threshold as empty
            meth[m] = {"mask": small_mask(mask, hw), "accepted": acc, "iou": iou(shown, gt), "iou_any": iou(mask, gt)}
        out.append({**rec, "query": OE.text(q[1], q[2], q[3]), "image": jpeg(image), "gt": small_mask(gt, hw),
                    "methods": meth})
    return out


def reason_family(cfg: dict, bcfg: dict, k: int, rng) -> list[dict]:
    import reason_eval as RE
    stage = RE.stage_name(cfg, "ours")
    want, recs = {}, {}
    for bench in REASON_BENCHES:
        for c in choose(run_ious(cfg, f"reason_{bench}_{stage}"), run_ious(bcfg, f"reason_{bench}_qwen_box_sam"),
                        k, rng):
            want[c["qid"]] = {**c, "benchmark": bench}
        recs.update({r[0]: r for r in RE.records(cfg, bench) if r[0] in want})
    if len(recs) != len(want):
        raise ProtocolError(f"found {len(recs)} of {len(want)} picked queries")
    preds = {}
    for m in ("ours", *REASON_BASELINES):
        p = RE.Ours(cfg) if m == "ours" else RE.Baseline(bcfg, m)
        for qid, key, text, load_image, _ in recs.values():
            preds[(qid, m)] = p(load_image(), key, text)
        del p
        free()
    out = []
    for qid, rec in want.items():
        _, key, text, load_image, load_gt = recs[qid]
        image, gt = load_image(), load_gt()
        hw = small(image).shape[:2]
        meth = {m: {"mask": small_mask(preds[(qid, m)], hw), "iou": iou(preds[(qid, m)], gt)}
                for m in ("ours", *REASON_BASELINES)}
        out.append({**rec, "query": text, "image": jpeg(image), "gt": small_mask(gt, hw), "methods": meth})
    return out


def merge_runs(cfg: dict, editor: str, dataset: str) -> dict:
    # the D25 and D47 merges share a stage name; per condition / source the latest completed run that has it
    dirs = {"images": {}, "select": {}}
    for run in sorted((runs_root(cfg) / f"cocoedit_{editor}_{dataset}_merge").glob("*")):
        if not (run / "metrics.json").exists():
            continue
        m = read_json(run / "metrics.json")
        dirs["images"].update(m["images_dirs"])
        dirs["select"].update(m["select_dirs"])
    return dirs


def select_rows(d: str) -> pd.DataFrame:
    files = sorted((Path(d) / "shards").glob("*.parquet"))
    if not files:
        raise ProtocolError(f"no select shards in {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).set_index("sample_id")


def edit_family(cfg: dict, k: int, rng) -> list[dict]:
    import e_cocoedit as E
    out = []
    for dataset in E.CB.DATASETS:
        items = {s.sample_id: s for s in E.evaluated(E.samples(cfg, dataset), False, 0) if s.localized}
        dirs = {e: merge_runs(cfg, e, dataset) for e in EDIT_CONDITIONS}
        sel = {src: select_rows(dirs["qwen_image_edit"]["select"][src]) for src in EDIT_SOURCES}
        ids = sorted(items)
        ious = {src: pd.DataFrame({"qid": ids, "iou": sel[src].loc[ids, "iou"].to_numpy()}) for src in EDIT_SOURCES}
        for c in choose(ious["ours_softfam"], ious["qwen_box_sam"], k, rng):
            s = items[c["qid"]]
            image, gt = s.load()
            hw = small(image).shape[:2]
            masks = {}
            for src in EDIT_SOURCES:
                r = json.loads(sel[src].at[s.sample_id, "rle"])
                masks[src] = small_mask(M.decode(r).astype(bool) if r else None, hw)
            edits = {}
            for editor, conds in EDIT_CONDITIONS.items():
                for cond in conds:
                    p = Path(dirs[editor]["images"][cond]) / f"{E.safe_name(s.sample_id)}.png"
                    with Image.open(p) as im:
                        edits[f"{editor}/{cond}"] = jpeg(np.asarray(im.convert("RGB")))
            out.append({**c, "benchmark": dataset, "query": s.instruction, "referent": s.referent,
                        "image": jpeg(image), "gt": small_mask(gt, hw), "masks": masks,
                        "mask_iou": {src: float(sel[src].at[s.sample_id, "iou"]) for src in EDIT_SOURCES},
                        "edits": edits})
            print(f"edit {dataset} {c['group']} {s.sample_id}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("--config", required=True, help="ours (part, reason) or the CoCoEdit config (edit)")
    ap.add_argument("--baseline_config", help="the baselines' config (part, reason)")
    ap.add_argument("--per_group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    if args.family == "edit":
        rows = edit_family(cfg, args.per_group, rng)
    else:
        if not args.baseline_config:
            raise ProtocolError(f"{args.family} needs --baseline_config")
        bcfg = load_config(args.baseline_config)
        fam = part_family if args.family == "part" else reason_family
        rows = fam(cfg, bcfg, args.per_group, rng)
    dest = Path(os.environ["TG_ROOT"]) / "results" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"qual_{args.family}.json"
    path.write_text(json.dumps({"family": args.family, "config": args.config, "baseline_config": args.baseline_config,
                                "seed": args.seed, "per_group": args.per_group, "groups": GROUPS, "side": SIDE,
                                "samples": rows}))
    print(path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
