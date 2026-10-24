import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_eval as S2E  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from b_baselines import ratio_ci  # noqa: E402
from v1_ceiling import interior_point  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import (manifest_inputs, node_cap, open_features, open_proposals, open_queries,  # noqa: E402
                                 qwen_static)
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            require_approved, resolve_run, runs_root, start_run, v1_gate, write_json)
from treeground.selector import Selector  # noqa: E402

# Selection + re-prompt refinement (deviation D19): the selector's top node is re-prompted to the SAM 3 tracker of the
# proposals and the refined mask replaces it. Accept decisions are unchanged. Variants: r0 the node itself; r1 node
# box; r2 box + positive point at the node's interior point; r3 box + node mask as dense prompt; r4 r2 + negative
# points inside the node's tree parent but outside the node. Output mask: the multimask output with the highest
# predicted IoU (first on ties).

STAGES = ("v2", "s1", "s2", "s2_from_s1")
VARIANTS = ("r1", "r2", "r3", "r4")
THR = 0.5
SINGLE_LEVEL = V4.SINGLE_LEVEL
MODES = {"cal": "calibrated", "nr": "no_rejection"}


@dataclass
class Prompt:
    box: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    mask: np.ndarray | None = None


def mask_box(mask: np.ndarray) -> np.ndarray:
    # same convention as masks.boxes_xyxy: [x0, y0, x0 + w, y0 + h] in pixels
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def negative_points(node: np.ndarray, parent: np.ndarray | None, k: int, min_dist: float) -> np.ndarray:
    # greedy: interior point of parent \ node, then a disk of its clearance is removed and the next one is taken
    if parent is None or k <= 0:
        return np.zeros((0, 2))
    region = parent & ~node
    yy, xx = np.ogrid[:region.shape[0], :region.shape[1]]
    pts = []
    while len(pts) < k and region.any():
        clearance = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1].max()
        if clearance < min_dist:
            break
        x, y = interior_point(region)
        pts.append((x, y))
        region = region & ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > clearance ** 2)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def make_prompt(variant: str, node: np.ndarray, parent: np.ndarray | None, rcfg: dict) -> Prompt:
    p = Prompt(mask_box(node))
    if variant in ("r2", "r4"):
        pos = np.array([interior_point(node)])
        neg = negative_points(node, parent, rcfg["negatives"]["max_points"], rcfg["negatives"]["min_distance"]) \
            if variant == "r4" else np.zeros((0, 2))
        p.points = np.concatenate([pos, neg])
        p.labels = np.array([1] + [0] * len(neg), dtype=np.int64)
    if variant == "r3":
        p.mask = node
    return p


def mask_prompt(mask: np.ndarray, size: tuple[int, int], scale: float, bias: float) -> torch.Tensor:
    # binary mask -> low-res logits in the processor's square frame (image resized without keeping the aspect ratio)
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    m = F.interpolate(m, size=size, mode="bilinear", align_corners=False, antialias=True)
    return m * scale + bias


def choose(iou_scores: torch.Tensor) -> torch.Tensor:
    return iou_scores.argmax(-1)


@torch.no_grad()
def run_prompts(model, post, emb: list[torch.Tensor], prompts: list[Prompt], hw: tuple[int, int], target: int,
                rcfg: dict, mask_threshold: float, dtype) -> list[tuple[np.ndarray, float]]:
    # prompts with the same number of points (and with or without a mask) form one image batch; nothing is padded
    h, w = hw
    device = emb[-1].device
    scale = np.array([target / w, target / h])
    groups: dict[tuple[int, bool], list[int]] = {}
    for i, p in enumerate(prompts):
        groups.setdefault((len(p.points), p.mask is not None), []).append(i)
    out: list = [None] * len(prompts)
    for (n_pts, has_mask), idx in sorted(groups.items()):
        for s in range(0, len(idx), rcfg["batch"]):
            b = idx[s:s + rcfg["batch"]]
            n = len(b)
            boxes = np.stack([prompts[i].box for i in b]) * np.tile(scale, 2)
            kw = {"input_boxes": torch.tensor(boxes, dtype=dtype, device=device)[:, None]}
            if n_pts:
                pts = np.stack([prompts[i].points for i in b]) * scale
                kw["input_points"] = torch.tensor(pts, dtype=dtype, device=device)[:, None]
                kw["input_labels"] = torch.tensor(np.stack([prompts[i].labels for i in b]), device=device)[:, None]
            if has_mask:
                mp = rcfg["mask_prompt"]
                kw["input_masks"] = torch.cat([mask_prompt(prompts[i].mask, model.prompt_encoder.mask_input_size,
                                                           mp["scale"], mp["bias"]) for i in b]).to(device, dtype)
            res = model(image_embeddings=[e.repeat(n, *[1] * (e.dim() - 1)) for e in emb], multimask_output=True,
                        **kw)
            scores = res.iou_scores[:, 0].float()
            k = choose(scores)
            chosen = res.pred_masks[torch.arange(n, device=k.device), 0, k]
            logits = post([chosen[:, None].float()], [[h, w]], binarize=False)[0][:, 0]
            binary = (logits > mask_threshold).cpu().numpy()
            for j, i in enumerate(b):
                out[i] = (binary[j], float(scores[j, k[j]]))
    return out


# ---- selections -----------------------------------------------------------------------------------------------

@dataclass
class Source:
    stage: str
    sel: Selector
    store: object
    qdir: Path
    li: int
    layer: int
    tau: float
    run: Path        # selector run (V2 / S1) or S2 training run
    eval_run: Path   # run whose eval_queries.csv holds this selector's saved per-query table
    chains_csv: Path | None


def s2_eval_run(cfg: dict, stage: str, s2_run: Path, smoke: bool) -> Path:
    ref = cfg["inputs"].get(f"{stage}_eval", "latest")
    runs = [resolve_run(cfg, f"{stage}_eval", ref, smoke)] if ref != "latest" else sorted(
        p for p in (runs_root(cfg) / ("smoke" if smoke else "") / f"{stage}_eval").glob("*")
        if (p / "metrics.json").exists())
    for run in reversed(runs):
        m, meta = read_json(run / "metrics.json"), read_json(run / "meta.json")
        if (m["smoke"] == smoke and m["s2_run"] == s2_run.name
                and meta.get("checkpoint") == cfg["s2_eval"]["checkpoint"]):
            return run
    raise ProtocolError(f"no completed {stage}_eval run of {s2_run.name}/{cfg['s2_eval']['checkpoint']}")


def v4_chains_csv(cfg: dict, stage: str, sel_run: Path, smoke: bool) -> Path | None:
    root = runs_root(cfg) / ("smoke" if smoke else "") / ("v4" if stage == "v2" else f"v4_{stage}")
    for run in sorted(root.glob("*"), reverse=True):
        if (run / "metrics.json").exists() and Path(read_json(run / "meta.json").get("v2_run", "")).name \
                == sel_run.name:
            return run / "chains_natural.csv"
    return None


def stage_source(cfg: dict, stage: str, smoke: bool, manifest, mhash: str, static: dict, qdir: Path) -> Source:
    if stage in ("v2", "s1"):
        store = open_features(cfg, smoke, manifest, mhash, static)
        sel, tau, layer, run = V4.load_selector({**cfg, "inputs": {**cfg["inputs"], "selector_stage": stage}},
                                                smoke, store)
        return Source(stage, sel, store, qdir, static["layers"].index(layer), layer, tau, run, run,
                      v4_chains_csv(cfg, stage, run, smoke))
    run = resolve_run(cfg, stage, cfg["inputs"].get(stage, "latest"), smoke)
    ev = s2_eval_run(cfg, stage, run, smoke)
    ckpt = torch.load(run / cfg["s2_eval"]["checkpoint"], map_location="cpu")
    meta = ckpt["meta"]
    if meta["smoke"] != smoke:
        raise ProtocolError("S2 checkpoint smoke flag does not match the current mode")
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    sel.load_state_dict(ckpt["selector"])
    tau = float(read_json(ev / "metrics.json")["single_step"]["tau_rej"])
    return Source(stage, sel.eval(), S2E.RecomputedStore(ev / "recomputed"), ev / "recomputed", 0, meta["layer"],
                  tau, run, ev, ev / "chains_natural.csv")


@dataclass
class ImageSel:
    sha1: str
    key: str
    rles: list
    tree: T.Tree
    single: list[dict]
    chains: list[dict]


def chain_pick(sig: np.ndarray, tree: T.Tree, tau: float) -> tuple[int, int]:
    # V4 natural search on the true tree; -1 = no answer at that step
    n = sig.shape[1]
    a1 = V4.decide(sig[0], np.arange(n), tau) if n else None
    a2 = V4.decide(sig[1], tree.strict_descendants(tree.root if a1 is None else a1), tau)
    return -1 if a1 is None else a1, -1 if a2 is None else a2


@torch.no_grad()
def select_image(src: Source, sha1: str, key: str, rles: list, tree: T.Tree, single: pd.DataFrame,
                 chains: pd.DataFrame, probes: V2.Probes, device: str) -> ImageSel:
    # mirrors V2.evaluate (single step: argmax over all nodes) and V4.chain_inputs + natural (true tree)
    z = src.store.load(sha1, [f"hidden_L{src.layer}"])[f"hidden_L{src.layer}"]
    if z.shape[0] != len(rles):
        raise ProtocolError(f"{key}: feature rows do not match proposal nodes")
    e = src.sel.nodes(z.to(device)) if len(z) else None
    out = ImageSel(sha1, key, rles, tree, [], [])
    for q in single.itertuples():
        gt = json.loads(q.gt)[0]
        if e is None:
            out.single.append({"qid": q.qid, "kind": q.kind, "gt": gt, "a": -1, "sigma": -np.inf, "accept": False})
            continue
        sig = torch.sigmoid(src.sel.scores(src.sel.queries(probes.get(sha1, q.offset, 1).to(device)), e)[0])
        sig = sig.cpu().numpy()
        a = int(np.argmax(sig))
        out.single.append({"qid": q.qid, "kind": q.kind, "gt": gt, "a": a, "sigma": float(sig[a]),
                           "accept": bool(sig[a] >= src.tau)})
    for q in chains.itertuples():
        gt = json.loads(q.gt)
        if e is None:
            sig = np.zeros((2, 0))
        else:
            sig = torch.sigmoid(src.sel.scores(src.sel.queries(probes.get(sha1, q.offset, 2).to(device)), e))
            sig = sig.cpu().numpy()
        cal, nr = chain_pick(sig, tree, src.tau), chain_pick(sig, tree, -np.inf)
        out.chains.append({"qid": q.qid, "gt_obj": gt[0], "gt_part": gt[1], "cal": cal, "nr": nr})
    return out


def eval_queries(queries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = queries[queries.split == "val"]
    return val[val.kind.isin(["object", "part"])], val[val.kind == "chain"]


def eval_images(queries: pd.DataFrame, smoke: bool, n_smoke: int) -> list[str]:
    single, chains = eval_queries(queries)
    sha = sorted(set(single.sha1) | set(chains.sha1))
    return sha[:n_smoke] if smoke else sha


def selections(cfg: dict, src: Source, props, queries: pd.DataFrame, sha1s: list[str], device: str):
    # per image: node masks, tree, and per query the selected node, its sigma and the accept decision
    single, chains = eval_queries(queries)
    probes = V2.Probes(src.qdir, src.li)
    for sha1 in sha1s:
        nodes = props.nodes(sha1, cfg["proposals"]["node_dedup"])
        tree = T.build_tree(nodes["rles"], device)
        s, c = single[single.sha1 == sha1], chains[chains.sha1 == sha1]
        key = (s if len(s) else c).image_key.iloc[0]
        yield select_image(src, sha1, key, nodes["rles"], tree, s, c, probes, device), (nodes["h"], nodes["w"])


# ---- refinement of one image ----------------------------------------------------------------------------------

def overlap(pred: M.Rle | None, gt: M.Rle) -> tuple[int, int]:
    ga = int(M.areas([gt])[0])
    if pred is None:
        return 0, ga
    inter = int(M.intersections([pred], [gt])[0, 0])
    return inter, int(M.areas([pred])[0]) + ga - inter


def best_nodes(s: ImageSel, gts: dict) -> dict[str, int]:
    # oracle: the node with the highest IoU with the query's GT (first on ties, as V2.evaluate)
    if not s.rles:
        return {q["qid"]: -1 for q in s.single}
    ids = sorted({q["gt"] for q in s.single})
    ious = M.iou(s.rles, [gts[g] for g in ids])
    col = {g: j for j, g in enumerate(ids)}
    return {q["qid"]: int(np.argmax(ious[:, col[q["gt"]]])) for q in s.single}


def refine_image(model, sproc, image: np.ndarray, s: ImageSel, gts: dict, rcfg: dict, pcfg: P.ProposalConfig
                 ) -> tuple[list[dict], list[dict]]:
    h, w = image.shape[:2]
    dtype = getattr(torch, pcfg.dtype)
    best = best_nodes(s, gts)
    selected = {q["a"] for q in s.single} | {a for c in s.chains for a in (*c["cal"], *c["nr"])}
    selected.discard(-1)
    oracle = {b for b in best.values() if b >= 0}
    jobs = [(a, v) for a in sorted(selected) for v in rcfg["variants"]]
    jobs += [(b, v) for b in sorted(oracle - selected) for v in rcfg["oracle_variants"]]
    dense = {a: M.decode(s.rles[a]) for a in sorted(selected | oracle)}
    parent = {a: None if s.tree.parent[a] == s.tree.root else M.decode(s.rles[s.tree.parent[a]]) for a in dense}
    prompts = [make_prompt(v, dense[a], parent[a], rcfg) for a, v in jobs]
    gt_ids = sorted({q["gt"] for q in s.single})
    jobs += [("gt_box", g) for g in gt_ids]
    prompts += [Prompt(M.boxes_xyxy([gts[g]])[0]) for g in gt_ids]
    refined = {(a, "r0"): (s.rles[a], float("nan")) for a in dense}
    if prompts:
        device = next(model.parameters()).device
        pix = sproc(images=image, return_tensors="pt")["pixel_values"].to(device, dtype)
        emb = model.get_image_embeddings(pix)
        res = run_prompts(model, sproc.post_process_masks, emb, prompts, (h, w), sproc.target_size, rcfg,
                          pcfg.mask_threshold, dtype)
        for job, (m, sc) in zip(jobs, res):
            refined[job] = (M.encode(m)[0] if m.any() else None, sc)

    def score(a, v, gt) -> tuple[int, int, float]:
        if a == -1:
            return 0, int(M.areas([gt])[0]), float("nan")
        rle, sc = refined[(a, v)]
        return (*overlap(rle, gt), sc)

    single = []
    for q in s.single:
        gt = gts[q["gt"]]
        row = {k: q[k] for k in ("qid", "kind", "a", "sigma", "accept")}
        row.update(image=s.key, sha1=s.sha1, best=best[q["qid"]], gt_area=int(M.areas([gt])[0]))
        for v in ("r0", *rcfg["variants"]):
            row[f"inter_{v}"], row[f"union_{v}"], row[f"pred_iou_{v}"] = score(q["a"], v, gt)
        for v in ("r0", *rcfg["oracle_variants"]):
            row[f"inter_best_{v}"], row[f"union_best_{v}"], _ = score(row["best"], v, gt)
        row["inter_gt_box"], row["union_gt_box"], row["pred_iou_gt_box"] = score("gt_box", q["gt"], gt)
        single.append(row)
    chains = []
    for c in s.chains:
        go, gp = gts[c["gt_obj"]], gts[c["gt_part"]]
        ga = M.areas([go, gp])
        row = {"qid": c["qid"], "image": s.key, "sha1": s.sha1,
               "single_level": bool(ga[1] / max(1, ga[0]) >= SINGLE_LEVEL)}
        for mode in MODES:
            a1, a2 = c[mode]
            row[f"{mode}_a1"], row[f"{mode}_a2"] = a1, a2
            for v in ("r0", *rcfg["variants"]):
                for name, a, gt in (("obj", a1, go), ("part", a2, gp)):
                    i, u, _ = score(a, v, gt)
                    row[f"{mode}_{name}_{v}"] = i / u if u else 0.0
        chains.append(row)
    return single, chains


# ---- aggregation ----------------------------------------------------------------------------------------------

def ci(values, groups, bs: dict) -> dict:
    return mean_ci(np.asarray(values, np.float64), np.asarray(groups), bs["n"], bs["seed"])


def answered(inter, union, gt_area, accept) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # a rejected query adds its GT area to the union and nothing to the intersection (V2 cIoU definition)
    inter = np.where(accept, np.asarray(inter, np.float64), 0.0)
    union = np.where(accept, np.asarray(union, np.float64), np.asarray(gt_area, np.float64))
    return inter, union, np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)


def single_block(df: pd.DataFrame, inter, union, accept, bs: dict) -> dict:
    inter, union, iou = answered(inter, union, df.gt_area, accept)
    kind, image = df.kind.to_numpy(), df.image.to_numpy()
    return {"n": int(len(df)), "gIoU": ci(iou, image, bs),
            "cIoU": {"value": float(inter.sum() / max(union.sum(), 1.0)),
                     "ci95": ratio_ci(inter, union, image, bs["n"], bs["seed"])},
            "success@0.5": ci(iou >= THR, image, bs),
            "by_kind": {k: ci(iou[kind == k], image[kind == k], bs) for k in sorted(set(kind))}}


def single_summary(df: pd.DataFrame, absent: pd.DataFrame, rcfg: dict, bs: dict) -> dict:
    acc, every = df.accept.to_numpy(bool), np.ones(len(df), bool)
    abstain = (~absent.accept.to_numpy(bool)).astype(np.float64)
    out = {}
    for v in ("r0", *rcfg["variants"]):
        cal = single_block(df, df[f"inter_{v}"], df[f"union_{v}"], acc, bs)
        iou = answered(df[f"inter_{v}"], df[f"union_{v}"], df.gt_area, acc)[2]
        cal["gIoU_present_and_no_target"] = ci(np.concatenate([iou, abstain]),
                                               np.concatenate([df.image.to_numpy(), absent.image.to_numpy()]), bs)
        out[v] = {"calibrated": cal, "no_rejection": single_block(df, df[f"inter_{v}"], df[f"union_{v}"], every, bs)}
        if v != "r0":
            out[v]["mean_predicted_iou"] = float(np.nanmean(df[f"pred_iou_{v}"])) if df[f"pred_iou_{v}"].notna().any() \
                else None
    return out


def chain_summary(ch: pd.DataFrame, rcfg: dict, bs: dict) -> dict:
    single = ch.single_level.to_numpy(bool)
    image = ch.image.to_numpy()
    out = {}
    for mode, name in MODES.items():
        out[name] = {}
        for v in ("r0", *rcfg["variants"]):
            s1 = ch[f"{mode}_obj_{v}"].to_numpy() >= THR
            s2 = ch[f"{mode}_part_{v}"].to_numpy() >= THR
            out[name][v] = {split: {"chains": int(m.sum()),
                                    "chain_success_end_to_end": ci((s1 & s2)[m], image[m], bs),
                                    "step1_object_success@0.5": ci(s1[m], image[m], bs),
                                    "final_part_success@0.5": ci(s2[m], image[m], bs)}
                            for split, m in (("all", np.ones(len(ch), bool)), ("two_level", ~single),
                                             ("single_level", single))}
    return out


def consistency(df: pd.DataFrame, ch: pd.DataFrame, saved: pd.DataFrame, chains_csv: Path | None) -> dict:
    # r0 recomputed here vs the stage's own saved tables (same selector, features, probes and proposals)
    pres = saved[saved.present].set_index("qid")
    j = df.set_index("qid").join(pres[["sel_iou", "max_sigma"]], how="left")
    r0 = np.where(j.union_r0 > 0, j.inter_r0 / j.union_r0.clip(lower=1), 0.0)
    out = {"single_missing_in_saved": int(j.sel_iou.isna().sum()),
           "single_sel_iou_agreement": float((np.abs(r0 - j.sel_iou) < 1e-6).mean()) if len(j) else None,
           "single_sigma_max_abs_diff": float(np.nanmax(np.abs(np.where(np.isfinite(j.sigma), j.sigma - j.max_sigma,
                                                                          0.0)))) if len(j) else None}
    if chains_csv is not None and chains_csv.exists() and len(ch):
        nat = pd.read_csv(chains_csv)
        true = nat[nat.structure == "true"].set_index("qid").chain
        mine = ch.set_index("qid")
        mine = (mine.cal_obj_r0 >= THR) & (mine.cal_part_r0 >= THR)
        both = mine.index.intersection(true.index)
        out["chains_saved"] = str(chains_csv)
        out["chain_success_agreement"] = float((mine[both] == true[both].astype(bool)).mean()) if len(both) else None
        out["chains_missing_in_saved"] = int(len(mine) - len(both))
    else:
        out["chains_saved"] = None
    return out


def v1_ceiling_reference(cfg: dict) -> dict | None:
    try:
        run = resolve_run(cfg, "v1_ceiling", cfg["inputs"].get("v1_ceiling", "latest"), False)
    except ProtocolError:
        return None
    o = read_json(run / "metrics.json").get("oracle", {})
    return {"v1_ceiling_run": run.name,
            "note": "recall@0.5 over all PACO val object / part annotations (not the I1 query set), own GT box",
            **{k: {v: o[k][v]["recall"] for v in ("top_scored_mask", "any_of_3_masks")}
               for k in ("object/own_box", "part/own_box") if k in o}}


# ---- shards and merge -----------------------------------------------------------------------------------------

def approvals(stage: str) -> list[str]:
    return ["D19", "D15", "D12"] + {"s1": ["D11"], "s2": ["D10"], "s2_from_s1": ["D10", "D11"]}.get(stage, [])


def identity(cfg: dict, src: Source, mhash: str, smoke: bool) -> dict:
    return {"kind": "refine", "stage": src.stage, "smoke": smoke, "manifest": mhash, "selector_run": src.run.name,
            "eval_run": src.eval_run.name, "tau_rej": src.tau, "layer": src.layer, "sam3": cfg["models"]["sam3"],
            "proposals": {k: cfg["proposals"][k] for k in ("node_dedup", "mask_threshold", "dtype")},
            "node_cap": node_cap(cfg, smoke), "settings": cfg["refine"],
            "code": hashlib.sha256((ROOT / "scripts" / "refine.py").read_bytes()).hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if ident["smoke"] else "full") / "refine" / ident["stage"]
            / canonical_hash(ident)[:16])


def inputs(cfg: dict, stage: str, smoke: bool):
    require_approved(cfg, approvals(stage))
    gate = v1_gate(cfg, smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    static = qwen_static(cfg)
    qdir, queries = open_queries(cfg, smoke, mhash, static)
    src = stage_source(cfg, stage, smoke, manifest, mhash, static, qdir)
    ident = identity(cfg, src, mhash, smoke)
    return gate, mrun, manifest, mhash, wl, queries, src, ident, out_dir(cfg, ident)


def write_parquet(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)


def run_shard(cfg: dict, stage: str, shard: int, num_shards: int, smoke: bool) -> None:
    gate, mrun, manifest, mhash, wl, queries, src, ident, d = inputs(cfg, stage, smoke)
    target = d / "shards" / f"{shard:05d}_single.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    sha1s = eval_images(queries, smoke, cfg["refine"]["smoke_images"])[shard::num_shards]
    run = start_run(f"refine_{stage}", cfg, smoke, {"shard": shard, "num_shards": num_shards, "cache": str(d),
                                                     "manifest_run": str(mrun), "selector_run": str(src.run),
                                                     "eval_run": str(src.eval_run), "v1_gate": gate})
    device = cfg["device"]
    src.sel = src.sel.to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = open_proposals(cfg, smoke, manifest, mhash, capped=True)
    p, sam = cfg["proposals"], cfg["models"]["sam3"]
    pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"], p["stability_offset"],
                            p["mask_threshold"], p["min_area"], p["points_per_batch"], p["dtype"])
    model, sproc = P.load_sam3(sam["repo"], sam["revision"], device, p["dtype"], cfg.get("hf_home"))
    paths = dict(zip(wl.sha1, wl.path))
    keys = dict(zip(queries.sha1, queries.image_key))
    annos = paco.load(cfg["data"]["paco_lvis"], "val", {int(keys[s].split(":")[1]) for s in sha1s})
    single, chains, t0 = [], [], time.time()
    for n, (s, hw) in enumerate(selections(cfg, src, props, queries, sha1s, device), 1):
        img = annos[int(s.key.split(":")[1])]
        image = load_rgb(assert_real_path(paths[s.sha1]))
        if not image.shape[:2] == hw == (img.h, img.w):
            raise ProtocolError(f"{s.key}: image {image.shape[:2]}, proposals {hw}, annotation {(img.h, img.w)}")
        a, b = refine_image(model, sproc, image, s, paco.gt_rles(img), cfg["refine"], pcfg)
        single += a
        chains += b
        if n % 50 == 0 or n == len(sha1s):
            print(f"[{n}/{len(sha1s)}] queries {len(single)} chains {len(chains)} {time.time() - t0:.0f}s", flush=True)
    write_parquet(chains, d / "shards" / f"{shard:05d}_chains.parquet")
    write_parquet(single, target)
    write_json(run / "metrics.json", {"images": len(sha1s), "queries": len(single), "chains": len(chains),
                                      "seconds": time.time() - t0})


def merge(cfg: dict, stage: str, num_shards: int, smoke: bool) -> None:
    gate, mrun, manifest, mhash, wl, queries, src, ident, d = inputs(cfg, stage, smoke)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}_single.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} refine shards missing under {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}_single.parquet") for k in range(num_shards)],
                   ignore_index=True)
    ch = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}_chains.parquet") for k in range(num_shards)],
                   ignore_index=True)
    if not len(ch):
        ch = pd.DataFrame(columns=["qid", "image", "sha1", "single_level"])
    sha1s = set(eval_images(queries, smoke, cfg["refine"]["smoke_images"]))
    single_q, chain_q = eval_queries(queries)
    for name, got, want in (("single-step", df, single_q), ("chain", ch, chain_q)):
        expected = set(want[want.sha1.isin(sha1s)].qid)
        if set(got.qid) != expected or got.qid.duplicated().any():
            raise ProtocolError(f"merged {name} queries do not match the evaluation set "
                                f"({len(got)} vs {len(expected)})")
    saved = pd.read_csv(src.eval_run / "eval_queries.csv")
    absent = saved[~saved.present.astype(bool)].copy()
    absent["accept"] = absent.max_sigma >= src.tau
    run = start_run(f"refine_{stage}_merge", cfg, smoke, {"cache": str(d), "manifest_run": str(mrun),
                                                           "selector_run": str(src.run), "eval_run": str(src.eval_run)})
    bs, rcfg = cfg["v2"]["bootstrap"], cfg["refine"]
    every = np.ones(len(df), bool)
    metrics = {
        "deviation": "D19", "label": "dev", "smoke": smoke, "stage": stage, "selector_run": src.run.name,
        "eval_run": src.eval_run.name, "tau_rej": src.tau, "layer": src.layer, "images": int(df.image.nunique()),
        "split": "PACO-LVIS val I1 object/part queries and object->part chains (+ gRefCOCO val no-target)",
        "variants": {"r0": "selected node mask", "r1": "node box", "r2": "node box + interior positive point",
                     "r3": "node box + node mask as dense prompt (low-res logits)",
                     "r4": "r2 + negative points inside the tree parent, outside the node"},
        "output_rule": "multimask output (3) with the highest predicted IoU, first on ties; logits > mask_threshold",
        "settings": rcfg, "sam3": cfg["models"]["sam3"],
        "single_step": single_summary(df, absent, rcfg, bs),
        "no_target": {"n": int(len(absent)), "abstention_rate": float((~absent.accept).mean()) if len(absent) else None,
                      "note": "refinement does not change accept decisions; from the stage's saved table"},
        "oracle": {"best_node": {v: single_block(df, df[f"inter_best_{v}"], df[f"union_best_{v}"], every, bs)
                                 for v in ("r0", *rcfg["oracle_variants"])},
                   "gt_box": single_block(df, df.inter_gt_box, df.union_gt_box, every, bs),
                   "v1_ceiling_reference": v1_ceiling_reference(cfg),
                   "note": "no rejection; best node = highest-IoU node of the capped proposal set"},
        "chains": {"search": "V4 natural search on the true tree; step 1 and step 2 nodes both refined",
                   "chains": int(len(ch)), **(chain_summary(ch, rcfg, bs) if len(ch) else {})},
        "consistency": consistency(df, ch, saved, src.chains_csv), "v1_gate": gate}
    df.to_csv(run / "queries.csv", index=False)
    ch.to_csv(run / "chains.csv", index=False)
    write_json(run / "metrics.json", metrics)
    for v in ("r0", *rcfg["variants"]):
        m = metrics["single_step"][v]
        print(f"{stage} {v}: present gIoU calibrated {m['calibrated']['gIoU']['mean']:.3f} "
              f"no rejection {m['no_rejection']['gIoU']['mean']:.3f}")
    print(json.dumps(metrics["consistency"]))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = 1 if args.smoke else cfg["refine"]["num_shards"]
    if args.merge:
        merge(cfg, args.stage, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.stage, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
