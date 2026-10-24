import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refine as R  # noqa: E402
import v2_train_eval as V2  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import rerank as RR  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, node_cap, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import ProtocolError, load_config, require_approved, start_run, v1_gate, write_json  # noqa: E402

# Re-ranking of the selector's top-k nodes (deviation D20): cheap per-node features, rankers fit on PACO val with
# image-level K-fold cross-validation (each query scored by a model that never saw its image). Accept decisions are
# the selector's. Numbers are cross-validated dev numbers, not a held-out test.

STAGES = R.STAGES
THR = R.THR
METHODS = ("selector", "rule", "pointwise", "listwise", "oracle_topk")


def approvals(stage: str) -> list[str]:
    return ["D20", "D15"] + {"s1": ["D11"], "s2": ["D10"], "s2_from_s1": ["D10", "D11"]}.get(stage, [])


@torch.no_grad()
def image_rows(src: R.Source, props, probes: V2.Probes, sha1: str, s: pd.DataFrame, img: paco.PacoImage, k: int,
               tau_d: float, device: str) -> tuple[list[dict], list[pd.DataFrame]]:
    nodes = props.nodes(sha1, tau_d)
    rles, key = nodes["rles"], s.image_key.iloc[0]
    if (nodes["h"], nodes["w"]) != (img.h, img.w):
        raise ProtocolError(f"{key}: annotation size differs from proposal masks")
    gts = paco.gt_rles(img)
    gt_ids = sorted({json.loads(g)[0] for g in s.gt})
    col = {g: j for j, g in enumerate(gt_ids)}
    ga = M.areas([gts[g] for g in gt_ids]).astype(np.int64)
    n = len(rles)
    if n:
        tree = T.build_tree(rles, device)
        meta = props.rows(sha1).set_index("src_index").loc[nodes["src_index"]]
        sam_iou = meta["sam_iou"].to_numpy(np.float64)
        area_frac = nodes["areas"] / float(nodes["h"] * nodes["w"])
        inter = M.intersections(rles, [gts[g] for g in gt_ids]).astype(np.int64)
        union = nodes["areas"][:, None] + ga[None] - inter
        z = src.store.load(sha1, [f"hidden_L{src.layer}"])[f"hidden_L{src.layer}"]
        if z.shape[0] != n:
            raise ProtocolError(f"{key}: feature rows do not match proposal nodes")
        e = src.sel.nodes(z.to(device))
    qs, cands = [], []
    for q in s.itertuples():
        j = col[json.loads(q.gt)[0]]
        rec = {"qid": q.qid, "image": key, "sha1": sha1, "kind": q.kind, "gt_area": int(ga[j]), "n_nodes": n}
        if not n:
            qs.append({**rec, "sigma_top1": -np.inf, "accept": False, "best_inter": 0, "best_union": int(ga[j])})
            continue
        logit = src.sel.scores(src.sel.queries(probes.get(sha1, q.offset, 1).to(device)), e)[0]
        sig = torch.sigmoid(logit).cpu().numpy().astype(np.float64)
        logit = logit.cpu().numpy().astype(np.float64)
        iou = inter[:, j] / np.maximum(union[:, j], 1)
        b = int(np.argmax(iou))
        f = RR.candidate_features(sig, logit, tree, area_frac, sam_iou, nodes["stability"], k)
        nd = f.node.to_numpy()
        f = f.assign(qid=q.qid, image=key, sha1=sha1, kind=q.kind, inter=inter[nd, j], union=union[nd, j],
                     iou=iou[nd])
        top = float(sig[f.node.iloc[0]])
        qs.append({**rec, "sigma_top1": top, "accept": bool(top >= src.tau), "best_inter": int(inter[b, j]),
                   "best_union": int(union[b, j])})
        cands.append(f)
    return qs, cands


def extract(cfg: dict, src: R.Source, props, queries: pd.DataFrame, sha1s: list[str], device: str
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = R.eval_queries(queries)[0]
    single = single[single.sha1.isin(set(sha1s))]
    keys = dict(zip(single.sha1, single.image_key))
    annos = paco.load(cfg["data"]["paco_lvis"], "val", {int(keys[s].split(":")[1]) for s in keys})
    probes = V2.Probes(src.qdir, src.li)
    k = max(cfg["rerank"]["ks"])
    qs, cands, t0 = [], [], time.time()
    groups = list(single.groupby("sha1", sort=True))
    for n, (sha1, s) in enumerate(groups, 1):
        a, b = image_rows(src, props, probes, sha1, s, annos[int(keys[sha1].split(":")[1])], k,
                          cfg["proposals"]["node_dedup"], device)
        qs += a
        cands += b
        if n % 200 == 0 or n == len(groups):
            print(f"[{n}/{len(groups)}] queries {len(qs)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(qs), pd.concat(cands, ignore_index=True)


def changes(c: pd.DataFrame, base: pd.Series, new: pd.Series) -> dict:
    a, b = c.loc[base.to_numpy(), "iou"].to_numpy(), c.loc[new.reindex(base.index).to_numpy(), "iou"].to_numpy()
    moved = base.to_numpy() != new.reindex(base.index).to_numpy()
    return {"changed": float(moved.mean()) if len(moved) else 0.0, "improved": int((b > a).sum()),
            "worsened": int((b < a).sum()), "mean_iou_delta_changed": float((b - a)[moved].mean()) if moved.any()
            else 0.0}


def blocks(qs: pd.DataFrame, absent: pd.DataFrame, inter, union, bs: dict) -> dict:
    acc, every = qs.accept.to_numpy(bool), np.ones(len(qs), bool)
    cal = R.single_block(qs, inter, union, acc, bs)
    iou = R.answered(inter, union, qs.gt_area, acc)[2]
    abstain = (~absent.accept.to_numpy(bool)).astype(np.float64)
    cal["gIoU_present_and_no_target"] = R.ci(np.concatenate([iou, abstain]),
                                             np.concatenate([qs.image.to_numpy(), absent.image.to_numpy()]), bs)
    return {"calibrated": cal, "no_rejection": R.single_block(qs, inter, union, every, bs)}


def analyse(qs: pd.DataFrame, cand: pd.DataFrame, absent: pd.DataFrame, rc: dict, bs: dict) -> tuple[dict, dict]:
    n_folds = min(rc["folds"], qs.image.nunique())
    fold_of = RR.assign_folds(qs.image, n_folds, rc["seed"])
    out, detail = {}, {}
    for k in rc["ks"]:
        c = cand[cand["rank"] < k].reset_index(drop=True)
        fold = c.image.map(fold_of).to_numpy()
        picks = {"selector": RR.pick(c, -c["rank"].to_numpy()), "oracle_topk": RR.pick(c, c["iou"])}
        s, ws = RR.cross_validate(c, fold, lambda d: RR.fit_rule(d, rc["rule_grid"]),
                                  lambda w, d: RR.rule_score(d, w))
        picks["rule"] = RR.pick(c, s)
        s, pw = RR.cross_validate(c, fold, RR.pointwise_fit(rc["l2"], THR), RR.linear_predict)
        picks["pointwise"] = RR.pick(c, s)
        s, lw = RR.cross_validate(c, fold, RR.listwise_fit(rc["l2"]), RR.linear_predict)
        picks["listwise"] = RR.pick(c, s)
        names = RR.design(c.head(1))[1]
        out[f"k{k}"] = {m: blocks(qs, absent, *RR.picked_overlap(qs, c, picks[m]), bs) for m in METHODS}
        detail[f"k{k}"] = {
            "candidates": int(len(c)), "folds": n_folds, "rule_weight_per_fold": [float(w) for w in ws],
            "pick_changes_vs_selector": {m: changes(c, picks["selector"], picks[m])
                                         for m in METHODS if m != "selector"},
            "importances": {"pointwise": RR.importances(pw, names), "listwise": RR.importances(lw, names)}}
        for m in picks:
            qs[f"pick_k{k}_{m}"] = qs.qid.map(c.loc[picks[m].to_numpy(), "node"].set_axis(picks[m].index))
    return out, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg, stage, smoke = load_config(args.config), args.stage, args.smoke
    rc, bs = cfg["rerank"], cfg["v2"]["bootstrap"]
    require_approved(cfg, approvals(stage))
    gate = v1_gate(cfg, smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, smoke)
    static = qwen_static(cfg)
    qdir, queries = open_queries(cfg, smoke, mhash, static)
    src = R.stage_source(cfg, stage, smoke, manifest, mhash, static, qdir)
    run = start_run(f"rerank_{stage}", cfg, smoke, {"manifest_run": str(mrun), "selector_run": str(src.run),
                                                     "eval_run": str(src.eval_run), "v1_gate": gate})
    device = cfg["device"]
    src.sel = src.sel.to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = open_proposals(cfg, smoke, manifest, mhash, capped=True)
    sha1s = R.eval_images(queries, smoke, rc["smoke_images"])
    t0 = time.time()
    qs, cand = extract(cfg, src, props, queries, sha1s, device)
    expected = R.eval_queries(queries)[0]
    if set(qs.qid) != set(expected[expected.sha1.isin(set(sha1s))].qid) or qs.qid.duplicated().any():
        raise ProtocolError("extracted queries do not match the evaluation set")
    saved = pd.read_csv(src.eval_run / "eval_queries.csv")
    absent = saved[~saved.present.astype(bool)].copy()
    absent["accept"] = absent.max_sigma >= src.tau
    single, detail = analyse(qs, cand, absent, rc, bs)
    every = np.ones(len(qs), bool)
    top = RR.picked_overlap(qs, cand, RR.pick(cand, -cand["rank"].to_numpy()))
    r0 = qs.assign(inter_r0=top[0], union_r0=top[1], sigma=qs.sigma_top1)
    metrics = {
        "deviation": "D20", "label": "dev (image-level K-fold CV on PACO val, not a held-out test)", "smoke": smoke,
        "stage": stage, "selector_run": src.run.name, "eval_run": src.eval_run.name, "tau_rej": src.tau,
        "layer": src.layer, "node_cap": node_cap(cfg, smoke), "images": int(qs.image.nunique()),
        "split": "PACO-LVIS val I1 object/part queries (+ gRefCOCO val no-target, accept unchanged)",
        "settings": rc, "primary_k": rc["primary_k"], "features": list(RR.FEATURES),
        "methods": {"selector": "top-1 by sigma (stage's own pick)",
                    "rule": "logit + w log(max(sam_iou, 1e-3)), w from rule_grid by CV (max mean IoU on train folds)",
                    "pointwise": "L2 logistic regression, target IoU >= 0.5, features + features x part + part",
                    "listwise": "L2 conditional logit over the query's top-k, target = highest-IoU candidate",
                    "oracle_topk": "highest-IoU candidate of the top-k (reranking ceiling)"},
        "single_step": single, "detail": detail,
        "oracle_all_nodes": R.single_block(qs, qs.best_inter, qs.best_union, every, bs),
        "no_target": {"n": int(len(absent)), "abstention_rate": float((~absent.accept).mean()) if len(absent) else None,
                      "note": "reranking does not change accept decisions; from the stage's saved table"},
        "consistency": R.consistency(r0, pd.DataFrame(), saved, None), "v1_gate": gate,
        "seconds": time.time() - t0}
    qs.to_parquet(run / "queries.parquet", index=False)
    cand.to_parquet(run / "candidates.parquet", index=False)
    write_json(run / "metrics.json", metrics)
    for k in rc["ks"]:
        for m in METHODS:
            b = single[f"k{k}"][m]
            print(f"{stage} k={k} {m}: gIoU calibrated {b['calibrated']['gIoU']['mean']:.3f} "
                  f"no rejection {b['no_rejection']['gIoU']['mean']:.3f} "
                  f"part {b['no_rejection']['by_kind'].get('part', {}).get('mean', float('nan')):.3f}")
    print(json.dumps(metrics["consistency"]))
    print(run)


if __name__ == "__main__":
    main()
