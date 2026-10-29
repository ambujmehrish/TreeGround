import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_eval as S2E  # noqa: E402
import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json, resolve_run, start_run, write_json  # noqa: E402
from treeground.selector import Selector  # noqa: E402

# Selector diagnosis of S2 long on PACO-LVIS VAL (development only; no test split is read). Uses the node features
# and <SEL> probes the S2 long evaluation run already recomputed with the LoRA model (s2_eval.py, recomputed/), the
# capped tree nodes (no SAM 3 text instances: the pool S2 long was trained on) and the selected checkpoint.
# Per present object / part query: rank of the best node under the selector, the picked node's relation to the
# best node (too big / too small / near duplicate / elsewhere), feature similarity of picked vs best (raw layer
# states and selector embeddings), relative size, score margin, and whether different queries of one image get
# the same pick (query sensitivity).


def relation(pick: dict, best: dict, inter_pb: int) -> str:
    if pick["i"] == best["i"]:
        return "correct"
    ap, ab = pick["area"], best["area"]
    iou_pb = inter_pb / max(1, ap + ab - inter_pb)
    if iou_pb >= 0.5:
        return "near_duplicate"
    if inter_pb >= 0.9 * ab and ap > ab:
        return "too_big"          # the pick contains the best node (an ancestor / larger region)
    if inter_pb >= 0.9 * ap and ap < ab:
        return "too_small"        # the pick lies inside the best node (a part of the target)
    if inter_pb > 0:
        return "partial_overlap"
    return "elsewhere"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    stage = S2T.s2_stage(cfg)
    ev = resolve_run(cfg, f"{stage}_eval", cfg["inputs"].get(f"{stage}_eval", "latest"), False)
    ckpt, s2_run = S2E.load_checkpoint(cfg, False)
    if read_json(ev / "metrics.json")["s2_run"] != s2_run.name:
        raise ProtocolError(f"eval run {ev.name} was made with {read_json(ev / 'metrics.json')['s2_run']}, "
                            f"not {s2_run.name}")
    meta = ckpt["meta"]
    layer = meta["layer"]
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    sel.load_state_dict(ckpt["selector"])
    sel.eval()
    _, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    _, queries = open_queries(cfg, False, mhash, qwen_static(cfg))
    store, probes = S2E.RecomputedStore(ev / "recomputed"), V2.Probes(ev / "recomputed", 0)
    tau = cfg["proposals"]["node_dedup"]
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    qval = queries[(queries.split == "val") & queries.kind.isin(["object", "part"])]
    groups = list(qval.groupby("sha1"))
    if args.max_images:
        groups = groups[:args.max_images]
    run = start_run(f"{stage}_diag", cfg, False, {"eval_run": str(ev), "s2_run": str(s2_run), "split": "PACO val"})
    rows, image_rows = [], []
    with torch.no_grad():
        for n_img, (sha1, g) in enumerate(groups, 1):
            key = g.image_key.iloc[0]
            img = images[int(key.split(":")[1])]
            nodes = props.nodes(sha1, tau)
            z = store.load(sha1, [f"hidden_L{layer}"])[f"hidden_L{layer}"].float()
            if not len(z):
                continue
            e = sel.nodes(z)
            zn = F.normalize(z, dim=-1)
            areas = nodes["areas"].astype(np.float64)
            hw = nodes["h"] * nodes["w"]
            gts = paco.gt_rles(img)
            picks, bests = {}, {}
            rng = np.random.default_rng(len(rows))
            for q in g.itertuples():
                gt = json.loads(q.gt)
                ious = M.iou(nodes["rles"], [gts[gt[0]]])[:, 0]
                s = sel.scores(sel.queries(probes.get(sha1, q.offset, 1).float()), e)[0].numpy()
                order = np.argsort(-s)
                a, b = int(order[0]), int(np.argmax(ious))
                rank_best = int(np.flatnonzero(order == b)[0]) + 1
                good = np.flatnonzero(ious >= 0.5)
                rank_good = int(min(np.flatnonzero(np.isin(order, good))) + 1) if len(good) else None
                inter_pb = int(M.intersections([nodes["rles"][a]], [nodes["rles"][b]])[0, 0])
                rel = relation({"i": a, "area": areas[a]}, {"i": b, "area": areas[b]}, inter_pb)
                picks[q.qid], bests[q.qid] = a, b
                other = [i for i in range(len(z)) if i != b]
                r = int(rng.choice(other)) if other else b
                rows.append({
                    "qid": q.qid, "kind": q.kind, "image": key, "n_nodes": len(z),
                    "best_iou": float(ious[b]), "pick_iou": float(ious[a]),
                    "covered": bool(ious[b] >= 0.5), "success": bool(ious[a] >= 0.5),
                    "rank_best": rank_best, "rank_first_good": rank_good, "relation": rel,
                    "cos_raw_pick_best": float(zn[a] @ zn[b]), "cos_sel_pick_best": float(e[a] @ e[b]),
                    "cos_raw_best_random": float(zn[b] @ zn[r]), "cos_sel_best_random": float(e[b] @ e[r]),
                    "score_margin_pick_minus_best": float(s[a] - s[b]),
                    "best_rel_area": float(areas[b] / hw), "pick_rel_area": float(areas[a] / hw),
                    "best_stability": float(nodes["stability"][b]),
                })
            qids = list(picks)
            for i in range(len(qids)):
                for j in range(i + 1, len(qids)):
                    if bests[qids[i]] != bests[qids[j]]:
                        # two queries whose targets are different nodes: does the selector still pick the same one?
                        image_rows.append({"image": key, "same_pick": picks[qids[i]] == picks[qids[j]]})
            if n_img % 500 == 0:
                print(f"{n_img}/{len(groups)} images, {len(rows)} queries", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "diag_queries.csv", index=False)
    cov = df[df.covered]
    wrong = cov[~cov.success]

    def frac(x):
        return {k: round(float(v), 4) for k, v in x.value_counts(normalize=True).items()}

    bins = [0, 0.005, 0.02, 0.08, 0.3, 1.01]
    cov = cov.assign(size_bin=pd.cut(cov.best_rel_area, bins).astype(str))
    out = {
        "label": "dev", "split": "PACO-LVIS val, present object / part queries", "stage": stage,
        "eval_run": ev.name, "checkpoint_step": meta["step"], "queries": int(len(df)), "covered": int(len(cov)),
        "success_if_covered": float(cov.success.mean()),
        "oracle_best_iou_mean": float(df.best_iou.mean()), "pick_iou_mean": float(df.pick_iou.mean()),
        "by_kind": {k: {"success_if_covered": float(gk.success.mean()), "n": int(len(gk))}
                    for k, gk in cov.groupby("kind")},
        "recall_of_first_good_at_k": {k: float((cov.rank_first_good <= k).mean()) for k in (1, 2, 3, 5, 10)},
        "rank_best_median": float(cov.rank_best.median()),
        "wrong_pick_relation": frac(wrong.relation),
        "wrong_pick_relation_by_kind": {k: frac(gk.relation) for k, gk in wrong.groupby("kind")},
        "feature_similarity_wrong_picks": {
            "raw_cos_median": float(wrong.cos_raw_pick_best.median()),
            "raw_cos_ge_0.95": float((wrong.cos_raw_pick_best >= 0.95).mean()),
            "sel_cos_median": float(wrong.cos_sel_pick_best.median()),
            "sel_cos_ge_0.9": float((wrong.cos_sel_pick_best >= 0.9).mean())},
        "feature_similarity_reference_best_vs_random_node": {
            "raw_cos_median": float(cov.cos_raw_best_random.median()),
            "sel_cos_median": float(cov.cos_sel_best_random.median())},
        "success_by_target_size": {k: {"success": float(gk.success.mean()), "n": int(len(gk))}
                                   for k, gk in cov.groupby("size_bin")},
        "score_margin_wrong_median": float(wrong.score_margin_pick_minus_best.median()),
        "query_sensitivity": {
            "query_pairs_with_different_targets": len(image_rows),
            "same_pick_rate": float(np.mean([r["same_pick"] for r in image_rows])) if image_rows else None},
    }
    write_json(run / "metrics.json", out)
    print(json.dumps(out, indent=1))
    print(run)


if __name__ == "__main__":
    main()
