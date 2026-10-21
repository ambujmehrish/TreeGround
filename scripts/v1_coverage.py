import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco, partonomy, sceneparser  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import ProtocolError, ensure_deviation_log, load_config, read_deviations, start_run, write_json  # noqa: E402

THR = 0.5


def chain_coverage(tree: T.Tree, step_ious: np.ndarray) -> tuple[bool, bool]:
    return T.max_iou_path(tree, step_ious, THR) is not None, T.best_match_path(tree, step_ious, THR) is not None


def summarize_chains(rows: list[dict]) -> dict:
    if not rows:
        return {"chains": 0}
    df = pd.DataFrame(rows)
    out = {"chains": len(df), "feasible_path_coverage": float(df.feasible.mean()),
           "best_match_path_coverage": float(df.best_match.mean())}
    out["gap"] = out["feasible_path_coverage"] - out["best_match_path_coverage"]
    if "length" in df:
        out["by_length"] = {int(k): {"chains": len(g), "feasible_path_coverage": float(g.feasible.mean()),
                                     "best_match_path_coverage": float(g.best_match.mean())}
                            for k, g in df.groupby("length")}
    return out


def paco_val(cfg, props, wl, tau, chains: bool = True):
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    parts, chain_rows, i1 = [], [], defaultdict(int)
    for row in wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key").itertuples():
        img = images[int(row.image_key.split(":")[1])]
        nodes = props.nodes(row.sha1, tau)
        if (nodes["h"], nodes["w"]) != (img.h, img.w):
            raise ProtocolError(f"{row.image_key}: annotation size differs from proposal masks")
        part_iou = M.iou(nodes["rles"], [p.rle for p in img.parts])
        gt_area = M.areas([p.rle for p in img.parts])
        for k, p in enumerate(img.parts):
            best = float(part_iou[:, k].max()) if len(nodes["rles"]) else 0.0
            parts.append({"label": p.label, "area_bin": area_bin(gt_area[k]), "best_iou": best, "hit": best >= THR})
        queries, stats = paco.i1_queries(img)
        for k, v in stats.items():
            i1[k] += v
        if not chains:
            continue
        tree = T.build_tree(nodes["rles"])
        gts = paco.gt_rles(img)
        for q in (q for q in queries if q.kind == "chain"):
            step = M.iou(nodes["rles"], [gts[a] for a in q.gt]).T if len(nodes["rles"]) else np.zeros((2, 0))
            feas, best = chain_coverage(tree, step) if step.shape[1] else (False, False)
            chain_rows.append({"qid": q.qid, "feasible": feas, "best_match": best, "length": 2})
    return pd.DataFrame(parts), chain_rows, dict(i1)


def recall_tables(parts: pd.DataFrame) -> dict:
    return {
        "part_recall": float(parts.hit.mean()) if len(parts) else float("nan"),
        "part_annotations": len(parts),
        "by_area_bin": {k: {"recall": float(g.hit.mean()), "n": len(g)} for k, g in parts.groupby("area_bin")},
        "by_category": {k: {"recall": float(g.hit.mean()), "n": len(g)} for k, g in parts.groupby("label")},
    }


def partonomy_cov(cfg, props, wl, tau):
    imgs = partonomy.load(cfg["data"]["partonomy"])
    by_key = {i.key: i for i in imgs.values()}
    hits, chains = [], []
    for row in wl[wl.role == "partonomy"].drop_duplicates("image_key").itertuples():
        img = by_key[row.image_key]
        nodes = props.nodes(row.sha1, tau)
        all_parts = [r for rs in img.parts.values() for r in rs]
        for r in all_parts + img.roots:
            if tuple(r["size"]) != (nodes["h"], nodes["w"]):
                raise ProtocolError(f"{row.image_key}: PARTONOMY mask size differs from the image")
        if all_parts:
            best = M.iou(nodes["rles"], all_parts).max(0) if nodes["rles"] else np.zeros(len(all_parts))
            hits += list(best >= THR)
        tree = T.build_tree(nodes["rles"])
        for name, root, part in partonomy.chains(img):
            step = M.iou(nodes["rles"], [root, part]).T if nodes["rles"] else np.zeros((2, 0))
            feas, best = chain_coverage(tree, step) if step.shape[1] else (False, False)
            chains.append({"feasible": feas, "best_match": best, "length": 2})
    out = {"matching": "mask", "part_recall": float(np.mean(hits)) if hits else float("nan"),
           "part_annotations": len(hits), **summarize_chains(chains)}
    if not any(i.roots for i in imgs.values()):
        out["chains_note"] = "no object-level masks (is_root_concept) in the release; chain coverage not computable"
    return out


def sceneparser_cov(cfg, props, wl, tau):
    wanted = set(wl[wl.role == "sceneparser_val"].image_key)
    keys = dict(zip(wl.image_key, wl.sha1))
    chains = []
    for img in sceneparser.iter_split(cfg["data"]["sceneparser"], "val"):
        if img.key not in wanted:
            continue
        nodes = props.nodes(keys[img.key], tau)
        tree = T.build_tree(nodes["rles"])
        for c in sceneparser.chains(img):
            step = M.box_iou(np.asarray(c["boxes"]), nodes["boxes"]) if len(nodes["rles"]) else np.zeros((len(c["boxes"]), 0))
            feas, best = chain_coverage(tree, step) if step.shape[1] else (False, False)
            chains.append({"feasible": feas, "best_match": best, "length": len(c["boxes"])})
    return {"matching": "box-matched", **summarize_chains(chains)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    props = open_proposals(cfg, args.smoke, manifest, mhash)
    run = start_run("v1", cfg, args.smoke, {"manifest_run": str(mrun), "manifest_sha256": mhash,
                                             "proposals": str(props.path)})
    roles = set(wl.role)
    out = {"split": "PACO-LVIS val", "label": "dev", "smoke": args.smoke, "by_tau": {}}
    for tau in cfg["v1"]["taus"]:
        parts, chains, i1 = paco_val(cfg, props, wl, tau)
        parts.to_csv(run / f"paco_parts_tau{tau}.csv", index=False)
        res = {"paco": {**recall_tables(parts), **summarize_chains(chains), "i1_filter": i1,
                        "i1_part_fraction_kept": i1.get("parts_kept", 0) / max(1, i1.get("parts", 0)),
                        "i1_object_fraction_kept": i1.get("objects_kept", 0) / max(1, i1.get("objects", 0))}}
        res["partonomy"] = partonomy_cov(cfg, props, wl, tau) if "partonomy" in roles else "missing"
        res["sceneparser"] = sceneparser_cov(cfg, props, wl, tau) if "sceneparser_val" in roles else "missing"
        out["by_tau"][str(tau)] = res
    raw_parts, _, _ = paco_val(cfg, props, wl, None, chains=False)
    raw = recall_tables(raw_parts)
    out["diagnostic_pre_dedup"] = {
        "note": "recall of all stability-filtered SAM 3 masks before dedup; separates proposal misses from dedup "
                "loss. Not a gate quantity (the gate is measured on the final tree).",
        "part_recall": raw["part_recall"], "by_area_bin": raw["by_area_bin"]}
    g = out["by_tau"][str(cfg["v1"]["gate_tau"])]["paco"]
    th = cfg["v1"]["gate"]
    passed = g["part_recall"] >= th["part_recall"] and g.get("feasible_path_coverage", 0.0) >= th["feasible_path_coverage"]
    out["gate"] = {
        "passed": bool(passed), "tau_d": cfg["v1"]["gate_tau"], "part_recall": g["part_recall"],
        "feasible_path_coverage": g.get("feasible_path_coverage"), "thresholds": th,
        "consequence": None if passed else "gate failed: report ceilings; attention-seeded re-proposal inside "
                                           "parents (spec 5, V1) requires approval before any further stage",
    }
    ensure_deviation_log(cfg)
    d2 = read_deviations(cfg).get("D2", {}).get("status") == "approved"
    if not passed and d2:
        out["gate"]["continuation"] = "D2 approved: later stages run, labelled below_v1_gate"
    write_json(run / "metrics.json", out)
    print(run, "gate passed" if passed else "GATE FAILED" + (" (continuing under D2)" if d2 else ""),
          g["part_recall"], g.get("feasible_path_coverage"))
    if not passed and not args.smoke and not d2:
        sys.exit(2)


if __name__ == "__main__":
    main()
