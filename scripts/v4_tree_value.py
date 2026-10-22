import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import load_tensors  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import (manifest_inputs, open_features, open_proposals, open_queries,  # noqa: E402
                                 qwen_static)
from treeground.metrics import describe, mean_ci  # noqa: E402
from treeground.run import (ProtocolError, load_config, read_deviations, read_json, resolve_run,  # noqa: E402
                            start_run, v1_gate, write_json)  # noqa: E402
from treeground.selector import Selector  # noqa: E402

THR = 0.5
SINGLE_LEVEL = 0.5  # D5 split: |GT part| >= 0.5 |GT object| makes the object's own mask a valid part answer
I11_NOTE = ("confound (I11): the V2 checkpoint was trained with chain negatives drawn from the true subtree, "
            "so the true-tree row is evaluated under the candidate distribution it was trained on")


def tree_seed(seed: int, sha1: str) -> int:
    return int(np.random.SeedSequence([seed, zlib.crc32(sha1.encode())]).generate_state(1)[0])


def decide(sig: np.ndarray, pool: np.ndarray, tau: float) -> int | None:
    if len(pool) == 0:
        return None
    a = int(pool[np.argmax(sig[pool])])  # pool is sorted, so ties resolve to the lower node index
    return a if sig[a] >= tau else None


def load_selector(cfg: dict, smoke: bool, store) -> tuple[Selector, float, int, Path]:
    run = resolve_run(cfg, "v2", cfg["inputs"].get("v2", "latest"), smoke)
    ckpt = torch.load(run / "selector.pt", map_location="cpu")
    if ckpt["smoke"] != smoke:
        raise ProtocolError("V2 checkpoint smoke flag does not match the current mode")
    if ckpt["features_identity"] != store.identity_hash:
        raise ProtocolError("V2 checkpoint was trained on a different feature cache")
    sel = Selector(ckpt["d_node"], ckpt["d_query"], ckpt["temperature"])
    sel.load_state_dict(ckpt["state_dict"])
    tau = read_json(run / "metrics.json")["tau_rej"]
    return sel.eval(), float(tau), int(ckpt["layer"]), run


@torch.no_grad()
def chain_inputs(cfg, sel, store, props, queries, qdir, li, layer, device):
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    seeds = cfg["v4"]["seeds"]
    out = []
    chains = queries[(queries.split == "val") & (queries.kind == "chain")]
    for sha1, g in chains.groupby("sha1"):
        key = g.image_key.iloc[0]
        img = images[int(key.split(":")[1])]
        nodes = props.nodes(sha1, cfg["proposals"]["node_dedup"])
        z = store.load(sha1, [f"hidden_L{layer}"])[f"hidden_L{layer}"]
        tree = T.build_tree(nodes["rles"])
        rand = [T.randomised_tree(tree, tree_seed(s, sha1)) for s in seeds]
        e = sel.nodes(z.to(device)) if len(z) else None
        probes = load_tensors(qdir / "probes" / f"{sha1}.safetensors", ["probes"])[0]["probes"]
        gts = paco.gt_rles(img)
        for q in g.itertuples():
            gt = json.loads(q.gt)
            step_ious = M.iou(nodes["rles"], [gts[a] for a in gt]).T if nodes["rles"] else np.zeros((2, 0))
            if e is None:
                sig = np.zeros((2, 0))
            else:
                qv = sel.queries(probes[q.offset:q.offset + 2, li].to(device))
                sig = torch.sigmoid(sel.scores(qv, e)).cpu().numpy()
            ga = M.areas([gts[a] for a in gt])
            out.append({"qid": q.qid, "image": key, "sha1": sha1, "tree": tree, "rand": rand,
                        "ious": step_ious, "sig": sig, "part_ratio": float(ga[1] / max(1, ga[0]))})
    return out


def feasible_inclusive(tree: T.Tree, ious: np.ndarray) -> bool:
    # D5: some object hit o and part hit p with p == o or p a strict descendant of o
    hits = T.step_candidates(ious, THR)
    return any(p == o or tree.is_strict_descendant(p, o) for o in hits[0] for p in hits[1])


def natural(chains: list[dict], tau: float, inclusive: bool = False) -> tuple[pd.DataFrame, list[int]]:
    rows, anchors = [], []
    for c in chains:
        n = c["ious"].shape[1]
        tree = c["tree"]
        a1 = decide(c["sig"][0], np.arange(n), tau) if n else None
        u1 = tree.root if a1 is None else a1
        s1 = a1 is not None and c["ious"][0, a1] >= THR
        anchors.append(len(tree.strict_descendants(u1)) if a1 is not None else -1)
        structures = [("flat", None), ("true", tree)] + [(f"rand{k}", t) for k, t in enumerate(c["rand"])]
        best2 = int(np.argmax(c["ious"][1])) if n else -1
        for name, t in structures:
            pool = np.arange(n) if t is None else t.strict_descendants(u1)
            if inclusive and t is not None and a1 is not None:
                pool = np.sort(np.append(pool, a1))
            a2 = decide(c["sig"][1], pool, tau)
            s2 = a2 is not None and c["ious"][1, a2] >= THR
            if t is None:
                feasible = T.flat_feasible(c["ious"])
            elif inclusive:
                feasible = n > 0 and feasible_inclusive(t, c["ious"])
            else:
                feasible = n > 0 and T.max_iou_path(t, c["ious"]) is not None
            rows.append({"qid": c["qid"], "image": c["image"], "structure": name, "feasible": feasible,
                         "single_level": c.get("part_ratio", 0.0) >= SINGLE_LEVEL,
                         "step1": bool(s1), "step2": bool(s2), "chain": bool(s1 and s2), "candidates": len(pool),
                         "best_id_step2": bool(a2 is not None and a2 == best2 and c["ious"][1, best2] >= THR),
                         "best_id_step1": bool(a1 is not None and a1 == int(np.argmax(c["ious"][0]))
                                               and c["ious"][0].max() >= THR)})
    return pd.DataFrame(rows), anchors


def count_controlled(chains: list[dict], tau: float, seeds: list[int], B: int) -> dict:
    res = {}
    for s_i, seed in enumerate(seeds):
        acc = {k: [] for k in ("flat", "true", "rand")}
        cov = {k: [] for k in ("flat", "true", "rand")}
        empty = 0
        for c_i, c in enumerate(chains):
            n = c["ious"].shape[1]
            a1 = decide(c["sig"][0], np.arange(n), tau) if n else None
            u1 = c["tree"].root if a1 is None else a1
            pools = {"flat": np.arange(n), "true": c["tree"].strict_descendants(u1),
                     "rand": c["rand"][s_i].strict_descendants(u1)}
            b = min(B, *(len(p) for p in pools.values()))
            if b == 0:
                empty += 1
                continue
            for k_i, (name, pool) in enumerate(pools.items()):
                rng = np.random.default_rng([seed, c_i, k_i])
                sample = S.pre_scoring_sample(pool, b, rng)
                a2 = decide(c["sig"][1], sample, tau)
                acc[name].append(bool(a2 is not None and c["ious"][1, a2] >= THR))
                cov[name].append(bool((c["ious"][1, sample] >= THR).any()))
        res[seed] = {"step2_success": {k: float(np.mean(v)) if v else None for k, v in acc.items()},
                     "sampled_pool_target_coverage": {k: float(np.mean(v)) if v else None for k, v in cov.items()},
                     "empty_pools": empty, "scored_chains": len(acc["flat"])}
    summary = {}
    for metric in ("step2_success", "sampled_pool_target_coverage"):
        summary[metric] = {}
        for k in ("flat", "true", "rand"):
            vals = [res[s][metric][k] for s in seeds if res[s][metric][k] is not None]
            summary[metric][k] = {"mean": float(np.mean(vals)) if vals else None,
                                  "std": float(np.std(vals)) if vals else None, "per_seed": vals}
    summary["empty_pools"] = {"mean": float(np.mean([res[s]["empty_pools"] for s in seeds])),
                              "per_seed": [res[s]["empty_pools"] for s in seeds]}
    summary["b_cap_B"] = B
    summary["per_seed"] = res
    return summary


def natural_summary(df: pd.DataFrame, n_rand: int, bs: dict) -> dict:
    wide = df.pivot(index="qid", columns="structure", values="feasible")
    common = wide.all(axis=1)
    out = {"chains": int(len(wide)), "common_eligible_subset": {"size": int(common.sum()),
                                                                "fraction": float(common.mean()) if len(wide) else None}}
    images = df.drop_duplicates("qid").set_index("qid")["image"]
    rand_names = [f"rand{k}" for k in range(n_rand)]

    def per_chain(col, structures, subset=None):
        sub = df[df.structure.isin(structures)]
        v = sub.groupby("qid")[col].mean()
        if subset is not None:
            v = v[subset.reindex(v.index).fillna(False).astype(bool)]
        return v

    for name, structs in (("flat", ["flat"]), ("true", ["true"]), ("random", rand_names)):
        c = per_chain("chain", structs, common)
        s2 = per_chain("step2", structs, common)
        e2e = per_chain("chain", structs)
        feas = per_chain("feasible", structs)
        cand = df[df.structure.isin(structs)]["candidates"]
        out[name] = {
            "chain_success_common": mean_ci(c.to_numpy(), images[c.index].to_numpy(), bs["n"], bs["seed"]),
            "step2_success_common": mean_ci(s2.to_numpy(), images[s2.index].to_numpy(), bs["n"], bs["seed"]),
            "chain_success_end_to_end": mean_ci(e2e.to_numpy(), images[e2e.index].to_numpy(), bs["n"], bs["seed"]),
            "feasible_path_coverage": float(feas.mean()) if len(feas) else None,
            "candidate_count_step2": describe(cand.to_numpy()),
            "diagnostic_best_proposal_identification_step2": float(per_chain("best_id_step2", structs).mean())
            if len(df) else None,
        }
        if name == "random":
            out[name]["per_tree"] = {r: {"chain_success_common": float(per_chain("chain", [r], common).mean()),
                                         "feasible_path_coverage": float(per_chain("feasible", [r]).mean())}
                                     for r in rand_names}
    out["diagnostic_best_proposal_identification_step1"] = float(per_chain("best_id_step1", ["flat"]).mean()) \
        if len(df) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gate = v1_gate(cfg, args.smoke)
    device = cfg["device"]
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    store = open_features(cfg, args.smoke, manifest, mhash, static)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    sel, tau, layer, v2_run = load_selector(cfg, args.smoke, store)
    sel = sel.to(device)
    li = static["layers"].index(layer)
    run = start_run("v4", cfg, args.smoke, {"manifest_run": str(mrun), "v2_run": str(v2_run), "layer": layer,
                                             "tau_rej_chain": tau, "features_identity": store.identity_hash})
    seeds = cfg["v4"]["seeds"]
    chains = chain_inputs(cfg, sel, store, props, queries, qdir, li, layer, device)
    df, anchors = natural(chains, tau)
    df.to_csv(run / "chains_natural.csv", index=False)
    real_anchors = [a for a in anchors if a >= 0]
    B = int(np.median(real_anchors)) if real_anchors else 0
    metrics = {
        "label": "dev", "split": "PACO-LVIS val supplied object->part chains (I1)", "smoke": args.smoke,
        "tau_rej_chain": tau, "seeds": seeds,
        "natural_search": natural_summary(df, len(seeds), cfg["v4"]["bootstrap"]),
        "count_controlled_diagnostic": count_controlled(chains, tau, seeds, B),
        "B_definition": "median |sub+(u1)| over PACO val chains whose step-1 controller anchor is a node",
        "other_sources": {"partonomy": "not evaluated: S0 caches no probe states for PARTONOMY chains",
                          "sceneparser": "not evaluated: S0 caches no probe states for SceneParser chains"},
        "notes": [I11_NOTE],
        "v1_gate": gate,
    }
    bs = cfg["v4"]["bootstrap"]
    single = df.drop_duplicates("qid").set_index("qid")["single_level"]
    metrics["chain_split_spec_strict"] = {
        "rule": f"single-level if |GT part| >= {SINGLE_LEVEL} |GT object|",
        "single_level_chains": int(single.sum()), "two_level_chains": int((~single).sum()),
        "two_level": natural_summary(df[~df.single_level], len(seeds), bs),
        "single_level": natural_summary(df[df.single_level], len(seeds), bs) if single.any() else None}
    if read_deviations(cfg).get("D5", {}).get("status") == "approved":
        d5, _ = natural(chains, tau, inclusive=True)
        d5.to_csv(run / "chains_natural_d5.csv", index=False)
        metrics["natural_search_d5"] = {
            "deviation": "D5: step-2 candidates are the step-1 node and its strict descendants",
            "all": natural_summary(d5, len(seeds), bs),
            "two_level": natural_summary(d5[~d5.single_level], len(seeds), bs),
            "single_level": natural_summary(d5[d5.single_level], len(seeds), bs) if single.any() else None}
    else:
        metrics["natural_search_d5"] = "not computed: deviation D5 is not approved in deviations.jsonl"
    write_json(run / "metrics.json", metrics)
    print(run)


if __name__ == "__main__":
    main()
