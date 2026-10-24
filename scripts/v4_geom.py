import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import tree as T  # noqa: E402
from treeground.hyperbolic import (LorentzSelector, dist0, exterior_angle, half_aperture, make_selector,  # noqa: E402
                                   tree_edges)
from treeground.identity import manifest_inputs, open_features, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import (REPO, ProtocolError, load_config, read_json, require_approved, resolve_run,  # noqa: E402
                            start_run, v1_gate, write_json)

# D24: V4 natural search (incl. D12 no rejection) for a geometry-ablation selector (s1_geom_<arm>), plus geometry
# diagnostics on the PACO val containment trees. The V4 functions are reused; only the checkpoint loading differs
# (V4.load_selector builds the Euclidean Selector).


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_selector(cfg: dict, store):
    stage = cfg["inputs"]["selector_stage"]
    if not stage.startswith("s1_geom_"):
        raise ProtocolError(f"v4_geom evaluates s1_geom_* selectors, not {stage}")
    run = resolve_run(cfg, stage, cfg["inputs"].get(stage, "latest"), False)
    ckpt = torch.load(run / "selector.pt", map_location="cpu")
    if ckpt["smoke"]:
        raise ProtocolError("geometry checkpoint is a smoke checkpoint")
    if ckpt["features_identity"] != store.identity_hash:
        raise ProtocolError("geometry checkpoint was trained on a different feature cache")
    if ckpt["geometry"] != cfg["geometry"]:
        raise ProtocolError(f"checkpoint geometry {ckpt['geometry']} differs from the config's {cfg['geometry']}")
    sel = make_selector(ckpt["geometry"], ckpt["d_node"], ckpt["d_query"], ckpt["temperature"])
    sel.load_state_dict(ckpt["state_dict"])
    return sel.eval(), float(read_json(run / "metrics.json")["tau_rej"]), int(ckpt["layer"]), run


@torch.no_grad()
def diagnostics(cfg, sel, store, props, queries, layer, device, k: float) -> dict:
    if not isinstance(sel, LorentzSelector):
        return {"geometry": "euclidean", "note": "unit-norm cosine embeddings: no curvature, norm or cone"}
    c = sel.curv
    norms, depths, d_par, d_child, violated, excess = [], [], [], [], [], []
    shas = sorted(queries[(queries.split == "val") & queries.kind.isin(["object", "part", "chain"])].sha1.unique())
    for sha1 in shas:
        nodes = props.nodes(sha1, cfg["proposals"]["node_dedup"])
        z = store.load(sha1, [f"hidden_L{layer}"])[f"hidden_L{layer}"]
        if not len(z):
            continue
        tree = T.build_tree(nodes["rles"], device)
        e = sel.nodes(z.to(device))
        r = dist0(e, c).cpu().numpy()
        norms.append(r)
        depths.append(tree.depth[:tree.n])
        p, ch = tree_edges(tree.parent)
        if len(p):
            d_par.append(r[p])
            d_child.append(r[ch])
            pt, ct = torch.as_tensor(p, device=device), torch.as_tensor(ch, device=device)
            gap = (exterior_angle(e[pt], e[ct], c) - half_aperture(e[pt], c, k)).cpu().numpy()
            violated.append(gap > 0)
            excess.append(np.maximum(gap, 0))
    norms, depths = np.concatenate(norms), np.concatenate(depths)
    d_par, d_child = np.concatenate(d_par), np.concatenate(d_child)
    violated, excess = np.concatenate(violated), np.concatenate(excess)
    by_depth = {int(d): {"nodes": int((depths == d).sum()), "mean_norm": float(norms[depths == d].mean()),
                         "std_norm": float(norms[depths == d].std())} for d in np.unique(depths)}
    rho = spearmanr(depths, norms)
    return {"geometry": "lorentz", "images": len(shas), "nodes": int(len(norms)), "curvature": c.item(),
            "temperature": sel.tau, "offset": sel.offset.item(),
            "hyperbolic_norm": "geodesic distance from the origin, asinh(sqrt(c) |x_space|) / sqrt(c)",
            "mean_norm_by_depth": by_depth, "spearman_depth_norm": {"rho": float(rho.statistic),
                                                                    "p": float(rho.pvalue)},
            "edges": int(len(d_par)), "child_farther_than_parent": float((d_child > d_par).mean()),
            "cone_k": k, "cone_violation_rate": float(violated.mean()),
            "cone_mean_excess_angle": float(excess.mean()),
            "edge_rule": "parent -> child edges of the D3/D4 containment tree, virtual image root excluded"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D24"])
    V4 = load_script("v4_tree_value")
    gate = v1_gate(cfg, False)
    device = cfg["device"]
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    static = qwen_static(cfg)
    store = open_features(cfg, False, manifest, mhash, static)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, False, mhash, static)
    sel, tau, layer, sel_run = load_selector(cfg, store)
    sel = sel.to(device)
    li = static["layers"].index(layer)
    stage = cfg["inputs"]["selector_stage"]
    run = start_run(f"v4_{stage}", cfg, False, {"manifest_run": str(mrun), "selector_run": str(sel_run),
                                                "layer": layer, "tau_rej_chain": tau,
                                                "features_identity": store.identity_hash, "geometry": cfg["geometry"]})
    chains = V4.chain_inputs(cfg, sel, store, props, queries, qdir, li, layer, device)
    metrics = {"label": "dev", "split": "PACO-LVIS val supplied object->part chains (I1)", "smoke": False,
               "deviation": "D24", "geometry": cfg["geometry"], "selector_run": sel_run.name, "tau_rej_chain": tau,
               "seeds": cfg["v4"]["seeds"], "notes": [V4.I11_NOTE], "v1_gate": gate}
    metrics.update(V4.search_metrics(cfg, chains, tau, run))
    metrics["geometry_diagnostics"] = diagnostics(cfg, sel, store, props, queries, layer, device,
                                                  cfg["geometry"]["cone_k"])
    write_json(run / "metrics.json", metrics)
    print(run)


if __name__ == "__main__":
    main()
