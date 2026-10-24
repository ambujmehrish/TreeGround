import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.hyperbolic import LorentzSelector, cone_violation, make_selector, tree_edges  # noqa: E402
from treeground.identity import manifest_inputs, open_features, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import (REPO, canonical_hash, load_config, require_approved, start_run, v1_gate,  # noqa: E402
                            write_json)

# D24: selector-geometry ablation on the S1 setting. Identical to scripts/s1_train.py (same caches, V2 recipe, seed,
# steps, checkpoint selection on PACO val) except the selector geometry (Euclidean cosine or Lorentz, MERU) and an
# optional entailment-cone loss on the parent -> child edges of each training image's containment tree.


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def optimizer(sel, c: dict) -> torch.optim.Optimizer:
    if not isinstance(sel, LorentzSelector):
        return torch.optim.AdamW(sel.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    # MERU: no weight decay on the learnable scalars (alpha, curvature, temperature, offset)
    scalars = {id(p) for p in sel.scalar_parameters()}
    heads = [p for p in sel.parameters() if id(p) not in scalars]
    return torch.optim.AdamW([{"params": heads, "weight_decay": c["weight_decay"]},
                              {"params": sel.scalar_parameters(), "weight_decay": 0.0}], lr=c["lr"])


def cone_loss(sel: LorentzSelector, batch: list, device: str, k: float) -> torch.Tensor:
    parents, children, off = [], [], 0
    for r, _ in batch:
        p, ch = tree_edges(r.tree.parent)
        parents.append(p + off)
        children.append(ch + off)
        off += r.n
    p, ch = np.concatenate(parents), np.concatenate(children)
    if not len(p):
        return torch.zeros((), device=device, dtype=torch.float64)
    e = sel.nodes(torch.cat([r.z for r, _ in batch]).to(device))
    p, ch = torch.as_tensor(p, device=device), torch.as_tensor(ch, device=device)
    return cone_violation(e[p], e[ch], sel.curv, k).mean()


def geometry_state(sel) -> dict:
    if not isinstance(sel, LorentzSelector):
        return {"tau": sel.tau}
    return {"tau": sel.tau, "curvature": sel.curv.item(), "offset": sel.offset.item(),
            "alpha_node": sel.log_alpha_g.exp().item(), "alpha_query": sel.log_alpha_f.exp().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D11", "D24"])
    geom = cfg["geometry"]
    device = cfg["device"]
    V2, S1, E2E = load_script("v2_train_eval"), load_script("s1_train"), load_script("e2e_variants")
    gate = v1_gate(cfg, False)
    layer, v3_run = V2.chosen_layer(cfg, False)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    static = qwen_static(cfg)
    li = static["layers"].index(layer)
    store = open_features(cfg, False, manifest, mhash, static)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, False, mhash, static)
    probes = V2.Probes(qdir, li)
    text = {r.label: probes.get(V2.TEXT_ONLY, r.offset, 1)[0] for r in queries[queries.kind == "text"].itertuples()}
    recs, absent, counts, used = S1.load_records(cfg, V2, layer)
    run = start_run(f"s1_geom_{geom['arm']}", cfg, False, {
        "manifest_run": str(mrun), "features_identity": store.identity_hash, "layer": layer, "v3_run": v3_run,
        "caches": used, "v1_run": gate["v1_run"], "geometry": geom})
    torch.backends.cuda.matmul.allow_tf32 = False
    c, s1 = cfg["v2"], cfg["s1"]
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    d_node, d_query = recs[0].z.shape[1], next(iter(text.values())).shape[0]
    sel = make_selector(geom, d_node, d_query, c["temperature"]).to(device)
    w_cone = float(geom["cone_weight"])
    if w_cone and not isinstance(sel, LorentzSelector):
        raise ValueError("the entailment-cone loss needs the Lorentz selector")
    opt = optimizer(sel, c)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, V2.lr_lambda(c["warmup_steps"], s1["steps"]))
    items = [r for r in recs if r.queries] + absent
    n_text = int(round(c["text_only_fraction"] * c["batch_images"]))
    order, ptr = rng.permutation(len(items)), 0
    log, evals, best, t0 = [], [], None, time.time()
    print(f"S1 geometry {geom['arm']}: {len(items)} training items ({len(absent)} no-target), {s1['steps']} steps",
          flush=True)
    for step in range(1, s1["steps"] + 1):
        if ptr + c["batch_images"] > len(order):
            order, ptr = rng.permutation(len(items)), 0
        idx = order[ptr:ptr + c["batch_images"]]
        ptr += c["batch_images"]
        batch, left = [], n_text
        for i in idx:
            r = items[i]
            as_text = left > 0 and not r.absent and any(lab and lab in text for lab in r.labels)
            left -= int(as_text)
            batch.append((r, as_text))
        stats = {k: 0 for k in ["terms_object", "terms_part", "terms_chain", "terms_text", "terms_absent",
                                "empty_negatives", "shortfall_in_image", "shortfall_cross", "chain_topup"]}
        sel.train()
        sel_loss = torch.stack(V2.batch_terms(sel, batch, text, rng, device, c["gamma"], stats)).mean()
        cone = cone_loss(sel, batch, device, geom["cone_k"]) if w_cone else None
        loss = sel_loss + w_cone * cone.float() if w_cone else sel_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % c["log_every"] == 0:
            log.append({"step": step, "loss": loss.item(), "selection_loss": sel_loss.item(),
                        "cone_loss": cone.item() if w_cone else None, "lr": sched.get_last_lr()[0],
                        "seconds": time.time() - t0, **geometry_state(sel), **stats})
        if step % 200 == 0:
            print(f"step {step}/{s1['steps']} loss {loss.item():.4f} {geometry_state(sel)} {time.time() - t0:.0f}s",
                  flush=True)
        if step % s1["eval_every"] == 0 or step == s1["steps"]:
            sel.eval()
            metrics, per_query, tau = V2.evaluate(cfg, sel, store, props, queries, qdir, li, layer, wl, device)
            score = metrics["ii_selector_success"]["all"]["mean"]
            evals.append({"step": step, "selector_success": score, "tau_rej": tau, **geometry_state(sel)})
            print(f"eval step {step}: PACO val selector success {score:.4f} tau_rej {tau:.3f}", flush=True)
            if best is None or score > best[0]:
                best = (score, step, metrics)
                per_query.to_csv(run / "eval_queries.csv", index=False)
                torch.save({"state_dict": sel.state_dict(), "d_node": d_node, "d_query": d_query, "layer": layer,
                            "temperature": sel.tau, "geometry": geom, "smoke": False,
                            "features_identity": store.identity_hash, "step": step,
                            "config_hash": canonical_hash({"v2": cfg["v2"], "s1": cfg["s1"], "geometry": geom})},
                           run / "selector.pt")
            pd.DataFrame(evals).to_csv(run / "checkpoints.csv", index=False)
    pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
    score, step, metrics = best
    # D15 definitions on the selected checkpoint's per-query table: calibrated and rejection disabled
    variants, curve = E2E.variants(pd.read_csv(run / "eval_queries.csv"), metrics["tau_rej"], c["bootstrap"])
    curve.to_csv(run / "threshold_curve.csv", index=False)
    metrics.update({"stage": f"S1 geometry ablation ({geom['arm']})", "deviation": "D24", "geometry": geom,
                    "selected_step": step, "geometry_state_selected": next(e for e in evals if e["step"] == step),
                    "checkpoint_rule": "highest PACO val single-step selector success (V2 definition)",
                    "evals": evals, "no_rejection_present": variants["no_rejection_present"],
                    "calibrated_variants": variants["calibrated"], "training_items": len(items),
                    "training_no_target": len(absent), "cache_counts": counts, "sources": cfg["s1"]["sources"],
                    "caches": used, "v1_gate": gate})
    write_json(run / "metrics.json", metrics)
    print(run, "selected step", step, "selector success", score, "tau_rej", metrics["tau_rej"])


if __name__ == "__main__":
    main()
