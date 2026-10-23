import argparse
import glob
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.identity import manifest_inputs, open_features, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import (REPO, ProtocolError, canonical_hash, load_config, read_json, require_approved,  # noqa: E402
                            start_run, v1_gate, write_json)
from treeground.selector import Selector  # noqa: E402

# S1 (spec 2.4) at full scale: the V2 recipe (same selector, loss, negative sampling, 20% text-only samples,
# gRefCOCO no-target as the only all-negative examples) trained on the packed S1 caches of all available train
# sources, with checkpoint selection on PACO val (single-step selection success, as V2 reports it) and the
# rejection threshold calibrated by the V2 rule at the selected checkpoint. Deviation D11.


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cache_dirs(cfg: dict, source: str) -> list[Path]:
    root = Path(cfg["cache_root"]) / "full" / "s1" / source
    return sorted(p for p in root.glob("*") if (p / "identity.json").exists())


def load_records(cfg: dict, V2, layer: int):
    recs, absent, counts, used = [], [], {}, {}
    for source in cfg["s1"]["sources"]:
        dirs = cache_dirs(cfg, source)
        if len(dirs) != 1:
            raise ProtocolError(f"expected exactly one S1 cache for {source}, found {[str(d) for d in dirs]}")
        d = dirs[0]
        files = sorted(glob.glob(str(d / "shards" / "*.pt")))
        n = cfg["s1"]["num_shards"][source]
        if len(files) != n:
            raise ProtocolError(f"{source}: {len(files)} of {n} S1 shards present in {d}")
        used[source] = {"cache": d.name, "identity": read_json(d / "identity.json")["hash"]}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for blob in pool.map(lambda f: torch.load(f, map_location="cpu", weights_only=False), files):
                if blob["layer"] != layer:
                    raise ProtocolError(f"{source}: cache layer {blob['layer']} differs from the chosen layer {layer}")
                for k, v in blob["counts"].items():
                    counts[f"{source}:{k}"] = counts.get(f"{source}:{k}", 0) + v
                for r in blob["records"]:
                    tree = T.from_parents(np.asarray(r["parent"], dtype=np.int64))
                    rec = V2.ImageRec(r["key"], r["key"], r["z"], tree, r["labels"], r["parts"], r["high"])
                    for q in r["queries"]:
                        steps = [S.StepTarget(np.asarray(p, np.int64), np.asarray(i, np.int64)) for p, i in q["steps"]]
                        rec.queries.append(V2.QRec(q["qid"], q["kind"], q["probes"], steps, q["path"]))
                    recs.append(rec)
                    for a in r["absent"]:
                        arec = V2.ImageRec(r["key"], r["key"], r["z"], tree, r["labels"], r["parts"], r["high"],
                                           absent=True)
                        arec.queries = [V2.QRec(a["qid"], "absent", a["probes"],
                                                [S.StepTarget(np.zeros(0, np.int64), np.zeros(0, np.int64))])]
                        absent.append(arec)
        print(f"loaded {source}: {len(files)} shards, records so far {len(recs)}, absent {len(absent)}", flush=True)
    return recs, absent, counts, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D11"])
    device = cfg["device"]
    V2 = load_script("v2_train_eval")
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
    recs, absent, counts, used = load_records(cfg, V2, layer)
    run = start_run("s1", cfg, False, {"manifest_run": str(mrun), "features_identity": store.identity_hash,
                                       "layer": layer, "v3_run": v3_run, "caches": used, "v1_run": gate["v1_run"]})
    torch.backends.cuda.matmul.allow_tf32 = False
    c, s1 = cfg["v2"], cfg["s1"]
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    d_node, d_query = recs[0].z.shape[1], next(iter(text.values())).shape[0]
    sel = Selector(d_node, d_query, c["temperature"]).to(device)
    opt = torch.optim.AdamW(sel.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, V2.lr_lambda(c["warmup_steps"], s1["steps"]))
    items = [r for r in recs if r.queries] + absent
    n_text = int(round(c["text_only_fraction"] * c["batch_images"]))
    order, ptr = rng.permutation(len(items)), 0
    log, evals, best, t0 = [], [], None, time.time()
    print(f"S1: {len(items)} training items ({len(absent)} no-target), {s1['steps']} steps", flush=True)
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
        loss = torch.stack(V2.batch_terms(sel, batch, text, rng, device, c["gamma"], stats)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % c["log_every"] == 0:
            log.append({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0], "seconds": time.time() - t0,
                        **stats})
        if step % 200 == 0:
            print(f"step {step}/{s1['steps']} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % s1["eval_every"] == 0 or step == s1["steps"]:
            sel.eval()
            metrics, per_query, tau = V2.evaluate(cfg, sel, store, props, queries, qdir, li, layer, wl, device)
            score = metrics["ii_selector_success"]["all"]["mean"]
            evals.append({"step": step, "selector_success": score, "tau_rej": tau})
            print(f"eval step {step}: PACO val selector success {score:.4f} tau_rej {tau:.3f}", flush=True)
            if best is None or score > best[0]:
                best = (score, step, metrics)
                per_query.to_csv(run / "eval_queries.csv", index=False)
                torch.save({"state_dict": sel.state_dict(), "d_node": d_node, "d_query": d_query, "layer": layer,
                            "temperature": sel.tau, "smoke": False, "features_identity": store.identity_hash,
                            "config_hash": canonical_hash({"v2": cfg["v2"], "s1": cfg["s1"]}), "step": step},
                           run / "selector.pt")
            pd.DataFrame(evals).to_csv(run / "checkpoints.csv", index=False)
    pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
    score, step, metrics = best
    metrics.update({"stage": "S1", "deviation": "D11", "selected_step": step, "checkpoint_rule":
                    "highest PACO val single-step selector success (V2 definition)", "evals": evals,
                    "training_items": len(items), "training_no_target": len(absent), "cache_counts": counts,
                    "sources": cfg["s1"]["sources"], "caches": used, "v1_gate": gate})
    write_json(run / "metrics.json", metrics)
    print(run, "selected step", step, "selector success", score, "tau_rej", metrics["tau_rej"])


if __name__ == "__main__":
    main()
