import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_s2 import build_scene, load_script, wrapped
from treeground.mllm import SEL

LABELS = {"bottle:cap", "mug:handle"}


def config(tmp_path, variant):
    return {"device": "cpu", "proposals": {"node_dedup": 0.9}, "inputs": {},
            "models": {"qwen": {"repo": "tiny", "revision": "0"}}, "v4": {"seeds": [0]},
            "ta": {"variant": variant, "readout_layers": [2, 4], "seed": 0, "steps": 3, "batch_images": 3,
                   "micro_batch": 3, "text_only_fraction": 0.34, "gamma": 2.0, "head_lr": 1e-3,
                   "head_weight_decay": 0.01, "lora_lr": 1e-3, "lora_weight_decay": 0.0, "warmup_steps": 1,
                   "max_grad_norm": 1.0, "lora": {"r": 4, "alpha": 8, "dropout": 0.05,
                                                  "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                              "up_proj", "down_proj"]},
                   "sources": {"paco_train": 1.0}, "paco_rest": False, "eval_every": 2, "eval_images": 2,
                   "eval_batch": 2, "workers": 0, "log_every": 1},
            "ta_eval": {"checkpoint": "best.pt", "batch_size": 2}}


def patch_common(mod, monkeypatch, s2t, tmp_path, cfg, items, props):
    # s2t: the s2_train module object that `mod` imported (patches must land there)
    model, proc, extra = wrapped(s2t)
    monkeypatch.setattr(mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(mod, "require_approved", lambda cfg, ids: None)
    monkeypatch.setattr(mod, "v1_gate", lambda cfg, smoke: {"v1_gate_passed": True})
    monkeypatch.setattr(mod, "qwen_static", lambda cfg: None)
    monkeypatch.setattr(mod, "open_proposals", lambda *a, **k: props)
    monkeypatch.setattr(s2t, "open_proposals", lambda *a, **k: props)     # Source opens its own store lazily
    monkeypatch.setattr(s2t, "build_model", lambda cfg, c, device, static: (model, proc, extra))
    runs = tmp_path / "runs"

    def start_run(stage, cfg, smoke, meta):
        d = runs / stage
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"smoke": smoke}))
        return d

    monkeypatch.setattr(mod, "start_run", start_run)
    return runs


def query_table(items):
    rows = []
    for m in items:
        for q in m.queries:
            text = f"{q.user} {SEL}" if q.kind == "absent" else q.user
            rows.append({"qid": q.qid, "kind": q.kind, "text": text, "sha1": m.sha1, "split": "val",
                         "image_key": m.key, "n_sel": q.answer.count(SEL), "label": None})
    rows += [{"qid": f"text|{lab}", "kind": "text", "text": "", "sha1": "textonly", "split": "any",
              "image_key": None, "n_sel": 1, "label": lab} for lab in sorted(LABELS)]
    queries = pd.DataFrame(rows).sort_values(["sha1", "qid"]).reset_index(drop=True)
    queries["offset"] = queries.groupby("sha1")["n_sel"].cumsum() - queries["n_sel"]
    return queries


@pytest.mark.parametrize("variant", ["flat", "full", "tree", "random_tree"])
def test_train_then_eval_main(tmp_path, monkeypatch, variant):
    s2t, tat = load_script("s2_train"), load_script("ta_train")
    items, props = build_scene(s2t, tmp_path)
    cfg = config(tmp_path, variant)
    queries = query_table(items)
    wl = pd.DataFrame({"sha1": [m.sha1 for m in items], "path": [m.file for m in items],
                       "image_key": [m.key for m in items]})
    runs = patch_common(tat, monkeypatch, tat.S2T, tmp_path, cfg, items, props)
    monkeypatch.setattr(tat, "ensure_deviation_log", lambda cfg: None)
    monkeypatch.setattr(tat, "manifest_inputs", lambda cfg, smoke: ("m", None, "h", wl))
    monkeypatch.setattr(tat, "open_queries", lambda *a: ("q", queries))
    monkeypatch.setattr(tat, "node_cap", lambda cfg, smoke: {"k": 5, "order": "stability"})
    monkeypatch.setattr(tat.S2T, "all_train_meta", lambda *a: (items, {"queries": 7}))
    vm = [s2t.ImageMeta(m.sha1, m.key, m.file, m.n, m.parents, None, None, None,
                        [s2t.Query(q.qid, q.kind, q.user, q.answer, [], None, np.linspace(0, 1, m.n))
                         for q in m.queries if q.kind != "chain"]) for m in items[:2]]
    monkeypatch.setattr(tat.S2T, "val_meta", lambda *a: vm)
    monkeypatch.setattr(sys, "argv", ["ta_train.py", "--config", "x"])
    tat.main()
    run = runs / f"ta_{variant}"
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["variant"] == variant and metrics["best_layer"] in (2, 4)
    assert [e["step"] for e in metrics["eval_curve"]] == [0, 2, 3]
    assert metrics["training_exclusions"]["trees_built_for_no_target_items"] == 1
    assert (run / "best.pt").exists() and len(pd.read_csv(run / "train_curve.csv")) == 3
    best = torch.load(run / "best.pt")
    assert best["meta"]["variant"] == variant and best["meta"]["step"] == metrics["best_step"]

    # evaluation of that checkpoint through the unchanged V2 / V4 interfaces (evaluators replaced by probes)
    tae = load_script("ta_eval")
    patch_common(tae, monkeypatch, tae.TAT.S2T, tmp_path / "e", cfg, items, props)
    monkeypatch.setattr(tae, "manifest_inputs", lambda cfg, smoke: ("m", None, "h", wl))
    monkeypatch.setattr(tae, "open_queries", lambda *a: ("q", queries))
    monkeypatch.setattr(tae, "resolve_run", lambda cfg, stage, ref, smoke: run)
    seen = {}

    def evaluate(cfg, sel, store, props_, queries_, qdir, li, layer, wl_, device):
        probes = tae.V2.Probes(qdir, li)
        for m in items:
            e = sel.nodes(store.load(m.sha1, [f"hidden_L{layer}"])[f"hidden_L{layer}"])
            for q in queries_[queries_.sha1 == m.sha1].itertuples():
                seen[q.qid] = sel.scores(sel.queries(probes.get(m.sha1, q.offset, q.n_sel)), e)
        df = pd.DataFrame([{"qid": k, "kind": "part", "present": True, "sel_iou": 0.6, "best_iou": 0.7}
                           for k in seen])
        return {"stub": True}, df, 0.5

    monkeypatch.setattr(tae.V2, "evaluate", evaluate)
    monkeypatch.setattr(tae.V4, "chain_inputs", lambda *a: [])
    monkeypatch.setattr(tae.V4, "search_metrics", lambda cfg, chains, tau, run: {"chains": len(chains)})
    monkeypatch.setattr(sys, "argv", ["ta_eval.py", "--config", "x"])
    tae.main()
    ev = json.loads((tmp_path / "e" / "runs" / f"ta_{variant}_eval" / "metrics.json").read_text())
    assert ev["layer"] == best["meta"]["layer"] and ev["no_rejection"]["success@0.5"] == 1.0
    assert len(seen) == sum(len(m.queries) for m in items)
    for m in items:
        for q in m.queries:
            assert seen[q.qid].shape == (q.answer.count(SEL), m.n)
    if variant == "flat":
        return
    # relation swap: same checkpoint, node tokens see only themselves; own stage, different scores
    trained = {k: v.clone() for k, v in seen.items()}
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["ta_eval.py", "--config", "x", "--relations", "self"])
    tae.main()
    sw = json.loads((tmp_path / "e" / "runs" / f"ta_{variant}_eval_rel_self" / "metrics.json").read_text())
    assert sw["relations"] == "self" and sw["layer"] == ev["layer"] and ev["relations"] == "as trained"
    assert any(not torch.allclose(seen[k], trained[k]) for k in trained)
