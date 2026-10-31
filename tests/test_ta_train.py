import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_s2 import build_scene, load_script, wrapped
from treeground import lora as LR
from treeground import tree as T
from treeground import tree_attn as TA
from treeground.cache import load_tensors, save_tensors
from treeground.mllm import SEL

LAYERS = [2, 4]
LABELS = {"bottle:cap", "mug:handle"}


@pytest.fixture(scope="module")
def s2t():
    return load_script("s2_train")


@pytest.fixture(scope="module")
def tat():
    return load_script("ta_train")


@pytest.fixture(scope="module")
def tae():
    return load_script("ta_eval")


def make(s2t, tat, tmp_path, variant, seed=0):
    items, props = build_scene(s2t, tmp_path)
    assert tat.add_missing_trees(items, props, 0.9, None) == 1          # the no-target item had no tree
    model, proc, extra = wrapped(s2t, seed)
    torch.manual_seed(seed)
    embed = None if variant == "flat" else TA.NodeEmbed(64, TA.embedding_rms(model))
    readout = tat.Readout(64, LAYERS, 5, embed, variant in tat.HIER)
    src = tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], LABELS, variant, 0)
    src._props = props
    return items, props, model, proc, extra, readout, src


@pytest.mark.parametrize("variant", ["flat", "full", "tree", "random_tree", "hier", "hier_random_tree"])
def test_training_step_every_variant(s2t, tat, tmp_path, variant):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, variant)
    model.train()
    c = {"batch_images": 3, "micro_batch": 3, "seed": 0, "text_only_fraction": 0.34, "steps": 1}
    entries = s2t.plan(items, c, LABELS, 1, 0)
    mb = tat.MicroBatches(entries, items, src)[0]
    assert len(mb["seqs"]) == 4 and all(s["nodes"].n == items[s["item"]].n for s in mb["samples"])
    stats = {k: 0 for k in s2t.STAT_KEYS}
    l_sel, l_lm = tat.micro_step(model, extra, readout, mb, items, "cpu", np.random.default_rng(0), 2.0, stats)
    assert torch.isfinite(l_sel) and torch.isfinite(l_lm) and l_lm > 0
    assert stats["terms_text"] == 1 and stats["terms_absent"] == 1              # counted once, not per layer
    (l_sel + l_lm).backward()
    params = LR.trainable_parameters(model, extra["sel_emb"]) + list(readout.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    stops = [readout.head(k) for k in LAYERS]
    assert all(h.bias.grad.abs() > 0 and h.log_scale.grad is not None for h in stops)
    if variant in tat.HIER:                                              # descend and root heads are trained
        assert all(h.desc.bias.grad.abs() > 0 and h.root[2].bias.grad.abs() > 0 for h in readout.heads)
    assert any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith("lora_B"))
    if variant != "flat":
        assert readout.embed.mix[0].weight.grad.abs().sum() > 0
    s2t.sync_grads(params, 1)
    torch.optim.AdamW(params, lr=1e-3).step()
    assert all(torch.isfinite(p).all() for p in params)


def val_metas(s2t, items):
    return [s2t.ImageMeta(m.sha1, m.key, m.file, m.n, m.parents, None, None, None,
                          [s2t.Query(q.qid, q.kind, q.user, q.answer, [], None, np.linspace(0, 1, m.n))
                           for q in m.queries if q.kind != "chain"]) for m in items[:2]]


def test_val_success_reports_every_layer(s2t, tat, tmp_path):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    res = tat.val_success(model, readout, src, val_metas(s2t, items), "cpu", 1, 2)
    assert set(res["by_layer"]) == set(LAYERS) and res["best_layer"] in LAYERS
    assert all(v["queries"] == 4 for v in res["by_layer"].values())
    assert res["selector_success"] == max(v["selector_success"] for v in res["by_layer"].values())
    assert model.training and readout.training


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_query_logits_do_not_depend_on_batching(s2t, tat, tmp_path, monkeypatch, precision):
    # batching only pads; in fp32 the scores agree to rounding, in bf16 to bf16 precision
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    model.eval()
    if precision == "fp32":
        import contextlib
        monkeypatch.setattr(tat.S2, "autocast", lambda device: contextlib.nullcontext())
    m = items[0]
    together = tat.query_logits(model, readout, src, m, m.queries, "cpu", 4)
    alone = [tat.query_logits(model, readout, src, m, [q], "cpu", 1)[0] for q in m.queries]
    for q, a, b in zip(m.queries, together, alone):
        for k in LAYERS:
            assert a[k].shape == (q.answer.count(SEL), m.n)
            assert torch.allclose(a[k], b[k], atol=1e-5 if precision == "fp32" else 3e-2)


def test_checkpoint_round_trip(s2t, tat, tmp_path):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    with torch.no_grad():
        for p in list(readout.parameters()) + LR.trainable_parameters(model, extra["sel_emb"]):
            p.add_(0.1 * torch.randn_like(p))
    torch.save(tat.checkpoint_state(model, extra, readout, {"layer": 4}), tmp_path / "ck.pt")
    (tmp_path / "b").mkdir()
    _, _, model2, _, extra2, readout2, src2 = make(s2t, tat, tmp_path / "b", "tree")   # same frozen base weights
    LR.load_trainable(model2, extra2["sel_emb"], readout2, torch.load(tmp_path / "ck.pt"))
    model.eval()
    model2.eval()
    m = items[0]
    a = tat.query_logits(model, readout, src, m, m.queries[:1], "cpu", 1)[0]
    b = tat.query_logits(model2, readout2, src2, m, m.queries[:1], "cpu", 1)[0]
    assert all(torch.equal(a[k], b[k]) for k in LAYERS)


def test_eval_cache_reproduces_the_model_scores(s2t, tat, tae, tmp_path):
    # the identity adapter must hand the V2 / V4 evaluators exactly the model's node scores
    v2 = load_script("v2_train_eval")
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    model.eval()
    rows = []
    for m in items:
        for q in m.queries:
            text = f"{q.user} {SEL}" if q.kind == "absent" else q.user
            rows.append({"qid": q.qid, "kind": q.kind, "text": text, "sha1": m.sha1, "split": "val",
                         "image_key": m.key, "n_sel": q.answer.count(SEL)})
    queries = pd.DataFrame(rows).sort_values(["sha1", "qid"]).reset_index(drop=True)
    queries["offset"] = queries.groupby("sha1")["n_sel"].cumsum() - queries["n_sel"]
    wl = pd.DataFrame({"sha1": [m.sha1 for m in items], "path": [m.file for m in items],
                       "image_key": [m.key for m in items]})
    out = tmp_path / "logits"
    assert tae.compute_logits(model, readout, src, props, queries, wl, 4, "cpu", out, 2, 0, 1) == 3
    sel, store, probes = tae.IdentitySelector(), tae.IdentityStore(props, 0.9), v2.Probes(out, 0)
    for m in items:
        e = sel.nodes(store.load(m.sha1, ["hidden_L4"])["hidden_L4"])
        assert e.shape == (m.n, m.n)
        byq = {q.qid: q for q in m.queries}
        g = queries[queries.sha1 == m.sha1].sort_values("offset")
        ref = tat.query_logits(model, readout, src, m, [byq[q] for q in g.qid], "cpu", 2, [4])   # same batching
        for q, r in zip(g.itertuples(), ref):
            got = sel.scores(sel.queries(probes.get(m.sha1, q.offset, q.n_sel)), e)
            assert torch.equal(got, r[4] @ torch.eye(m.n)) and torch.allclose(got, r[4], atol=1e-6)
    meta = s2t.ImageMeta("s0", "k", items[0].file, 0, np.array([-1]), None, None, None, [])
    assert meta.n == 0
    save_tensors(tmp_path / "z.safetensors", {"probes": torch.zeros((2, 1, 0))}, {"image_sha1": "z"})
    assert load_tensors(tmp_path / "z.safetensors", ["probes"])[0]["probes"].shape == (2, 1, 0)


def test_rest_cache_items_load_like_worklist_items(s2t, tat, tmp_path, monkeypatch):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    m = items[0]
    rles = props.nodes(m.sha1, 0.9)["rles"]
    tree = T.build_tree(rles)
    d = tmp_path / "cache"
    (d / "shards").mkdir(parents=True)
    (d / "nodes").mkdir()
    (d / "nodes" / "k.json").write_text(json.dumps({"h": rles[0]["size"][0], "w": rles[0]["size"][1],
                                                    "counts": [r["counts"] for r in rles]}))
    q = {"qid": "paco_lvis:9|part|bottle:cap", "kind": "part", "text": f"Locate the cap of the bottle. {SEL}",
         "steps": [(np.array([1], np.int32), np.array([], np.int32))], "path": None}
    rec = {"key": "paco_lvis:9", "path": m.file, "h": 1, "w": 1, "nodes": "nodes/k.json",
           "parent": tree.parent.astype(np.int32), "queries": [q], "absent": [],
           "labels": list(m.labels), "parts": list(m.parts), "high": np.asarray(m.high)}
    torch.save({"records": [rec, {**rec, "key": "paco_lvis:10", "queries": []}], "counts": {"queries": 1}},
               d / "shards" / "00000.pt")
    monkeypatch.setattr(tat.TPN, "out_dir", lambda cfg, smoke: d)
    monkeypatch.setattr(tat.TPN, "num_shards", lambda cfg, smoke: 1)
    metas, counts = tat.rest_meta({}, False)
    assert len(metas) == 1 and counts["paco_train_rest:training_items"] == 1
    r = metas[0]
    assert r.source == "paco_train" and r.n == m.n and list(r.labels) == list(m.labels)
    assert r.queries[0].answer == f"cap of the bottle {SEL}" and r.queries[0].user == "Locate the cap of the bottle."
    (seq,), nb = src.image_item(r, r.queries)
    assert nb.n == m.n and np.array_equal(T.from_parents(r.parents).parent, tree.parent)


def test_rest_items_skip_the_worklist(tat, monkeypatch):
    rows = [("paco_lvis:1", "a", [], None), ("paco_lvis:2", "b", [], None), ("paco_lvis:3", "c", [], None)]
    wl = pd.DataFrame({"image_key": ["paco_lvis:2", "coco:1"], "role": ["paco_train", "grefcoco_train_nt"]})
    monkeypatch.setattr(tat.TPN, "manifest_inputs", lambda cfg, smoke: (None, None, None, wl))
    monkeypatch.setattr(tat.TPN.S1C, "items", lambda cfg, source, shard, n: iter(rows))
    assert [k for k, *_ in tat.TPN.rest_items({}, 0, 1)] == ["paco_lvis:1", "paco_lvis:3"]


def test_variant_configs_share_one_recipe(monkeypatch):
    from pathlib import Path
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    root = Path(__file__).resolve().parents[1] / "configs"
    cfgs = {v: load_config(root / f"ta_{v}.yaml") for v in ("flat", "full", "tree", "random_tree", "hier",
                                                            "hier_random_tree")}
    base = {k: v for k, v in cfgs["tree"]["ta"].items() if k != "variant"}
    for v, cfg in cfgs.items():
        assert cfg["ta"]["variant"] == v
        assert {k: x for k, x in cfg["ta"].items() if k != "variant"} == base
        assert {k: x for k, x in cfg.items() if k != "ta"} == {k: x for k, x in cfgs["tree"].items() if k != "ta"}
    s2l = load_config(root / "s2_long.yaml")
    for key in ("proposals", "models", "s2_nodes", "manifest", "worklist", "data"):
        assert cfgs["tree"][key] == s2l[key]                                    # same caches as S2-long
    assert base["lora"] == s2l["s2"]["lora"] and base["steps"] == s2l["s2"]["steps"]


def test_check_config_keeps_its_own_stage(tat, monkeypatch):
    from pathlib import Path
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    root = Path(__file__).resolve().parents[1] / "configs"
    check, real = load_config(root / "ta_tree_check.yaml"), load_config(root / "ta_tree.yaml")
    assert tat.stage(check) == "ta_tree_check" and tat.stage(real) == "ta_tree"
    assert check["ta"]["steps"] == 20 and not check["ta"]["paco_rest"]
    keep = {k: v for k, v in check["ta"].items() if k not in ("stage_suffix", "steps", "eval_every", "eval_images",
                                                            "log_every", "paco_rest")}
    assert keep == {k: v for k, v in real["ta"].items() if k in keep}


def test_relation_swap_for_evaluation(s2t, tat, tmp_path):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "tree")
    m = items[0]
    base = {"cfg": {"proposals": {"node_dedup": 0.9}}, "smoke": False, "manifest": None, "mhash": None,
            "processor": proc, "sel_id": extra["sel_id"], "text_labels": set()}

    def source(variant, relations):
        s = tat.Source(*base.values(), variant, 0, relations)
        s._props = props
        return s

    (_, nb_tree), (_, nb_self) = (source("tree", r).image_item(m, m.queries[:1]) for r in (None, "self"))
    assert all(r.tolist() == [v] for v, r in enumerate(nb_self.rel))
    assert all(np.array_equal(a, b) for a, b in zip(nb_tree.region, nb_self.region))
    assert source("random_tree", None).mode == "random_tree" and source("tree", "full").mode == "full"
    with pytest.raises(Exception):
        source("flat", "self")
    with pytest.raises(Exception):
        source("tree", "nonsense")
    assert "self" not in tat.VARIANTS                                       # never a training variant


def test_hier_logits_follow_the_tree_in_use(s2t, tat, tmp_path):
    # hier scores the decision tree it was given: true tree vs shuffled tree give different logits
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "hier")
    model.eval()
    m = items[0]
    true = tat.query_logits(model, readout, src, m, m.queries[:1], "cpu", 1)[0]
    (_, nb_true) = src.image_item(m, m.queries[:1])
    for seed in range(20):                       # a 5-node tree can shuffle into itself; take a seed that changes it
        rand_src = tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], LABELS,
                              "hier_random_tree", seed)
        rand_src._props = props
        (_, nb_rand) = rand_src.image_item(m, m.queries[:1])
        if not np.array_equal(nb_true.anc, nb_rand.anc):
            break
    assert not np.array_equal(nb_true.anc, nb_rand.anc)
    other = tat.query_logits(model, readout, rand_src, m, m.queries[:1], "cpu", 1)[0]
    assert all(true[k].shape == (m.queries[0].answer.count(SEL), m.n) for k in LAYERS)
    assert any(not torch.allclose(true[k], other[k]) for k in LAYERS)
    assert tat.deviations("hier") == ["D10", "D21", "D41", "D42"] and tat.deviations("tree") == ["D10", "D21", "D41"]
