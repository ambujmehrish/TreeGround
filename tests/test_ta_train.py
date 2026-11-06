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
    mode, fam = tat.D43.get(variant) or tat.D44.get(variant) or ("", False)
    d_geom = TA.GEOM_DIM + (TA.INST_DIM if tat.is_pool(variant) else 0)
    embed = None if variant in tat.FLAT else TA.NodeEmbed(64, TA.embedding_rms(model), d_geom)
    soft = TA.SoftTreeBias(4) if mode.startswith("soft") else None
    family = (0.1, 1.0) if fam else None
    readout = tat.Readout(64, LAYERS, 5, embed, variant in tat.HIER, soft, family)
    if soft is not None:
        soft.install(model)
    src = tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], LABELS, variant, 0)
    src._props = props
    return items, props, model, proc, extra, readout, src


@pytest.mark.parametrize("variant", ["flat", "full", "tree", "random_tree", "hier", "hier_random_tree", "soft",
                                     "soft_random_tree", "fam", "fam_random_tree", "soft_fam", "soft_fam_random_tree"])
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
    if readout.soft is not None:                                         # the relation biases are trained
        assert readout.soft.bias.grad.abs().sum() > 0
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


def tat_variants():
    return ("flat", "full", "tree", "random_tree", "hier", "hier_random_tree", "soft", "soft_random_tree", "fam",
            "fam_random_tree", "soft_fam", "soft_fam_random_tree")


def test_variant_configs_share_one_recipe(monkeypatch):
    from pathlib import Path
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    root = Path(__file__).resolve().parents[1] / "configs"
    cfgs = {v: load_config(root / f"ta_{v}.yaml") for v in tat_variants()}
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


def test_partial_decisions_of_a_hier_model(s2t, tat, tmp_path):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "hier")
    model.eval()
    m = items[0]
    q = m.queries[:1]
    got = {}
    for d in ("product", "stop", "stop_root"):
        readout.decision = d
        got[d] = tat.query_logits(model, readout, src, m, q, "cpu", 1, [4])[0][4]
    readout.decision = "product"
    (seq,), nb = src.image_item(m, q)
    batch = {k: v for k, v in tat.S2.collate([seq], src.pad_id).items()}
    with torch.no_grad(), tat.S2.autocast("cpu"):
        res = tat.run_forward(model, readout, batch, [nb])
    h = readout.heads[readout.layers.index(4)]
    probe, states = res.hidden[4][0, seq.sel].float(), res.nodes[4][0].float()
    with torch.no_grad():
        stop = h.stop.scores(h.stop.queries(probe), h.stop.nodes(states))
        root = h.root_logit(probe)
    assert torch.allclose(got["stop"], stop, atol=1e-5)
    p = torch.sigmoid(root)[:, None] * torch.sigmoid(stop)
    assert torch.allclose(torch.sigmoid(got["stop_root"]), p, atol=1e-5)
    assert torch.all(torch.sigmoid(got["product"]) <= torch.sigmoid(got["stop_root"]) + 1e-6)   # extra factors <= 1
    readout.decision = "nonsense"
    with pytest.raises(Exception):
        tat.query_logits(model, readout, src, m, q, "cpu", 1, [4])


def test_family_loss_enters_the_selection_loss(s2t, tat, tmp_path):
    # same batch, same weights: the fam variant adds a non-negative family term to the full-attention loss
    losses = {}
    for variant in ("full", "fam"):
        (tmp_path / variant).mkdir()
        items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path / variant, variant)
        model.eval()
        c = {"batch_images": 3, "micro_batch": 3, "seed": 0, "text_only_fraction": 0.0, "steps": 1}
        mb = tat.MicroBatches(s2t.plan(items, c, LABELS, 1, 0), items, src)[0]
        with torch.no_grad():
            torch.manual_seed(0)
            losses[variant] = tat.micro_step(model, extra, readout, mb, items, "cpu", np.random.default_rng(0), 2.0,
                                             {k: 0 for k in s2t.STAT_KEYS})[0]
    assert losses["fam"] >= losses["full"] - 1e-6
    assert tat.deviations("soft_fam") == ["D10", "D21", "D41", "D43"]
    with pytest.raises(Exception):
        tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, None, 0, set(), "soft", 0, "self")


def as_pool(s2t, tat, items):
    # the grid scene as pool items: every entry a grid node (provenance zero), masks read from the store as before
    return [tat.PoolMeta(**{f: getattr(m, f) for f in s2t.ImageMeta.__dataclass_fields__},
                         inst=TA.pool_features(m.n, [])) for m in items]


@pytest.mark.parametrize("variant", sorted(load_script("ta_train").D44))
def test_training_step_pool_variants(s2t, tat, tmp_path, variant):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, variant)
    if variant == "pool_flat":
        assert readout.embed is None
    else:
        assert readout.embed.d_geom == TA.GEOM_DIM + TA.INST_DIM
    items = as_pool(s2t, tat, items)
    c = {"batch_images": 3, "micro_batch": 3, "seed": 0, "text_only_fraction": 0.34, "steps": 1}
    mb = tat.MicroBatches(s2t.plan(items, c, LABELS, 1, 0), items, src)[0]
    assert all(s["nodes"].inst.shape == (items[s["item"]].n, TA.INST_DIM) for s in mb["samples"])
    stats = {k: 0 for k in s2t.STAT_KEYS}
    l_sel, l_lm = tat.micro_step(model, extra, readout, mb, items, "cpu", np.random.default_rng(0), 2.0, stats)
    (l_sel + l_lm).backward()
    assert torch.isfinite(l_sel)
    if readout.embed is not None:
        assert readout.embed.geom[0].weight.grad[:, TA.GEOM_DIM:].abs().sum() == 0      # zero provenance inputs
        assert readout.embed.geom[0].weight.grad[:, :TA.GEOM_DIM].abs().sum() > 0
    if readout.soft is not None:
        assert readout.soft.bias.grad.abs().sum() > 0
    with pytest.raises(Exception):                                            # no relation swap for pool checkpoints
        tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], LABELS, variant, 0,
                   "self")


def test_variant_tables(tat):
    assert set(tat.D44) <= set(tat.VARIANTS) and all(tat.is_pool(v) for v in tat.D44)
    assert not any(tat.is_pool(v) for v in ("flat", "full", "tree", "soft_fam", "hier"))
    assert tat.deviations("pool_full") == ["D10", "D21", "D41", "D44"]
    assert tat.deviations("pool_flat") == ["D10", "D21", "D41", "D44"] and tat.node_mode("pool_flat") == "tree"
    assert tat.deviations("pool_soft_fam_random_tree") == ["D10", "D21", "D41", "D43", "D44"]
    assert tat.deviations("pool_fam") == ["D10", "D21", "D41", "D43", "D44"]
    assert tat.deviations("soft") == ["D10", "D21", "D41", "D43"] and tat.deviations("tree") == ["D10", "D21", "D41"]
    assert tat.node_mode("pool_soft_fam") == "soft" and tat.node_mode("pool_full_random_tree") == "full_random_tree"
    assert tat.node_mode("pool_soft_fam_free") == "soft_free" and tat.is_pool("pool_soft_fam_free")
    assert tat.deviations("pool_soft_fam_free") == ["D10", "D21", "D41", "D43", "D44", "D48"]
    assert tat.deviations("pool_soft_fam") == ["D10", "D21", "D41", "D43", "D44"]
    assert tat.node_mode("pool_fam") == "full" and tat.node_mode("hier") == "tree"


def pool_shard(tmp_path, items, props):
    # an s3_pool shard over the scene: grid nodes + one text instance per image (a copy of the part mask)
    d = tmp_path / "pool"
    (d / "nodes").mkdir(parents=True)
    (d / "shards").mkdir()
    recs = []
    for m in items:
        if m.absent:
            continue
        rles = list(props.nodes(m.sha1, 0.9)["rles"])
        pool = rles + [rles[1]]
        parent = T.build_tree(pool).parent
        rel = f"nodes/{m.sha1}.json"
        (d / rel).write_text(json.dumps({"h": 96, "w": 128, "counts": [r["counts"] for r in pool]}))
        qs = [{"qid": q.qid, "kind": q.kind, "text": q.user + (f" {SEL}" if q.kind != "chain" else ""),
               "steps": [(s.pos.tolist(), s.ignore.tolist()) for s in q.steps], "path": q.path} for q in m.queries]
        recs.append({"key": m.key, "path": m.file, "h": 96, "w": 128, "nodes": rel, "parent": parent, "queries": qs,
                     "absent": [{"qid": "gref|x", "text": f"nothing {SEL}"}], "n_nodes": len(rles),
                     "phrases": ["the cap"], "instances": [{"phrase": 0, "rank": 1, "score": 0.7}],
                     "labels": list(m.labels) + [""], "parts": list(m.parts) + [""], "high": list(m.high) + [True]})
    torch.save({"records": recs, "counts": {"queries": 6}}, d / "shards" / "00000.pt")
    return d


def test_pool_meta_reads_the_d30_cache(s2t, tat, tmp_path, monkeypatch):
    items, props = build_scene(s2t, tmp_path)
    d = pool_shard(tmp_path, items, props)
    monkeypatch.setattr(tat, "pool_cache", lambda cfg, source, smoke: (d, sorted((d / "shards").glob("*.pt"))))
    metas, counts = tat.pool_meta({}, "paco_train", False)
    assert counts["paco_train_pool:training_items"] == 4 and counts["paco_train_pool:queries"] == 6
    present = [m for m in metas if not m.absent]
    assert len(present) == 2 and all(m.source == "paco_train" and m.n == 6 for m in metas)
    m = present[0]
    assert m.inst.shape == (6, TA.INST_DIM) and m.inst[5].tolist() == [1, 0, 1, 0, pytest.approx(0.7)]
    assert not m.inst[:5].any() and m.rles is None and m.nodes.endswith("nodes/s0.json")
    assert m.high[5] and len(m.labels) == 6 and [q.kind for q in m.queries] == ["chain", "object", "part"]
    absent = [m for m in metas if m.absent]
    assert len(absent) == 2 and absent[0].queries[0].kind == "absent" and absent[0].parents is not None
    bad = torch.load(d / "shards" / "00000.pt", weights_only=False)
    bad["records"][0]["instances"] = []
    torch.save(bad, d / "shards" / "00000.pt")
    with pytest.raises(Exception):
        tat.pool_meta({}, "paco_train", False)


class FakeText:
    # s3_pool.TextInstances stand-in: the phrase's instance is a copy of the part mask, then an empty mask
    def __init__(self, part_mask):
        self.part, self.calls = part_mask, []

    def __call__(self, image, phrases, k):
        self.calls.append(list(phrases))
        return [[(self.part.copy(), 0.9), (np.zeros_like(self.part), 0.1)] for _ in phrases]


def test_pool_val_meta_builds_pools_with_chain_ious(s2t, tat, tmp_path, monkeypatch):
    from test_s2 import box, paco_image, query_rows
    items, props = build_scene(s2t, tmp_path)
    img = paco_image(1, "bottle", "cap", box(10, 90, 20, 60), box(10, 30, 30, 50))
    monkeypatch.setattr(tat.paco, "load", lambda root, split, ids: {1: img})
    q = query_rows(img, "s0").assign(split="val", text=lambda d: d.text)
    wl = pd.DataFrame({"sha1": ["s0"], "image_key": ["paco_lvis:1"], "path": [items[0].file], "role": ["paco_val"]})
    cfg = {"proposals": {"node_dedup": 0.9}, "data": {"paco_lvis": "x"},
           "s3_pool": {"instances_per_phrase": 2, "max_instances": 160, "dup_iou": 0.95}}
    text = FakeText(box(10, 30, 30, 50))
    metas = tat.pool_val_meta(cfg, wl, props, q, None, 5, 0, 0, 1, text)
    assert len(metas) == 1 and text.calls == [["the cap of the bottle", "the bottle"]]   # chain qid first
    m = metas[0]
    assert m.n == 6 and len(m.rles) == 6 and m.inst[5].tolist() == [1, 1, 0, 0, pytest.approx(0.9)]  # dedup drops
    kinds = {q.kind: q for q in m.queries}                                                           # the copies
    assert set(kinds) == {"object", "part", "chain"} and kinds["part"].ious.shape == (6,)
    assert kinds["chain"].ious.shape == (2, 6) and kinds["chain"].ious[1, 5] == 1.0 and kinds["part"].ious[5] == 1.0
    assert m.parents[5] == 0                                                   # the instance sits under the bottle


def test_val_success_scores_chains_and_chooses_an_anchor(s2t, tat, tmp_path):
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path, "pool_soft")
    items = as_pool(s2t, tat, items)
    metas = []
    for m in items[:2]:
        qs = [s2t.Query(q.qid, q.kind, q.user, q.answer, [], None,
                        np.stack([np.linspace(0, 1, m.n), np.linspace(1, 0, m.n)]) if q.kind == "chain"
                        else np.linspace(0, 1, m.n)) for q in m.queries]
        metas.append(tat.PoolMeta(m.sha1, m.key, m.file, m.n, m.parents, None, None, None, qs, inst=m.inst))
    res = tat.val_success(model, readout, src, metas, "cpu", 1, 2)
    for k, v in res["by_layer"].items():
        assert v["queries"] == 4 and v["chains"] == 2                        # chains do not enter the S2 criterion
        assert {"chain_object", "chain_part_none", "chain_part_hard", "chain_part_soft2"} <= set(v["by_kind"])
        assert v["anchor"] in ["none", TA.HARD] + [TA.anchor_name(a) for a in TA.ANCHORS]
    assert res["anchor"] == res["by_layer"][res["best_layer"]]["anchor"]
    (tmp_path / "g").mkdir()
    items, props, model, proc, extra, readout, src = make(s2t, tat, tmp_path / "g", "tree")
    grid = tat.val_success(model, readout, src, val_metas(s2t, items), "cpu", 1, 2)
    assert grid["anchor"] is None and all(v["chains"] == 0 for v in grid["by_layer"].values())


def test_8b_config_reads_the_4b_query_cache(tat, monkeypatch):
    from pathlib import Path
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    root = Path(__file__).resolve().parents[1] / "configs"
    big, base = load_config(root / "ta_pool_full_8b.yaml"), load_config(root / "ta_pool_full.yaml")
    assert big["models"]["qwen"]["repo"] == "Qwen/Qwen3-VL-8B-Instruct" and tat.stage(big) == "ta_pool_full_8b"
    assert big["models"]["qwen"] == load_config(root / "baselines2.yaml")["models"]["qwen8"]   # the pinned D31 revision
    q = tat.query_cfg(big)
    assert q["models"]["qwen"] == base["models"]["qwen"] and q["ta"] is big["ta"]
    assert tat.query_cfg(base) is base
    assert tat.run_deviations(big["ta"]) == tat.deviations("pool_full") + ["D46"]
    assert tat.run_deviations(base["ta"]) == tat.deviations("pool_full")
    same = {k: v for k, v in big["ta"].items() if k not in ("stage_suffix", "micro_batch", "query_cache_qwen")}
    assert same == {k: v for k, v in base["ta"].items() if k not in ("stage_suffix", "micro_batch")}
    check = load_config(root / "ta_pool_full_8b_check.yaml")
    assert tat.stage(check) == "ta_pool_full_8b_check" and check["ta"]["steps"] == 20
    t = load_config(root / "ta_test_pool_full_8b.yaml")
    assert t["ta"] == big["ta"] and t["models"]["qwen"] == big["models"]["qwen"]
