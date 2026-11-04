import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from test_s2 import H, W, box, build_scene, load_script, wrapped
from treeground import masks as M
from treeground import tree as T
from treeground import tree_attn as TA
from treeground.mllm import SEL


@pytest.fixture(scope="module")
def tte():
    return load_script("ta_test_eval")


def scene():
    masks = [box(10, 90, 10, 70), box(10, 40, 10, 40), box(50, 90, 40, 70), box(12, 24, 12, 24), box(20, 80, 90, 125)]
    rles = M.encode(np.stack(masks))
    return rles, T.build_tree(rles).parent


def test_memory_source_matches_the_training_source(tte, tmp_path):
    # the in-memory source must give the model exactly what ta_train.Source gives it for the same image and nodes
    s2t = load_script("s2_train")
    tat = tte.TAT
    items, props = build_scene(s2t, tmp_path)
    assert tat.add_missing_trees(items, props, 0.9, None) == 1
    model, proc, extra = wrapped(s2t)
    model.eval()
    torch.manual_seed(0)
    readout = tat.Readout(64, [2, 4], 5, TA.NodeEmbed(64, TA.embedding_rms(model)))
    src = tat.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], set(), "tree", 0)
    src._props = props
    m = items[0]
    image = np.asarray(Image.open(m.file).convert("RGB"))
    mem = tte.MemorySource(proc, extra["sel_id"], image, props.nodes(m.sha1, 0.9)["rles"], m.parents, "tree", 0, m.sha1)
    a = tat.query_logits(model, readout, src, m, m.queries, "cpu", 2)
    b = tat.query_logits(model, readout, mem, None, m.queries, "cpu", 2)
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in (2, 4))


def test_rows_follow_the_s3_eval_rules(tte):
    rles, parents = scene()
    gts = {0: rles[0], 1: rles[1]}
    g = pd.DataFrame([{"qid": "k|object|mug", "kind": "object", "image_key": "k", "gt": json.dumps([0])},
                      {"qid": "k|part|mug:handle", "kind": "part", "image_key": "k", "gt": json.dumps([1])},
                      {"qid": "k|chain|mug:handle", "kind": "chain", "image_key": "k", "gt": json.dumps([0, 1])}])
    lgs = [np.array([[0.9, 0.1, 0.2, 0.0, 0.3]]), np.array([[0.1, 0.2, 0.8, 0.0, 0.1]]),
           np.array([[0.9, 0.1, 0.0, 0.0, 0.2], [0.0, 0.3, 0.1, 0.0, 0.95]])]
    pool = tte.Pool(rles, parents, len(rles))
    rows = tte.single_rows(g, [pool] * 3, lgs, gts)
    assert rows[0]["sel_iou"] == 1.0 and rows[0]["pick_kind"] == "node" and rows[0]["best_pool_iou"] == 1.0
    assert rows[1]["sel_iou"] < 0.5 and rows[1]["best_node_iou"] == 1.0                 # picked node 2, not 1
    assert "anchor" not in rows[1] and rows[1]["pool"] == rows[1]["nodes"] == 5
    ch = rows[2]
    assert ch["object_iou"] == 1.0 and ch["part_flat_iou"] < 0.5                      # flat part pick: node 4
    assert ch["part_tree_iou"] == 1.0                                                 # inside node 0: node 1 wins
    pt = load_script("paco_test_eval")
    df = pd.DataFrame(rows).assign(image="k")
    res = pt.summarize_ours(df, {"n": 50, "seed": 0})                                 # the protocol summary reads them
    assert res["present_queries"] == 2 and res["chain"]["chains"] == 1
    leaf = [np.array([[0.0, 0.0, 0.0, 9.0, 0.0], [0.0, 0.3, 0.1, 0.0, 0.95]])]     # object pick without children
    assert tte.single_rows(g.iloc[2:], [pool], leaf, gts)[0]["part_tree_iou"] == M.iou([rles[3]], [rles[1]])[0, 0]
    empty = tte.Pool([], None, 0)
    rows = tte.single_rows(g, [empty] * 3, [np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((2, 0))], gts)
    assert rows[0]["sel_iou"] == 0.0 and rows[2]["part_tree_iou"] == 0.0


def test_anchored_part_rows_over_a_query_pool(tte):
    # a pool checkpoint: the part query carries object and part logits; the pool holds a text instance (a copy of
    # the handle) after the grid nodes; the object pick anchors the part pick inside the mug
    rles, parents = scene()
    pool_rles = rles + [rles[1]]
    pool = tte.Pool(pool_rles, T.build_tree(pool_rles).parent, 5, TA.pool_features(5, [{"phrase": 0, "rank": 0,
                                                                                          "score": 0.8}]))
    gts = {1: rles[1]}
    g = pd.DataFrame([{"qid": "k|part|mug:handle", "kind": "part", "image_key": "k", "gt": json.dumps([1]),
                       "text": "Locate the handle of the mug. <SEL>"}])
    lg = np.array([[3.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.2, 0.1, 0.0, 2.0, 0.5]])   # flat part pick: node 4
    hard = tte.single_rows(g, [pool], [lg], gts, "hard")[0]
    assert hard["sel_iou"] == 1.0 and hard["pick_kind"] == "inst" and hard["anchor"] == "hard"
    assert hard["anchor_object"] == "node" and hard["part_none_iou"] < 0.5 and hard["pool"] == 6 and hard["nodes"] == 5
    none = tte.single_rows(g, [pool], [lg], gts, "none")[0]
    assert none["sel_iou"] < 0.5 and none["pick_kind"] == "node"
    soft = tte.single_rows(g, [pool], [lg], gts, "soft2")[0]
    assert soft["sel_iou"] == 1.0 and soft["max_sigma"] == pytest.approx(tte.sigmoid(0.5))
    with pytest.raises(Exception):                                                    # one row cannot be anchored
        tte.single_rows(g, [pool], [lg[:1]], gts, "hard")
    q = tte.part_as_chain(tte.queries_of(g)[0])
    assert q.kind == "chain" and q.user == "Locate the mug and then its handle." and q.answer.count(SEL) == 2
    assert tte.queries_of(g, True)[0].answer == q.answer and tte.queries_of(g)[0].kind == "part"


class FakeScorer:
    def __init__(self, logits, pool=False, anchor="hard"):
        self.rles, self.parents = scene()
        self.device, self.calls, self.lg, self.pool = "cpu", [], logits, pool
        self.cfg = {"ta_eval": {"anchor": "auto"}}
        self.meta = {"pool": pool, "anchor": anchor if pool else None}
        self.anchor = anchor if pool else None

    def pools(self, image, texts):
        if not self.pool:
            return [tte_pool(self.rles, self.parents, len(self.rles))] * len(texts)
        rles = self.rles + [self.rles[1]]                               # + a text instance: a handle copy
        return [tte_pool(rles, T.build_tree(rles).parent, len(self.rles),
                         TA.pool_features(len(self.rles), [{"phrase": 0, "rank": 0, "score": 0.8}])) for _ in texts]

    def score(self, image, key, texts, queries):
        self.calls.append(queries)
        pools = self.pools(image, texts)
        return pools, [np.array([self.lg[:p.n]] * q.answer.count(SEL)) for p, q in zip(pools, queries)]


def tte_pool(*a, **k):
    return load_script("ta_test_eval").Pool(*a, **k)


def test_benchmark_predictors(tte, monkeypatch):
    lg = [2.0, -3.0, 1.0, -9.0, 0.5]
    for cls in (tte.GresOurs, tte.PartonomyOurs, tte.PacoOurs, tte.OphrgOurs):
        monkeypatch.setattr(cls, "__init__", lambda self, cfg: setattr(self, "s", FakeScorer(lg)))
    image = np.zeros((H, W, 3), np.uint8)
    gres = tte.GresOurs({})
    out = gres(image, ["the red mug", "nothing here"])
    sig = 1 / (1 + np.exp(-np.array(lg)))
    assert [round(s, 6) for s, _ in out[0]["nodes"]] == [round(x, 6) for x in sorted(sig[sig >= 0.05], reverse=True)]
    q = gres.s.calls[0][0]
    assert q.user == "the red mug" and q.answer == f"the red mug {SEL}" and q.kind == "object"
    ge = load_script("gres_eval")
    df = pd.DataFrame([{"qid": "a", "image": "i", "no_target": False, "gt": json.dumps(None),
                        "cand_nodes": json.dumps([[s, r] for s, r in out[0]["nodes"]])}])
    assert len(ge.scores(df, "cand_nodes", "set", 0.5)) == 1                         # gres_eval reads them
    part = tte.PartonomyOurs({})
    res = part(image, "mug", ["handle"])
    assert res[0]["nodes"].shape == (H, W) and res[0]["nodes"].sum() == M.areas([part.s.rles[0]])[0]
    q = part.s.calls[0][0]
    assert q.answer == f"handle of the mug {SEL}" and q.kind == "part"
    oph = tte.OphrgOurs({})
    gt = M.decode(part.s.rles[0]).astype(bool)
    r = oph(image, "img1", [("img1|object|mug", "object", "mug", None, gt)])
    assert r[0]["inter"] == r[0]["union"] == int(gt.sum()) and r[0]["pick_kind"] == "node"
    q = oph.s.calls[0][0]
    assert q.answer == f"mug {SEL}" and q.user == "Locate the mug."


def test_predictors_of_a_pool_checkpoint(tte, monkeypatch):
    # part queries are asked object-first and decided inside the object's subtree; object queries stay single
    lg = [3.0, 0.0, 0.0, 0.0, 2.0, 1.0]                                  # object pick 0; flat part pick 4
    for cls in (tte.PartonomyOurs, tte.PacoOurs, tte.OphrgOurs, tte.GresOurs):
        monkeypatch.setattr(cls, "__init__", lambda self, cfg: setattr(self, "s", FakeScorer(lg, pool=True)))
    image = np.zeros((H, W, 3), np.uint8)
    part = tte.PartonomyOurs({})
    res = part(image, "mug", ["handle"])
    q = part.s.calls[0][0]
    assert q.kind == "chain" and q.user == "Locate the mug and then its handle." and q.answer.count(SEL) == 2
    assert res[0]["nodes"].sum() == M.areas([part.s.rles[1]])[0]        # the handle, not the flat pick (node 4)
    oph = tte.OphrgOurs({})
    gt = M.decode(part.s.rles[1]).astype(bool)
    r = oph(image, "img1", [("img1|part|coffee_mug:handle", "part", "coffee_mug", "handle", gt),
                            ("img1|object|mug", "object", "mug", None, M.decode(part.s.rles[0]).astype(bool))])
    assert r[0]["inter"] == r[0]["union"] == int(gt.sum()) and r[0]["pick_kind"] == "inst"
    assert r[1]["inter"] == r[1]["union"] and r[1]["pick_kind"] == "node"
    assert [q.kind for q in oph.s.calls[0]] == ["chain", "object"]
    assert oph.s.calls[0][0].user == "Locate the coffee_mug and then its handle."     # the benchmark's own names
    gres = tte.GresOurs({})
    out = gres(image, ["the handle"])
    assert len(out[0]["nodes"]) == 6 and gres.s.calls[0][0].kind == "object"   # single pick over the query pool
    assert tte.anchor_setting({"ta_eval": {"anchor": "auto"}}, {"pool": True, "anchor": "soft2"}) == "soft2"
    assert tte.anchor_setting({"ta_eval": {"anchor": "none"}}, {"pool": True, "anchor": "hard"}) == "none"
    assert tte.anchor_setting({"ta_eval": {}}, {"pool": False}) is None
    with pytest.raises(Exception):
        tte.anchor_setting({"ta_eval": {"anchor": "auto"}}, {"pool": True, "anchor": None})
    with pytest.raises(Exception):
        tte.anchor_setting({"ta_eval": {"anchor": "tree"}}, {"pool": True, "anchor": "hard"})


def test_install_replaces_only_the_ours_bookkeeping(tte, monkeypatch):
    for bench in tte.BENCHES:
        mod = tte.install(bench)
        assert mod.Ours is tte.OURS[bench]
        cfg = {"ta": {"variant": "full"}, "ta_eval": {"checkpoint": "best.pt"}, "paco_test": {"num_shards": 32},
               "gres_eval": {"num_shards": 16}, "partonomy_eval": {"num_shards": 16}, "ophrg_eval": {"num_shards": 16},
               "gseval_eval": {"num_shards": 4}}
        tte.CONFIG["cfg"] = cfg
        monkeypatch.setattr(tte.TAE, "load_checkpoint", lambda c, s: ({"meta": {"step": 12000, "layer": 36}},
                                                                     SimpleNamespace(name="run1")))
        assert mod.stage_name(cfg, "ours") == "ta_full"
        extra = {"gres": ("testA",), "ophrg": ("instructpart",)}.get(bench, ())
        ident = mod.identity(cfg, "ours", *extra)
        assert ident["run"] == "run1" and ident["pool"] == "tree_nodes_only" and ident["layer"] == 36
        assert ident.get("split", ident.get("benchmark", "x")) in ("testA", "instructpart", "x")
        args = (cfg, "ours") if bench in ("paco", "ophrg", "gseval") else ("ours",)
        assert mod.approvals(*args) == [tte.PROTOCOL[bench], "D10", "D21", "D41"]
        assert mod.stage_name(cfg, "sam3_text") == "sam3_text"                        # baselines untouched
        pool_meta = {"step": 1, "layer": 12, "pool": True, "anchor": "hard", "pool_settings": {"instances_per_phrase": 3}}
        monkeypatch.setattr(tte.TAE, "load_checkpoint", lambda c, s: ({"meta": pool_meta}, SimpleNamespace(name="r2")))
        pcfg = {**cfg, "ta": {"variant": "pool_soft_fam"}, "ta_eval": {"checkpoint": "best.pt", "anchor": "auto"},
                "baselines": {"sam3_text": {"score_threshold": 0.5}}}
        ident = mod.identity(pcfg, "ours", *extra)
        assert ident["pool"] == "grid+text per query" and ident["anchor"] == "hard" and ident["run"] == "r2"
        assert ident["pool_settings"] == {"instances_per_phrase": 3} and ident["sam3_text"] == {"score_threshold": 0.5}


def test_launcher_refuses_baselines(tte, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ta_test_eval.py", "--bench", "paco", "--config", "x", "--method", "sam3_text"])
    with pytest.raises(Exception):
        tte.main()


def test_test_configs(monkeypatch):
    from pathlib import Path
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    root = Path(__file__).resolve().parents[1] / "configs"
    tat = load_script("ta_train")
    for v in ("full", "tree") + tuple(tat.D44):
        c, base = load_config(root / f"ta_test_{v}.yaml"), load_config(root / f"ta_{v}.yaml")
        assert c["ta"] == base["ta"] and c["ta_eval"] == base["ta_eval"]               # same checkpoint selection
        assert c["gres_eval"]["splits"] == ["val", "testA", "testB"] and c["paco_test"]["num_shards"] == 8
        assert c["gseval_eval"]["num_shards"] == 4 and c["data"]["gseval"].endswith("/gseval")
        if v in tat.D44:
            assert c["ta"]["variant"] == v and not c["ta"]["paco_rest"] and c["ta_eval"]["anchor"] == "auto"
            assert c["s3_pool"]["instances_per_phrase"] == 3 and "sam3_text" in c["baselines"]
            assert c["ta"]["sources"] == {"paco_train": 0.6, "pascal_part_train": 0.2, "grefcoco_train": 0.2}
    check = load_config(root / "ta_pool_check.yaml")
    assert check["ta"]["stage_suffix"] == "_check" and check["ta"]["steps"] == 20 and check["ta"]["variant"] == "pool_soft_fam"
