import numpy as np
import pytest

from test_s2 import H, W, load_script
from test_ta_test_eval import FakeScorer
from treeground import masks as M
from treeground.mllm import SEL
from treeground.run import load_config

ROOT = load_script("e_cocoedit").REPO


@pytest.fixture(scope="module")
def src():
    return load_script("cocoedit_ta_source")


def selector(src, logits, pool):
    s = src.TASelect.__new__(src.TASelect)
    s.TTE, s.s = load_script("ta_test_eval"), FakeScorer(logits, pool=pool)
    return s


def test_top1_mask_and_query_match_gres(src):
    # the same query as gRefCOCO (GresOurs) and the single highest-scoring candidate
    lg = [0.5, -3.0, 2.0, -9.0, 1.0]
    s = selector(src, lg, False)
    image = np.zeros((H, W, 3), np.uint8)
    mask, score, info = s(image, "the red mug")
    q = s.s.calls[0][0]
    assert q.user == "the red mug" and q.answer == f"the red mug {SEL}" and q.kind == "object"
    assert (mask == M.decode(s.s.rles[2]).astype(bool)).all() and mask.dtype == bool
    assert score == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert info == {"kind": "node", "pool": 5, "nodes": 5}
    gres = load_script("ta_test_eval").GresOurs.__new__(load_script("ta_test_eval").GresOurs)
    gres.s = FakeScorer(lg)
    gres(image, ["the red mug"])
    g = gres.s.calls[0][0]
    assert (g.user, g.answer, g.kind) == (q.user, q.answer, q.kind)


def test_text_instance_pick_and_empty_pool(src):
    s = selector(src, [0.0, -1.0, -1.0, -1.0, -1.0, 3.0], True)        # the text instance (index 5) wins
    mask, _, info = s(np.zeros((H, W, 3), np.uint8), "the handle")
    assert info == {"kind": "inst", "pool": 6, "nodes": 5} and (mask == M.decode(s.s.rles[1]).astype(bool)).all()
    tte = load_script("ta_test_eval")
    s.s.score = lambda image, key, texts, queries: ([tte.Pool([], None, 0)], [np.zeros((1, 0))])
    assert s(np.zeros((H, W, 3), np.uint8), "nothing") == (None, float("-inf"), {"kind": "none", "pool": 0,
                                                                                  "nodes": 0})


def test_pool_edit_config(src, monkeypatch):
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    C = load_script("e_cocoedit")
    cfg = load_config(ROOT / "configs/edit_cocoedit_pool.yaml")
    assert C.conditions(cfg, "qwen_image_edit") == ["ours_pool_blend"]
    assert C.conditions(cfg, "flux_fill") == ["ours_pool_fill"]
    assert C.used_sources(cfg, "qwen_image_edit") == C.used_sources(cfg, "flux_fill") == ["ours_pool"]
    spec = cfg["cocoedit"]["sources"]["ours_pool"]
    assert C.source_class(spec).__name__ == "TASelect"
    sub = src.TASelect.sub_cfg(spec)
    assert sub["ta"]["variant"] == "pool_full"
    req = src.TASelect.requires(spec)
    assert {"D3", "D4", "D44", "D47"} <= set(req)
