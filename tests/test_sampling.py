import inspect

import numpy as np
import pytest

from treeground import sampling as S
from treeground import tree as T


def test_in_image_sampler_hard_then_uniform_without_repeats():
    scores = np.arange(200, dtype=float)
    pos, ign = np.array([199]), np.array([198, 5])
    pool = S.eligible(np.arange(200), pos, ign)
    for seed in range(10):
        d = S.in_image_negatives(pool, scores, np.random.default_rng(seed))
        assert len(d.ids) == 64 and len(set(d.ids)) == 64 and d.shortfall == 0
        assert not set(d.ids) & {199, 198, 5}
        assert set(range(166, 198)) <= set(d.ids)  # the 32 highest-scoring eligible nodes


def test_in_image_shortfall_recorded():
    pool = S.eligible(np.arange(20), np.array([0]), np.array([1]))
    d = S.in_image_negatives(pool, np.zeros(20), np.random.default_rng(0))
    assert sorted(d.ids) == list(range(2, 20)) and d.shortfall == 64 - 18


def test_chain_negatives_subtree_first_then_topup():
    scores = np.random.default_rng(1).normal(size=300)
    subtree = np.arange(10, 50)
    pos, ign = np.array([10]), np.array([11])
    inside = S.eligible(subtree, pos, ign)
    outside = S.eligible(np.setdiff1d(np.arange(300), subtree), pos, ign)
    d = S.chain_negatives(inside, outside, scores, np.random.default_rng(0))
    assert len(d.ids) == 64 and len(set(d.ids)) == 64
    assert set(inside) <= set(d.ids)
    assert d.stats == {"subtree": 38, "topup": 26}
    assert not set(d.ids) & {10, 11}
    big = np.arange(0, 200)
    d2 = S.chain_negatives(big, outside, scores, np.random.default_rng(0))
    assert d2.stats["topup"] == 0 and set(d2.ids) <= set(big)


def test_cross_image_sampler():
    pool = np.arange(1000, 1100)
    d = S.cross_image_negatives(pool, np.random.default_rng(3))
    assert len(d.ids) == 64 and len(set(d.ids)) == 64 and set(d.ids) <= set(pool)
    small = S.cross_image_negatives(np.arange(10), np.random.default_rng(3))
    assert small.shortfall == 54 and len(small.ids) == 10


def test_pre_scoring_sample_takes_no_scores_and_is_seeded():
    assert list(inspect.signature(S.pre_scoring_sample).parameters) == ["pool", "b", "rng"]
    pool = np.arange(5, 105)
    a = S.pre_scoring_sample(pool, 17, np.random.default_rng([2, 7, 1]))
    b = S.pre_scoring_sample(pool, 17, np.random.default_rng([2, 7, 1]))
    c = S.pre_scoring_sample(pool, 17, np.random.default_rng([3, 7, 1]))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(set(a)) == 17 and set(a) <= set(pool)
    with pytest.raises(ValueError):
        S.pre_scoring_sample(pool, 101, np.random.default_rng(0))


def test_inference_pools_contain_only_proposal_nodes():
    tree = T.from_parents(np.array([4, 0, 0, 4, -1]))
    assert list(S.inference_pool(tree, None)) == [0, 1, 2, 3]
    assert list(S.inference_pool(tree, 0)) == [1, 2]
    assert list(S.inference_pool(tree, tree.root)) == [0, 1, 2, 3]
    assert list(inspect.signature(S.inference_pool).parameters) == ["tree", "anchor"]
