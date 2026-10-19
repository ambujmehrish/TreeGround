import numpy as np
import torch

from treeground import sampling as S
from treeground.selector import Selector, focal, selection_loss


def test_dp_positive_overrides_ignore_rule():
    step_ious = np.array([[0.9, 0.6, 0.0, 0.7], [0.0, 0.0, 0.8, 0.0]])
    single = S.single_step_target(step_ious[0])
    assert list(single.pos) == [0] and sorted(single.ignore) == [1, 3]
    steps = S.chain_step_targets((3, 2), step_ious)   # DP chose node 3 (non-best) for step 1
    assert list(steps[0].pos) == [3]
    assert 0 in steps[0].ignore and 1 in steps[0].ignore and 3 not in steps[0].ignore
    assert list(steps[1].pos) == [2] and len(steps[1].ignore) == 0


def test_proposal_failure_has_no_target():
    assert S.single_step_target(np.array([0.3, 0.49])) is None
    assert S.single_step_target(np.zeros(0)) is None


def test_text_only_matches_are_never_negatives():
    labels = np.array(["mug:handle", "", "cup:handle", "mug:handle", "mug:body", ""] * 30, dtype=object)
    parts = np.array([l.split(":")[1] if l else "" for l in labels], dtype=object)
    high = np.zeros(len(labels), dtype=bool)
    high[1::12] = True
    tgt = S.text_target(labels, parts, high, "mug:handle")
    assert set(tgt.pos) == set(np.flatnonzero(labels == "mug:handle"))
    assert set(np.flatnonzero(labels == "cup:handle")) <= set(tgt.ignore)
    assert set(np.flatnonzero(high)) <= set(tgt.ignore)
    pool = S.eligible(np.arange(len(labels)), tgt.pos, tgt.ignore)
    for seed in range(50):
        d = S.cross_image_negatives(pool, np.random.default_rng(seed))
        assert not set(d.ids) & set(tgt.pos) and not set(d.ids) & set(tgt.ignore)


def test_loss_normalisation_by_set_sizes():
    a, b = torch.tensor([1.5]), torch.tensor([-0.5])
    one, _, _ = selection_loss(a, b)
    rep, _, _ = selection_loss(a.repeat(5), b.repeat(7))
    assert torch.allclose(one, rep)
    pos = torch.tensor([2.0, -1.0, 0.5])
    neg = torch.tensor([0.3, -2.0])
    loss, p, n = selection_loss(pos, neg)
    assert torch.allclose(p, focal(pos, 1).mean()) and torch.allclose(n, focal(neg, 0).mean())
    assert torch.allclose(loss, p + n)


def test_focal_matches_definition():
    x = torch.tensor([-3.0, 0.0, 2.0])
    p = torch.sigmoid(x)
    assert torch.allclose(focal(x, 1), -((1 - p) ** 2) * torch.log(p), atol=1e-6)
    assert torch.allclose(focal(x, 0), -(p ** 2) * torch.log(1 - p), atol=1e-6)


def test_empty_sets_give_finite_loss():
    loss, p, n = selection_loss(torch.zeros(0), torch.zeros(0))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss, p, n = selection_loss(torch.zeros(0), torch.tensor([0.4, -1.0]))
    assert p.item() == 0.0 and torch.isfinite(loss)


def test_selector_shapes_and_temperature():
    sel = Selector(16, 8)
    e = sel.nodes(torch.randn(5, 16))
    q = sel.queries(torch.randn(2, 8))
    s = sel.scores(q, e)
    assert s.shape == (2, 5) and s.abs().max() <= 1 / 0.07 + 1e-4
    assert sum(p.numel() for p in sel.g.parameters()) == 16 * 1024 + 1024 + 2 * 1024 + 1024 * 256 + 256
