import numpy as np
import torch

from treeground.calib import calibrate_rejection, rejection_objective
from treeground.metrics import area_bin, bootstrap_ci, ranked_ap


def brute_ap(scores, positive, valid):
    idx = [i for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)) if valid[i]]
    hits, total = 0, 0.0
    for r, i in enumerate(idx, 1):
        if positive[i]:
            hits += 1
            total += hits / r
    return total / max(1, sum(positive[i] for i in idx)), float(positive[idx[0]])


def test_ranked_ap_matches_brute_force_with_ignored_items():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 50
        s = rng.normal(size=(1, n)).astype(np.float32)
        pos = rng.random((1, n)) < 0.2
        valid = rng.random((1, n)) < 0.8
        pos[0, np.flatnonzero(valid[0])[0]] = True
        ap, r1 = ranked_ap(torch.tensor(s), torch.tensor(pos), torch.tensor(valid))
        ref_ap, ref_r1 = brute_ap(s[0], pos[0] & valid[0], valid[0])
        assert abs(ap.item() - ref_ap) < 1e-6 and r1.item() == ref_r1


def test_ignored_item_does_not_count_against_positive():
    s = torch.tensor([[0.9, 0.8, 0.1]])
    pos = torch.tensor([[False, True, False]])
    ap_all, _ = ranked_ap(s, pos, torch.ones_like(pos))
    ap_ign, r1 = ranked_ap(s, pos, torch.tensor([[False, True, True]]))
    assert abs(ap_all.item() - 0.5) < 1e-6 and abs(ap_ign.item() - 1.0) < 1e-6 and r1.item() == 1.0


def test_bootstrap_is_deterministic_and_brackets_mean():
    v = np.r_[np.ones(30), np.zeros(70)]
    g = np.repeat(np.arange(25), 4)
    a, b = bootstrap_ci(v, g, 1000, 0), bootstrap_ci(v, g, 1000, 0)
    assert a == b and a[0] <= v.mean() <= a[1]


def test_rejection_objective_by_hand():
    grid = np.array([0.0, 0.5, 0.9])
    obj = rejection_objective(grid, np.array([0.8, 0.3]), np.array([0.7, 0.4]), np.array([0.6, -np.inf]))
    # tau=0: accept all present (0.7+0.4), absent: one accept (-1), one empty set abstains (+1)
    assert np.allclose(obj, [(1.1 - 1 + 1) / 4, (0.7 - 1 + 1) / 4, (0 + 2) / 4])
    tau, _ = calibrate_rejection(np.array([0.8, 0.3]), np.array([0.7, 0.4]), np.array([0.6, -np.inf]), grid=grid)
    assert tau == 0.9


def test_area_bins():
    assert area_bin(32 ** 2 - 1) == "small" and area_bin(32 ** 2) == "medium" and area_bin(96 ** 2) == "large"
