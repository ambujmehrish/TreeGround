import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from treeground import sampling as S
from treeground import tree as T
from treeground.hyperbolic import (LorentzSelector, cone_violation, dist0, expmap0, make_selector, pairwise_dist,
                                   time_component, tree_edges)
from treeground.run import load_config
from treeground.selector import Selector

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("euclid", "lorentz", "lorentz_cone")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def curv(c):
    return torch.tensor(c, dtype=torch.float64)


@pytest.mark.parametrize("c", [0.1, 1.0, 10.0])
def test_expmap_lands_on_hyperboloid_and_distance_is_a_metric(c):
    torch.manual_seed(0)
    k = curv(c)
    x = expmap0(torch.randn(6, 16) * 2, k)
    y = expmap0(torch.randn(5, 16) * 0.3, k)
    t = time_component(x, k)
    assert (((x * x).sum(-1) - t ** 2 + 1 / k).abs() <= 1e-12 * t ** 2 + 1e-12).all()
    d = pairwise_dist(x, y, k)
    assert (d >= 0).all()
    assert torch.allclose(d, pairwise_dist(y, x, k).T, atol=1e-9)
    assert pairwise_dist(x, x, k).diagonal().abs().max() < 1e-3  # float64 cancellation ~ sqrt(eps) * x_time
    o = torch.zeros(1, 16, dtype=torch.float64)
    assert torch.allclose(pairwise_dist(o, x, k)[0], dist0(x, k), atol=1e-7)
    # geodesic norm of expmap0(v) is |v|
    v = torch.randn(4, 16, dtype=torch.float64) * 0.5  # sqrt(c) |v| below the MAX_RADIUS clamp
    assert torch.allclose(dist0(expmap0(v, k), k), v.norm(dim=-1), atol=1e-7)
    z = expmap0(torch.randn(3, 16), k)
    assert (pairwise_dist(x, z, k) <= pairwise_dist(x, y, k)[:, :1] + pairwise_dist(y[:1], z, k) + 1e-7).all()


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0, 50.0])
def test_gradients_finite_near_origin_and_far(scale):
    torch.manual_seed(0)
    v = (torch.randn(4, 8) * scale).requires_grad_()
    w = (torch.randn(3, 8) * scale).requires_grad_()
    log_c = torch.zeros((), dtype=torch.float64, requires_grad=True)
    k = log_c.exp()
    x, y = expmap0(v, k), expmap0(w, k)
    loss = pairwise_dist(x, y, k).sum() + dist0(x, k).sum() + cone_violation(x[:3], y, k).sum()
    loss.backward()
    for g in (v.grad, w.grad, log_c.grad):
        assert torch.isfinite(g).all()


def test_cone_loss_zero_for_nested_and_positive_for_outside():
    k = curv(1.0)
    u = torch.nn.functional.normalize(torch.randn(1, 8, dtype=torch.float64), dim=-1)
    parent = expmap0(u * 0.5, k)
    children = expmap0(torch.cat([u * 1.0, u * 3.0]), k)
    assert torch.all(cone_violation(parent.expand(2, -1), children, k) < 1e-6)
    side = torch.zeros(1, 8, dtype=torch.float64)
    side[0, int(torch.argmin(u.abs()))] = 1.0
    off = expmap0(u * 0.5 + side * 2.0, k)
    assert cone_violation(expmap0(u * 2.0, k), off, k).item() > 0.1


def test_tree_edges_skip_the_virtual_root():
    p, c = tree_edges(np.array([4, 0, 0, 4, -1]))
    assert list(zip(p, c)) == [(0, 1), (0, 2)]


def toy_batch(d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    V2 = load_script("v2_train_eval")
    tree = T.from_parents(np.array([5, 0, 0, 1, 5, -1]))
    labels = np.array(["mug:handle", "", "mug:body", "", ""], dtype=object)
    recs = []
    for i in range(3):
        r = V2.ImageRec(f"s{i}", f"k{i}", torch.randn(5, d, generator=g), tree, labels,
                        np.array(["handle", "", "body", "", ""], dtype=object), np.zeros(5, bool))
        r.queries = [V2.QRec("q", "chain", torch.randn(2, d, generator=g),
                             [S.StepTarget(np.array([0]), np.zeros(0, np.int64)),
                              S.StepTarget(np.array([3]), np.zeros(0, np.int64))], (0, 3))]
        recs.append(r)
    return V2, [(r, False) for r in recs], {"mug:handle": torch.randn(d, generator=g)}


def test_lorentz_selector_matches_selector_interface_in_the_v2_loss():
    V2, batch, text = toy_batch()
    for sel in (Selector(12, 12), LorentzSelector(12, 12)):
        e, q = sel.nodes(torch.randn(7, 12)), sel.queries(torch.randn(2, 12))
        s = sel.scores(q, e)
        assert s.shape == (2, 7) and s.dtype == torch.float32 and isinstance(sel.tau, float)
        stats = {k: 0 for k in ["terms_chain", "empty_negatives", "shortfall_in_image", "shortfall_cross",
                                "chain_topup"]}
        loss = torch.stack(V2.batch_terms(sel, batch, text, np.random.default_rng(0), "cpu", 2.0, stats)).mean()
        loss.backward()
        assert torch.isfinite(loss) and stats["terms_chain"] == 6
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in sel.parameters())
        clone = type(sel)(12, 12)
        clone.load_state_dict(sel.state_dict())
        assert torch.allclose(clone.scores(clone.queries(q.new_ones(1, 12).float()), e),
                              sel.scores(sel.queries(q.new_ones(1, 12).float()), e))


def test_lorentz_heads_start_from_the_euclidean_initialisation():
    torch.manual_seed(0)
    a = Selector(12, 10)
    torch.manual_seed(0)
    b = LorentzSelector(12, 10)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_train_cone_loss_and_optimizer_groups():
    G = load_script("s1_geom_train")
    _, batch, _ = toy_batch()
    sel = LorentzSelector(12, 12)
    loss = G.cone_loss(sel, batch, "cpu", 0.1)
    loss.backward()
    assert torch.isfinite(loss) and loss >= 0 and torch.isfinite(sel.g.net[0].weight.grad).all()
    opt = G.optimizer(sel, {"lr": 3e-4, "weight_decay": 0.01})
    decay = {id(p) for grp in opt.param_groups if grp["weight_decay"] > 0 for p in grp["params"]}
    assert not decay & {id(p) for p in sel.scalar_parameters()}
    assert len(decay) + len(sel.scalar_parameters()) == len(list(sel.parameters()))
    euc = G.optimizer(Selector(12, 12), {"lr": 3e-4, "weight_decay": 0.01})
    assert len(euc.param_groups) == 1 and euc.param_groups[0]["weight_decay"] == 0.01


def test_config_arms(monkeypatch):
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    base = load_config(ROOT / "configs" / "s1.yaml")
    v4 = load_config(ROOT / "configs" / "v4_s1.yaml")
    cfgs = {a: load_config(ROOT / "configs" / f"s1_geom_{a}.yaml") for a in ARMS}
    for arm, cfg in cfgs.items():
        assert cfg["geometry"]["arm"] == arm and cfg["inputs"]["selector_stage"] == f"s1_geom_{arm}"
        assert cfg["v2"] == base["v2"] and cfg["s1"] == base["s1"] and cfg["v4"] == v4["v4"]
        assert cfg["s1"]["steps"] == 20000 and cfg["s1"]["eval_every"] == 2000
        sel = make_selector(cfg["geometry"], 8, 8, cfg["v2"]["temperature"])
        assert isinstance(sel, Selector if arm == "euclid" else LorentzSelector)
    assert cfgs["euclid"]["geometry"]["cone_weight"] == cfgs["lorentz"]["geometry"]["cone_weight"] == 0
    assert cfgs["lorentz_cone"]["geometry"]["cone_weight"] > 0
    assert {k: v for k, v in cfgs["lorentz"]["geometry"].items() if k not in ("arm", "cone_weight")} == \
           {k: v for k, v in cfgs["lorentz_cone"]["geometry"].items() if k not in ("arm", "cone_weight")}


def test_ablation_geometry_table(tmp_path):
    MT = load_script("make_tables")
    m = tmp_path / "metrics"
    m.mkdir()

    def ci(x):
        return {"mean": x, "ci95": [x - 0.01, x + 0.01], "n": 10}

    chain = {"natural_no_rejection_d12": {"all": {t: {"chain_success_common": ci(0.5)} for t in ("flat", "true")}}}
    files = {"s1_geom_lorentz": {"no_rejection_present": {"gIoU": ci(0.36), "by_kind": {"part": ci(0.25)}},
                                 "ii_selector_success": {"all": ci(0.64)}},
             "v4_s1_geom_lorentz": {**chain, "geometry_diagnostics": {"curvature": 1.7, "cone_violation_rate": 0.4}},
             "e2e_variants": {"stages": {"s1": {"no_rejection_present": {"gIoU": ci(0.359)}}}}}
    for k, v in files.items():
        (m / f"{k}.json").write_text(json.dumps(v))
    MT.write(tmp_path / "t", "ablation_geometry", *MT.ablation_geometry(m))
    with open(tmp_path / "t" / "ablation_geometry.csv", newline="") as f:
        rows = {r["arm"]: r for r in csv.DictReader(f)}
    assert list(rows) == ["S1 (published, Euclidean)", "Euclidean (rerun)", "Lorentz", "Lorentz + cone"]
    lor = rows["Lorentz"]
    assert (lor["gIoU"], lor["gIoU_part"], lor["true"], lor["random"]) == ("0.360", "0.250", "0.500", "–")
    assert (lor["curvature"], lor["cone_violation"]) == ("1.700", "0.400")
    assert rows["S1 (published, Euclidean)"]["gIoU"] == "0.359"
    assert all(v == "–" for k, v in rows["Lorentz + cone"].items() if k != "arm")
    assert "1.70 " in (tmp_path / "t" / "ablation_geometry.tex").read_text()
