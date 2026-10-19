import importlib.util
from pathlib import Path

import numpy as np
import torch

from treeground import sampling as S
from treeground import tree as T

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_image(v2, n, labels, rng, d=12):
    tree = T.from_parents(np.array([n] + [0] * (n - 2) + [n, -1]))  # node 0 contains 1..n-2; node n-1 alone
    lab = np.array(labels + [""] * (n - len(labels)), dtype=object)
    parts = np.array([l.split(":")[1] if l else "" for l in lab], dtype=object)
    rec = v2.ImageRec("sha", "key", torch.randn(n, d), tree, lab, parts, np.zeros(n, dtype=bool))
    ious = np.zeros((2, n))
    ious[0, 0], ious[0, n - 1] = 0.9, 0.6
    ious[1, 2], ious[1, 3] = 0.7, 0.55
    path = T.max_iou_path(tree, ious)
    rec.queries = [v2.QRec("q", "chain", torch.randn(2, 8), S.chain_step_targets(path, ious), path)]
    return rec


def test_v2_batch_terms_construct_valid_candidate_sets(monkeypatch):
    v2 = load_script("v2_train_eval")
    rng = np.random.default_rng(0)
    recs = [fixture_image(v2, 90, ["mug:handle", "mug:body"], rng), fixture_image(v2, 70, ["mug:handle"], rng)]
    absent = v2.ImageRec("a", "a", torch.randn(30, 12), T.from_parents(np.array([30] * 30 + [-1])),
                         np.full(30, "", dtype=object), np.full(30, "", dtype=object), np.zeros(30, bool))
    absent.absent = True
    absent.queries = [v2.QRec("n", "absent", torch.randn(1, 8), [S.StepTarget(np.zeros(0, np.int64),
                                                                              np.zeros(0, np.int64))])]
    sel = v2.Selector(12, 8)
    seen = []
    real = v2.selection_loss

    def spy(pos, neg, gamma):
        seen.append((pos.detach().clone(), neg.detach().clone()))
        return real(pos, neg, gamma)

    monkeypatch.setattr(v2, "selection_loss", spy)
    stats = {k: 0 for k in ["terms_object", "terms_part", "terms_chain", "terms_text", "terms_absent",
                            "empty_negatives", "shortfall_in_image", "shortfall_cross", "chain_topup"]}
    batch = [(recs[0], True), (recs[1], False), (absent, False)]
    text = {"mug:handle": torch.randn(8)}
    terms = v2.batch_terms(sel, batch, text, rng, "cpu", 2.0, stats)
    assert len(terms) == 1 + 2 + 1
    assert stats["terms_text"] == 1 and stats["terms_chain"] == 2 and stats["terms_absent"] == 1
    assert all(torch.isfinite(t) for t in terms)
    loss = torch.stack(terms).mean()
    loss.backward()
    assert all(p.grad is not None for p in sel.parameters())


def test_v2_chain_step_two_negatives_come_from_subtree_first():
    v2 = load_script("v2_train_eval")
    rec = fixture_image(v2, 90, [], np.random.default_rng(0))
    q = rec.queries[0]
    assert q.path == (0, 2)
    sub = rec.tree.strict_descendants(0)
    inside = S.eligible(sub, q.steps[1].pos, q.steps[1].ignore)
    d = S.chain_negatives(inside, S.eligible(np.setdiff1d(np.arange(rec.n), sub), q.steps[1].pos, q.steps[1].ignore),
                          np.zeros(rec.n), np.random.default_rng(0))
    assert set(d.ids) <= set(sub) and 2 not in d.ids and 3 not in d.ids


def test_v4_controller_and_count_controlled_diagnostic():
    v4 = load_script("v4_tree_value")
    n = 40
    tree = T.from_parents(np.array([n] + [0] * 19 + [n] * 20 + [-1]))
    rand = [T.randomised_tree(tree, s) for s in range(5)]
    ious = np.zeros((2, n))
    ious[0, 0], ious[1, 5] = 0.9, 0.8
    sig = np.full((2, n), 0.1)
    sig[0, 0], sig[1, 5] = 0.9, 0.95
    chains = [{"qid": "c", "image": "i", "sha1": "s", "tree": tree, "rand": rand, "ious": ious, "sig": sig}]
    df, anchors = v4.natural(chains, 0.5)
    true = df[df.structure == "true"].iloc[0]
    assert true.chain and true.candidates == 19 and anchors == [19]
    assert df[df.structure == "flat"].iloc[0].candidates == n
    diag = v4.count_controlled(chains, 0.5, [0, 1, 2, 3, 4], 10)
    cov = diag["sampled_pool_target_coverage"]["true"]["per_seed"]
    acc = diag["step2_success"]["true"]["per_seed"]
    assert all(a <= c for a, c in zip(acc, cov))
    again = v4.count_controlled(chains, 0.5, [0, 1, 2, 3, 4], 10)
    assert again["per_seed"] == diag["per_seed"]
    sig[0, 0] = 0.2  # step-1 rejection keeps the root as anchor
    df2, anchors2 = v4.natural(chains, 0.5)
    assert anchors2 == [-1] and not df2.chain.any() and (df2[df2.structure == "true"].candidates == n).all()
