import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from treeground import rerank as RR
from treeground import tree as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rerank_script", ROOT / "scripts" / "rerank.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

BS = {"n": 50, "seed": 0}


def features(sig, k):
    # 0 -> 1 -> 2, 3 at the top level, root 4
    tree = T.from_parents(np.array([4, 0, 1, 4, -1]))
    sig = np.asarray(sig, np.float64)
    return RR.candidate_features(sig, np.log(sig / (1 - sig)), tree, np.array([0.5, 0.2, 0.05, 0.1]),
                                 np.array([0.9, 0.8, -0.1, 0.7]), np.array([0.95, 0.9, 0.92, 0.97]), k)


def test_candidate_features_follow_sigma_order_and_tree():
    f = features([0.2, 0.9, 0.5, 0.7], 3)
    assert f.node.tolist() == [1, 3, 2] and f["rank"].tolist() == [0, 1, 2]
    assert np.allclose(f.sigma_gap, [0.0, 0.2, 0.4])
    assert f.is_top1.tolist() == [1, 0, 0]
    assert f.descendant_of_top1.tolist() == [0, 0, 1] and f.ancestor_of_top1.tolist() == [0, 0, 0]
    assert np.allclose(f.parent_sigma, [0.2, 0.0, 0.9]) and f.parent_is_root.tolist() == [0, 1, 0]
    assert np.allclose(f.best_child_sigma, [0.5, 0.0, 0.0])
    assert f.depth.tolist() == [2, 1, 3] and np.allclose(f.log_descendants, np.log1p([1, 0, 0]))
    assert np.allclose(f.log_sam_iou, np.log([0.8, 0.7, RR.SAM_IOU_FLOOR]))
    assert np.allclose(f.log_area_frac, np.log([0.2, 0.1, 0.05]))
    g = features([0.2, 0.9, 0.5, 0.7], 10)
    assert len(g) == 4 and g.set_index("node").ancestor_of_top1[0] == 1


def test_top_k_ties_go_to_lower_index_like_argmax():
    sig = np.array([0.3, 0.8, 0.8, 0.1])
    assert RR.top_k(sig, 3).tolist() == [1, 2, 0] and RR.top_k(sig, 1)[0] == np.argmax(sig)


def test_folds_are_per_image_deterministic_and_balanced():
    images = [f"img{i % 23}" for i in range(200)]
    a, b = RR.assign_folds(images, 5, 0), RR.assign_folds(list(reversed(images)), 5, 0)
    assert a == b and set(a) == set(images)
    assert sorted(np.bincount(list(a.values()))) == [4, 4, 5, 5, 5]
    assert RR.assign_folds(images, 5, 1) != a


def synthetic(n_images=40, q_per_image=5, k=6, seed=0, signal="log_sam_iou"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        for q in range(q_per_image):
            f = {c: rng.normal(size=k) for c in RR.FEATURES}
            f["rank"] = np.arange(k)
            iou = 1 / (1 + np.exp(-3 * f[signal]))
            rows.append(pd.DataFrame({**f, "qid": f"q{i}_{q}", "image": f"img{i}", "kind": ["object", "part"][q % 2],
                                      "node": np.arange(k), "iou": iou, "inter": (iou * 100).round(),
                                      "union": 100}))
    return pd.concat(rows, ignore_index=True)


def test_cross_validation_never_scores_a_query_with_a_model_that_saw_its_image():
    c = synthetic()
    fold_of = RR.assign_folds(c.image, 5, 0)
    seen = []

    def fit(d):
        seen.append(set(d.image))
        return set(d.image)

    def predict(train_images, d):
        assert not set(d.image) & train_images
        return np.zeros(len(d))

    scores, models = RR.cross_validate(c, c.image.map(fold_of).to_numpy(), fit, predict)
    assert len(models) == 5 and not np.isnan(scores).any()
    assert all(len(s) == 32 for s in seen)


def test_pick_takes_highest_score_then_lower_rank():
    c = pd.DataFrame({"qid": ["a", "a", "a", "b", "b"], "rank": [0, 1, 2, 0, 1], "iou": [0.1, 0.7, 0.7, 0.4, 0.2]})
    p = RR.pick(c, c["iou"])
    assert p["a"] == 1 and p["b"] == 3
    assert RR.pick(c, -c["rank"].to_numpy()).tolist() == [0, 3]


def test_rankers_recover_the_informative_feature_out_of_fold():
    c = synthetic()
    fold = c.image.map(RR.assign_folds(c.image, 5, 0)).to_numpy()
    oracle = c.loc[RR.pick(c, c["iou"]).to_numpy(), "iou"].mean()
    top1 = c.loc[RR.pick(c, -c["rank"].to_numpy()).to_numpy(), "iou"].mean()
    for fit in (RR.listwise_fit(1e-3), RR.pointwise_fit(1e-3, 0.5)):
        s, models = RR.cross_validate(c, fold, fit, RR.linear_predict)
        assert c.loc[RR.pick(c, s).to_numpy(), "iou"].mean() > top1 + 0.5 * (oracle - top1)
        imp = RR.importances(models, RR.design(c.head(1))[1])
        assert imp[0]["feature"] == "log_sam_iou"
    s, ws = RR.cross_validate(c, fold, lambda d: RR.fit_rule(d, [0.0, 1.0, 100.0]), lambda w, d: RR.rule_score(d, w))
    assert ws == [100.0] * 5
    c2 = synthetic(signal="logit")
    assert RR.fit_rule(c2, [0.0, 1.0, 100.0]) == 0.0


def test_listwise_rows_drop_queries_without_overlap_and_mark_one_target():
    c = pd.DataFrame({"qid": ["a", "a", "b", "b"], "rank": [0, 1, 0, 1], "iou": [0.3, 0.3, 0.0, 0.0]})
    d, target = RR.listwise_rows(c)
    assert d.qid.tolist() == ["a", "a"] and target.tolist() == [True, False]


def test_blocks_count_queries_without_candidates_and_keep_accept():
    qs = pd.DataFrame({"qid": ["a", "b", "c"], "image": ["i0", "i0", "i1"], "kind": ["object", "part", "part"],
                       "gt_area": [100, 50, 80], "accept": [True, False, True]})
    cand = pd.DataFrame({"qid": ["a", "a", "b"], "rank": [0, 1, 0], "inter": [50, 100, 25], "union": [100, 100, 50],
                         "iou": [0.5, 1.0, 0.5]})
    inter, union = RR.picked_overlap(qs, cand, RR.pick(cand, cand["iou"]))
    assert inter.tolist() == [100, 25, 0] and union.tolist() == [100, 50, 80]
    absent = pd.DataFrame({"image": ["i2"], "accept": [False]})
    b = S.blocks(qs, absent, inter, union, BS)
    assert np.isclose(b["no_rejection"]["gIoU"]["mean"], 0.5)
    assert np.isclose(b["calibrated"]["gIoU"]["mean"], 1 / 3)
    assert np.isclose(b["calibrated"]["gIoU_present_and_no_target"]["mean"], 0.5)
    assert np.isclose(b["calibrated"]["cIoU"]["value"], 100 / 230)
