import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treeground import masks as M
from treeground.calib import calibrate_rejection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("grefcoco_style_eval", ROOT / "scripts" / "grefcoco_style_eval.py")
GS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GS)
BS = {"n": 50, "seed": 0}


def rle(y0, y1, x0, x1, h=10, w=10):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return M.encode(m[None])[0]


def test_auc_matches_pairwise_count_and_roc_ends():
    rng = np.random.default_rng(0)
    pos, neg = np.round(rng.uniform(0, 1, 40), 1), np.round(rng.uniform(0, 1, 30) - 0.2, 1)
    pos[0] = -np.inf
    pairs = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    assert GS.auc(pos, neg) == pytest.approx(pairs)
    assert GS.auc(np.array([1.0]), np.array([])) is None
    r = GS.roc(pos, neg)
    assert (r.tpr.iloc[-1], r.fpr.iloc[-1]) == (1.0, 1.0) and r.threshold.is_monotonic_decreasing
    assert np.trapezoid(np.r_[0, r.tpr], np.r_[0, r.fpr]) == pytest.approx(pairs)


def frame(n_img=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        for j in range(4):
            present = j < 2
            rows.append({"qid": f"q{i}_{j}", "image": f"grefcoco:{i}", "present": present,
                         "n_targets": (1 + (j == 1)) if present else 0, "n_nodes": 5,
                         "score": rng.uniform(0.3, 1.0) if present else rng.uniform(0.0, 0.8),
                         "cand_iou": rng.uniform(0, 1) if present else 0.0, "best_iou": 0.0})
    df = pd.DataFrame(rows)
    df["native_accept"] = df.score >= 0.5
    return df


def test_folds_are_image_level_balanced_and_deterministic():
    f = GS.image_folds([f"k{i}" for i in range(11)] * 2, 2, 0)
    assert len(f) == 11 and sorted(np.bincount(list(f.values()))) == [5, 6]
    assert f == GS.image_folds([f"k{i}" for i in reversed(range(11))], 2, 0)


def test_two_fold_tau_is_calibrated_on_the_other_fold_only():
    df = frame()
    out = GS.two_fold(df, 1.0, BS, 2, 0)
    fold = df.image.map(GS.image_folds(df.image, 2, 0))
    for f in (0, 1):
        cal = df[fold != f]
        tau, _ = calibrate_rejection(cal.score[cal.present].to_numpy(), cal.cand_iou[cal.present].to_numpy(),
                                     cal.score[~cal.present].to_numpy(), 1.0)
        b = out["folds"][f"fold{f}"]
        test = df[fold == f]
        assert b["tau_calibrated_on_other_folds"] == tau
        assert b["no_target_abstention_rate"] == pytest.approx((test.score[~test.present] < tau).mean())
        assert b["present"] == int(test.present.sum())
    pooled = out["pooled_held_out"]
    assert pooled["present"] + pooled["no_target"] == len(df)


def test_block_scores_rejected_present_as_zero():
    df = frame(4)
    acc = df.score.to_numpy() >= 0.6
    b = GS.block(df, acc, BS)
    p = df.present.to_numpy()
    assert b["present_gIoU"]["mean"] == pytest.approx(np.where(acc, df.cand_iou, 0)[p].mean())
    assert b["present_acceptance_rate"] == pytest.approx(acc[p].mean())
    assert b["no_target_abstention_rate"] == pytest.approx((~acc[~p]).mean())
    assert b["present_gIoU_multi_target"]["n"] == 4
    s = GS.summarize(df, 0.5, True, 1.0, BS, 2, 0)
    assert s["no_rejection"]["present_acceptance_rate"] == 1.0 and s["separation"]["auc"] is not None
    assert GS.summarize(df, float("nan"), False, 1.0, BS, 2, 0)["two_fold_calibrated"] is None


def test_ours_and_baseline_rows():
    g = pd.DataFrame([{"qid": "a", "image": "grefcoco:1", "present": True, "n_targets": 1},
                      {"qid": "b", "image": "grefcoco:1", "present": False, "n_targets": 0}])
    nodes = {"rles": [rle(0, 5, 0, 5), rle(0, 10, 0, 10)]}
    gts = {"a": rle(0, 5, 0, 5)}
    r = GS.ours_rows(g, nodes, [np.array([0.2, 0.7]), np.array([0.9, 0.1])], gts, 0.5)
    assert (r[0]["score"], r[0]["cand_iou"], r[0]["best_iou"], r[0]["native_accept"]) == (0.7, 0.25, 1.0, True)
    assert (r[1]["cand_iou"], r[1]["native_accept"]) == (0.0, True)
    full = np.zeros((10, 10), bool)
    full[:5, :5] = True
    preds = [(full, 0.3, "", False), (np.zeros((10, 10), bool), 0.9, "", True)]
    r = GS.baseline_rows(g, preds, gts, (10, 10))
    assert (r[0]["cand_iou"], r[0]["native_accept"], r[0]["score"]) == (1.0, False, 0.3)
    assert (r[1]["native_accept"], r[1]["score"], r[1]["n_nodes"]) == (False, -np.inf, 0)


def gref_fixture(tmp_path):
    refs = [{"ref_id": 1, "image_id": 7, "split": "val", "ann_id": [100, 101], "no_target": False,
             "sentences": [{"sent_id": 5, "sent": " two cups "}]},
            {"ref_id": 2, "image_id": 7, "split": "val", "ann_id": [-1], "no_target": True,
             "sentences": [{"sent_id": 6, "sent": "a dog"}, {"sent_id": 8, "sent": "the cat"}]},
            {"ref_id": 3, "image_id": 9, "split": "val", "ann_id": 102, "sentences": [{"sent_id": 9, "sent": "x"}]},
            {"ref_id": 4, "image_id": 7, "split": "train", "ann_id": 100, "sentences": [{"sent_id": 1, "sent": "y"}]}]
    anns = [{"id": 100, "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]},
            {"id": 101, "segmentation": [[6, 6, 9, 6, 9, 9, 6, 9]]}]
    (tmp_path / "grefs(unc).json").write_text(json.dumps(refs))
    (tmp_path / "instances.json").write_text(json.dumps({"images": [{"id": 7, "height": 10, "width": 10,
                                                                    "file_name": "a.jpg"},
                                                                   {"id": 9, "height": 10, "width": 10,
                                                                    "file_name": "b.jpg"}], "annotations": anns}))
    wl = pd.DataFrame({"image_key": ["grefcoco:7"], "role": ["grefcoco_val_nt"], "sha1": ["s7"], "path": ["p7"]})
    return anns, wl


def test_build_queries_unions_targets_and_skips_images_without_nodes(tmp_path):
    anns, wl = gref_fixture(tmp_path)
    df, skipped = GS.build_queries({"grefcoco_style": {"split": "val"}, "data": {"grefcoco": str(tmp_path)}}, wl)
    assert df.qid.tolist() == ["grefcoco|val|5", "grefcoco|val|6", "grefcoco|val|8"]
    assert df.present.tolist() == [True, False, False] and df.text.iloc[0] == "two cups"
    assert skipped == {"queries_without_nodes": 1, "images_without_nodes": 1}
    u = GS.gt_union({a["id"]: a for a in anns}, json.loads(df.ann_ids.iloc[0]), 10, 10)
    parts = [M.from_annotation(a["segmentation"], 10, 10) for a in anns]
    assert M.areas([u])[0] == M.areas(parts).sum() > 0


def test_abstention_table_rows(tmp_path):
    spec_mt = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
    MT = importlib.util.module_from_spec(spec_mt)
    spec_mt.loader.exec_module(MT)
    m = tmp_path / "metrics"
    m.mkdir()
    s = GS.summarize(frame(), 0.5, True, 1.0, BS, 2, 0)
    (m / "grefcoco_style_s2_merge.json").write_text(json.dumps({**s, "paco_template_reference": {"auc": 0.99}},
                                                               default=float))
    b = GS.summarize(frame(), float("nan"), False, 1.0, BS, 2, 0)
    (m / "grefcoco_style_sa2va_merge.json").write_text(json.dumps(b, default=float))
    MT.write(tmp_path / "t", "abstention_style_matched", *MT.abstention_style_matched(m))
    rows = pd.read_csv(tmp_path / "t" / "abstention_style_matched.csv", dtype=str)
    assert rows[rows.method == "S2"].tau.tolist() == ["stage tau", "held-out (fold0)", "held-out (fold1)",
                                                       "held-out (pooled)", "no rejection"]
    assert rows.auc_paco.iloc[0] == "0.990" and rows[rows.method == "Sa2VA"].tau.tolist() == ["stage tau"]


def test_shard_and_merge_with_mocked_model(tmp_path, monkeypatch):
    import treeground.identity as identity
    anns, wl = gref_fixture(tmp_path)
    cfg = {"grefcoco_style": {"split": "val", "smoke_images": 1, "folds": 2, "fold_seed": 0, "batch_size": 4},
           "data": {"grefcoco": str(tmp_path)}, "cache_root": str(tmp_path / "cache"),
           "runs_root": str(tmp_path / "runs"), "models": {"qwen": {"repo": "q", "revision": "r"}},
           "proposals": {"node_dedup": 0.9, "dtype": "bfloat16"}, "v2": {"bootstrap": BS, "calib_lambda": 1.0}}
    nodes = {"h": 10, "w": 10, "rles": [rle(0, 5, 0, 5), rle(0, 10, 0, 10)]}
    sigs = {"two cups": [0.3, 0.8], "a dog": [0.6, 0.2], "the cat": [0.1, 0.4]}
    monkeypatch.setattr(identity, "manifest_inputs", lambda c, s: (tmp_path, None, "h", wl))
    monkeypatch.setattr(GS, "require_approved", lambda c, ids: None)
    monkeypatch.setattr(GS, "run_ref", lambda c, m: ({"run": "s1/x"}, 0.5))
    monkeypatch.setattr(GS, "ours_frozen", lambda *a: lambda g: (nodes, [np.array(sigs[t]) for t in g.text]))
    monkeypatch.setattr(GS, "paco_template", lambda *a: None)
    GS.run_shard(cfg, "s1", 0, 1, False)
    GS.merge(cfg, "s1", 1, False)
    run = next((tmp_path / "runs" / "grefcoco_style_s1_merge").glob("*"))
    out = json.loads((run / "metrics.json").read_text())
    q = pd.read_csv(run / "queries.csv").set_index("qid")
    u = GS.gt_union({a["id"]: a for a in anns}, [100, 101], 10, 10)
    assert q.loc["grefcoco|val|5", "cand_iou"] == pytest.approx(M.areas([u])[0] / 100)
    assert q.loc["grefcoco|val|5", "score"] == 0.8
    b = out["at_stage_tau"]
    assert (b["present"], b["no_target"], b["no_target_abstention_rate"], b["present_acceptance_rate"]) == (1, 2, 0.5, 1)
    assert out["separation"]["auc"] == 1.0 and out["stage_tau"] == 0.5
