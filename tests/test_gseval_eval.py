import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script
from treeground import masks as M


@pytest.fixture(scope="module")
def gs():
    return load_script("gseval_eval")


def rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def write_bench(tmp_path):
    root = tmp_path / "gseval"
    (root / "unlabeled2017").mkdir(parents=True)
    rows = []
    masks = [rect(20, 30, 2, 10, 2, 12), rect(20, 30, 10, 18, 14, 28), rect(20, 30, 0, 20, 0, 30)]
    for i, (cid, m) in enumerate(zip((2, 3, 1), masks)):
        rle = M.encode(m[None])[0]
        rows.append({"idx": i, "image_path": f"unlabeled2017/0000{i % 2}.jpg", "class_id": cid, "label": f"l{i}",
                     "caption": f"the thing {i} ", "box": [0, 0, 1, 1],
                     "segmentation": {"size": rle["size"], "counts": rle["counts"]}})
    (root / "GroundingSuite-Eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root, masks


def test_expressions_read_the_jsonl(gs, tmp_path):
    root, masks = write_bench(tmp_path)
    df = gs.expressions({"data": {"gseval": str(root)}})
    assert list(df.qid) == ["gseval|0", "gseval|1", "gseval|2"] and df.image.nunique() == 2
    assert list(df.text) == ["the thing 0", "the thing 1", "the thing 2"] and list(df.class_id) == [2, 3, 1]
    assert (df.h == 20).all() and (df.w == 30).all() and df.path.iloc[0].endswith("unlabeled2017/00000.jpg")
    bad = root / "GroundingSuite-Eval.jsonl"
    bad.write_text(bad.read_text().replace('"class_id": 2', '"class_id": 9'))
    with pytest.raises(Exception):
        gs.expressions({"data": {"gseval": str(root)}})


def test_iou_rules_follow_the_official_scorer(gs, tmp_path):
    root, masks = write_bench(tmp_path)
    df = gs.expressions({"data": {"gseval": str(root)}})
    a, b, whole = (M.encode(m[None])[0] for m in masks)
    # expression 0: top candidate is the right mask, second a wrong one; 1: top is wrong, the right one scores 0.4;
    # 2: no candidate at all
    df["cand_nodes"] = [json.dumps([[0.9, a], [0.6, b]]), json.dumps([[0.9, a], [0.4, b]]), json.dumps([])]
    top = gs.iou_rows(df, "cand_nodes", "top1", -1.0)
    assert top.iou.round(3).tolist() == [1.0, 0.0, 0.0] and top.pred_empty.tolist() == [False, False, True]
    st = gs.iou_rows(df, "cand_nodes", "set", 0.5)
    area_a, area_b = masks[0].sum(), masks[1].sum()
    assert st.iou.round(4).tolist() == [round(area_a / (area_a + area_b), 4), 0.0, 0.0]   # union a|b vs a; a vs b
    st2 = gs.iou_rows(df, "cand_nodes", "set", 0.3)
    assert st2.iou.round(4).tolist()[1] == round(area_b / (area_a + area_b), 4)             # a|b vs b
    res = gs.summarize(top, {"n": 20, "seed": 0})
    assert res["all"]["expressions"] == 3 and res["all"]["empty_predictions"] == 1
    assert set(res) == {"all", "part", "multi", "stuff"} and res["part"]["gIoU"]["mean"] == 1.0
    assert res["multi"]["gIoU"]["mean"] == 0.0 and res["stuff"]["cIoU"] == 0.0


def test_gres_threshold_reads_the_checkpoints_val_rule(gs, tmp_path, monkeypatch):
    run = tmp_path / "gres_x"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"results": {"nodes/set": {"t_val": 0.875}, "nodes/top1": {"t_val": 0.9}}}))
    monkeypatch.setattr(gs, "resolve_run", lambda cfg, stage, ref, smoke: run)
    monkeypatch.setattr(gs, "stage_name", lambda cfg, method: "x")
    assert gs.gres_threshold({}, "ours") == 0.875
    (run / "metrics.json").write_text(json.dumps({"results": {"official": {}}}))
    with pytest.raises(Exception):
        gs.gres_threshold({}, "ours")


def test_baseline_rows_abstain_as_empty(gs, monkeypatch):
    import b_baselines as B
    m = rect(20, 30, 0, 5, 0, 5)
    monkeypatch.setitem(B.PREDICTORS, "fake", lambda cfg: (lambda image, texts: [(m, 0.9, "", True), (m, 0.1, "", False)]))
    out = gs.Baseline({}, "fake")(np.zeros((20, 30, 3), np.uint8), ["a", "b"])
    assert len(out[0]["pred"]) == 1 and out[1]["pred"] == []
    assert out[0]["pred"][0][0] == 1.0 and M.areas([out[0]["pred"][0][1]])[0] == m.sum()


def test_merge_scores_ours_with_the_gres_threshold(gs, tmp_path, monkeypatch):
    root, masks = write_bench(tmp_path)
    a, b, whole = (M.encode(m[None])[0] for m in masks)
    cfg = {"data": {"gseval": str(root)}, "gseval_eval": {"num_shards": 1}, "v2": {"bootstrap": {"n": 10, "seed": 0}},
           "cache_root": str(tmp_path / "cache")}
    df = gs.expressions(cfg)
    df["cand_nodes"] = [json.dumps([[0.9, a], [0.6, b]]), json.dumps([[0.95, b]]), json.dumps([[0.9, whole]])]
    d = tmp_path / "cache"
    (d / "shards").mkdir(parents=True)
    df[["qid", "image", "class_id", "gt", "cand_nodes"]].to_parquet(d / "shards" / "00000.parquet", index=False)
    monkeypatch.setattr(gs, "require_approved", lambda cfg, ids: None)
    monkeypatch.setattr(gs, "identity", lambda cfg, method: {"x": 1})
    monkeypatch.setattr(gs, "out_dir", lambda cfg, method, ident: d)
    monkeypatch.setattr(gs, "gres_threshold", lambda cfg, method: 0.5)
    monkeypatch.setattr(gs, "stage_name", lambda cfg, method: "s")
    runs = tmp_path / "runs"
    monkeypatch.setattr(gs, "start_run", lambda stage, cfg, smoke, meta: (runs / stage).mkdir(parents=True) or runs / stage)
    gs.merge(cfg, "ours")
    res = json.loads((runs / "gseval_s" / "metrics.json").read_text())["results"]
    assert res["counts_by_class_id"] == {"1": 1, "2": 1, "3": 1}
    assert res["nodes/top1"]["all"]["gIoU"]["mean"] == 1.0 and res["nodes/top1"]["threshold"] is None
    assert res["nodes/set"]["threshold"] == 0.5 and res["nodes/set"]["part"]["gIoU"]["mean"] < 1.0   # a|b for a
    assert res["nodes/set"]["multi"]["gIoU"]["mean"] == 1.0
