import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script


def test_compare_pairs_the_same_parts():
    tc = load_script("ta_checks")
    a = pd.DataFrame({"q": [0, 0, 1, 2], "image": ["i0", "i0", "i1", "i2"], "part": ["p", "r", "p", "p"], "label": "x",
                      "iou": [0.1, 0.6, 0.5, 0.0], "pred_area": [10, 20, 30, 0], "gt_area": [10, 40, 30, 5]})
    b = a.assign(iou=[0.7, 0.6, 0.3, 0.0])
    r = tc.compare(a, b)
    assert r["parts"] == 4 and np.isclose(r["micro_a"], 0.3) and np.isclose(r["micro_b"], 0.4)
    assert r["b_better_by_0.1"] == 0.25 and r["a_better_by_0.1"] == 0.25 and r["same_within_0.1"] == 0.5
    assert r["empty_pred_a"] == 0.25 and r["largest_gains"][0]["part"] == "p"
    with pytest.raises(Exception):
        tc.compare(a, b.iloc[:3])


def test_train_images_cover_every_training_source(monkeypatch):
    tc = load_script("ta_checks")
    wl = pd.DataFrame({"path": ["a", "b", "c"], "role": ["paco_train", "grefcoco_train_nt", "paco_val"]})
    monkeypatch.setattr(tc, "manifest_inputs", lambda cfg, smoke: (None, None, None, wl))
    monkeypatch.setattr(tc.S2T, "node_cache", lambda cfg, source, smoke: (None, [source]))
    monkeypatch.setattr(tc.torch, "load", lambda f, **k: {"records": [{"path": f"{f}.jpg"}]})

    class D:
        def __truediv__(self, x):
            return self

        def glob(self, pat):
            return ["rest"]

    monkeypatch.setattr(tc.TPN, "out_dir", lambda cfg, smoke: D())
    monkeypatch.setattr(tc.TPN, "num_shards", lambda cfg, smoke: 1)
    cfg = {"ta": {"sources": {"paco_train": 0.6, "pascal_part_train": 0.2, "grefcoco_train": 0.2},
                  "paco_rest": True}}
    out = tc.ta_train_images(cfg)
    assert out == {"a": "paco_train", "b": "grefcoco_train_nt", "pascal_part_train.jpg": "pascal_part_train",
                   "grefcoco_train.jpg": "grefcoco_train", "rest.jpg": "paco_train_rest"}   # val image excluded
