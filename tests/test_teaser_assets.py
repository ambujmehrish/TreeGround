import json
import sys

import numpy as np
import pandas as pd

from test_s2 import load_script
from treeground import masks as M

H, W = 32, 40


def mask(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


class FakePool:
    def __init__(self, rles):
        self.rles, self.n_grid, self.parents = rles, len(rles), np.full(len(rles), -1)

    @property
    def n(self):
        return len(self.rles)


class FakeScorer:
    # candidates: two people (0, 1) and the background (2); logits make 0 and 1 pass t = 0.9, 2 not
    def __init__(self, logits):
        self.lg = np.asarray(logits, float)
        self.rles = M.encode(np.stack([mask(0, 16, 0, 10), mask(0, 16, 20, 30), mask(20, 32, 0, 40)]))

    def score(self, image, key, texts, queries):
        assert len(texts) == len(queries) == 1 and texts[0].endswith("<SEL>")
        return [FakePool(self.rles)], [self.lg[None]]


def run(monkeypatch, logits, ann_ids, no_target):
    ta = load_script("teaser_assets")
    ge = sys.modules.get("gres_eval") or load_script("gres_eval")
    sys.modules["gres_eval"] = ge
    gt = M._coco(M.encode(mask(0, 16, 0, 10) | mask(0, 16, 20, 30))[0])
    df = pd.DataFrame([{"qid": "grefcoco|testA|1", "image": "grefcoco:1", "file": "x.jpg", "h": H, "w": W,
                        "text": "two people", "no_target": no_target, "ann_ids": json.dumps(ann_ids)}])
    monkeypatch.setattr(ge, "expressions", lambda cfg, split: (df, {}))
    monkeypatch.setattr(ge, "gt_rle", lambda anns, ids, h, w: gt if ids else None)
    monkeypatch.setattr(ta, "load_rgb", lambda p: np.zeros((H, W, 3), np.uint8))
    return ta.gres_examples({"data": {"coco_images": "/x"}}, FakeScorer(logits), 5, 0.9, np.random.default_rng(0))


def test_multi_target_set_rule_and_iou(monkeypatch):
    out = run(monkeypatch, [4.0, 3.0, 1.0], [5, 6], False)
    r = out["multi"][0]
    assert r["chosen"] == [0, 1] and r["iou"] == 1.0 and r["n_targets"] == 2
    assert len(r["pool"]) == len(r["logits"]) == 3 and r["n_grid"] == 3
    json.dumps(out)


def test_no_target_correct_only_when_set_is_empty(monkeypatch):
    out = run(monkeypatch, [1.0, 0.5, -2.0], [], True)
    r = out["no_target"][0]
    assert r["chosen"] == [] and r["iou"] == 1.0 and r["gt"] is None
    assert abs(r["max_sigmoid"] - 1 / (1 + np.exp(-1.0))) < 1e-9
    out = run(monkeypatch, [4.0, 0.5, -2.0], [], True)
    assert out["no_target"][0]["chosen"] == [0] and out["no_target"][0]["iou"] == 0.0
