import json

import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script
from treeground import masks as M
from treeground import tree as T
from treeground.run import ProtocolError

H, W = 24, 32


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_choose_draws_disjoint_groups_random_first():
    qa = load_script("qual_assets")
    ours = pd.DataFrame({"qid": list("abcdef"), "iou": [0.9, 0.1, 0.5, 0.8, 0.2, 0.6]})
    base = pd.DataFrame({"qid": list("abcdef"), "iou": [0.1, 0.9, 0.5, 0.7, 0.2, 0.0]})
    out = qa.choose(ours, base, 2, np.random.default_rng(0))
    groups = {g: [r["qid"] for r in out if r["group"] == g] for g in qa.GROUPS}
    assert all(len(v) == 2 for v in groups.values()) and len({r["qid"] for r in out}) == 6
    margin = dict(zip(ours.qid, ours.iou - base.iou))
    rest = sorted(set("abcdef") - set(groups["random"]), key=lambda q: -margin[q])
    assert groups["ours_better"] == rest[:2] and groups["ours_worse"] == sorted(rest[2:], key=lambda q: margin[q])
    assert out[0]["run_iou"]["ours"] == float(ours.set_index("qid").iou[out[0]["qid"]])


def test_choose_refuses_runs_over_different_queries():
    qa = load_script("qual_assets")
    with pytest.raises(ProtocolError):
        qa.choose(pd.DataFrame({"qid": ["a", "b"], "iou": [1.0, 0.0]}), pd.DataFrame({"qid": ["a"], "iou": [0.5]}),
                  1, np.random.default_rng(0))


def test_small_mask_keeps_the_region_at_the_image_size_and_drops_empty_masks():
    qa = load_script("qual_assets")
    big = np.zeros((1000, 2000, 3), np.uint8)
    hw = qa.small(big).shape[:2]
    assert max(hw) == qa.SIDE and hw == (256, 512)
    m = np.zeros((1000, 2000), bool)
    m[:500, :1000] = True
    r = qa.small_mask(m, hw)
    dec = M.decode({"size": r["size"], "counts": r["counts"].encode()})
    assert dec.shape == hw and dec[:128, :256].all() and not dec[128:, 256:].any()
    assert qa.small_mask(np.zeros((10, 10), bool), (5, 5)) is None and qa.small_mask(None, (5, 5)) is None


@pytest.mark.parametrize("anchor", [None, "soft2.0", "hard"])
def test_pick_index_is_the_pick_ta_test_eval_scores(anchor):
    # the figure must show the mask the table scored: compare with ta_test_eval.single_rows' sel_iou
    qa, tte = load_script("qual_assets"), load_script("ta_test_eval")
    masks = [rect(0, 24, 0, 32), rect(0, 12, 0, 16), rect(0, 6, 0, 8), rect(12, 24, 16, 32), rect(18, 24, 24, 32)]
    rles = M.encode(np.stack(masks))
    parents = T.build_tree(rles, "cpu").parent
    gt = rect(12, 24, 16, 32)
    rng = np.random.default_rng(1)
    for _ in range(20):
        lg = rng.normal(size=(2, len(masks)))
        g = pd.DataFrame([{"qid": "k|part|mug:handle", "kind": "part", "image_key": "k", "gt": json.dumps([0])}])
        row = tte.single_rows(g, [tte.Pool(rles, parents, 3)], [lg], {0: M.encode(gt[None])[0]}, anchor)[0]
        a = qa.pick_index("part", lg, parents, anchor)
        assert abs(qa.iou(masks[a], gt) - row["sel_iou"]) < 1e-9
