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


def test_showcase_keeps_clear_wins_one_per_category_largest_margin_first():
    qa = load_script("qual_assets")
    q = ["k1|part|mug:handle", "k2|part|mug:lid", "k3|part|dog:tail", "k4|part|car:wheel", "k5|part|bird:wing"]
    ours = pd.DataFrame({"qid": q, "iou": [0.95, 0.99, 0.90, 0.80, 0.97]})
    b1 = pd.DataFrame({"qid": q, "iou": [0.10, 0.00, 0.20, 0.00, 0.70]})
    b2 = pd.DataFrame({"qid": q, "iou": [0.30, 0.05, 0.10, 0.00, 0.10]})
    out = qa.showcase(ours, {"box": b1, "sam3": b2}, 5, qa.part_category)
    # mug: lid (margin .94) beats handle (.65); car fails min_ours; bird's best baseline .70 -> margin .27 < .4
    assert [r["qid"] for r in out] == ["k2|part|mug:lid", "k3|part|dog:tail"]
    assert out[0]["group"] == "selected" and out[0]["run_iou"]["sam3"] == 0.05
    assert qa.part_category("instructpart:pic_7-kettle-handle") == "kettle"


def test_top_candidates_follow_the_decision_row_with_softmax_shares():
    qa = load_script("qual_assets")
    masks = [rect(0, 4, 0, 4), rect(4, 8, 4, 8), rect(8, 12, 8, 12)]
    pool = type("P", (), {"rles": M.encode(np.stack(masks))})()
    lg = np.array([[9.0, 0.0, 0.0], [0.0, 2.0, 1.0]])              # chain: the last row decides
    out = qa.top_candidates(pool, lg, 2)
    assert [m.sum() for m, _ in out] == [16, 16] and (out[0][0] == masks[1]).all()
    assert abs(sum(p for _, p in qa.top_candidates(pool, lg, 3)) - 1) < 1e-9 and out[0][1] > out[1][1]
