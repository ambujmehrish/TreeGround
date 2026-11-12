import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script
from treeground.run import ProtocolError


def meta():
    return pd.DataFrame({"qid": ["a", "b", "c", "d"], "gt_area": [10, 400, 600, 50], "image_area": [1000] * 4,
                         "sentence": [True, True, False, True]})


def test_size_bins_and_miss_kinds():
    rb = load_script("reason_breakdown")
    assert list(rb.size_bin(np.array([0.01, 0.05, 0.3, 0.9]))) == ["0%-5%", "5%-20%", "20%-50%", "50%-100%"]
    #                 inside        covers        elsewhere     empty
    kinds = rb.miss_kind(np.array([9, 50, 1, 0]), np.array([10, 200, 40, 0]), np.array([100, 60, 30, 20]))
    assert list(kinds) == ["inside", "covers", "elsewhere", "empty"]


def test_query_table_recovers_prediction_area_and_checks_rows():
    rb = load_script("reason_breakdown")
    rows = pd.DataFrame({"qid": ["a", "b", "c", "d"], "inter": [10, 100, 0, 5], "union": [20, 400, 600, 50]})
    t = rb.query_table(rows, meta())
    assert list(t.pred_area) == [20, 100, 0, 5] and abs(t.iou.iloc[1] - 0.25) < 1e-12
    with pytest.raises(ProtocolError):
        rb.query_table(rows.iloc[:3], meta())
    with pytest.raises(ProtocolError):
        rb.query_table(rows.assign(inter=[11, 100, 0, 5]), meta())


def test_summarize_reports_every_method_on_the_same_bins():
    rb = load_script("reason_breakdown")
    m = meta()
    ours = rb.query_table(pd.DataFrame({"qid": list("abcd"), "inter": [10, 100, 0, 5], "union": [10, 400, 600, 50]}), m)
    base = rb.query_table(pd.DataFrame({"qid": list("abcd"), "inter": [0, 400, 600, 0], "union": [20, 400, 600, 60]}), m)
    s = rb.summarize({"ours": ours, "qwen_box_sam": base})
    assert s["by_size"]["0%-5%"] == {"n": 1, "ours": 1.0, "qwen_box_sam": 0.0}
    assert s["by_size"]["50%-100%"]["qwen_box_sam"] == 1.0 and s["by_form"]["phrase"]["n"] == 1
    assert s["misses"]["ours"]["n"] == 3 and s["misses"]["ours"]["inside"] == pytest.approx(2 / 3)
