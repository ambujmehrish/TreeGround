import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script
from treeground.run import ProtocolError


def frame(qids, inter, union):
    sr = load_script("sig_reason")
    df = pd.DataFrame({"qid": qids, "inter": np.asarray(inter, float), "union": np.asarray(union, float)})
    df["image"] = df.qid.map(sr.photo)
    return df.set_index("qid").assign(iou=np.where(df.union > 0, df.inter / df.union.clip(lower=1), 0.0).tolist())


def test_photo_clusters_parts_of_one_photo_and_reasonseg_annotations_of_one_image():
    sr = load_script("sig_reason")
    assert sr.photo("instructpart:kettle_001-kettle-handle") == sr.photo("instructpart:kettle_001-kettle-lid") \
        == "instructpart:kettle_001"
    assert sr.photo("instructpart:a-b-c-wine_glass-stem") == "instructpart:a-b-c"
    assert sr.photo("reasonseg:img7:3") == sr.photo("reasonseg:img7:4") == "reasonseg:img7"


def test_ratio_bootstrap_delta_is_the_cIoU_difference():
    sr = load_script("sig_reason")
    q = [f"reasonseg:i{k}:0" for k in range(30)]
    a = frame(q, [8] * 30, [10] * 30)
    b = frame(q, [5] * 30, [10] * 30)
    r = sr.paired_ratio_bootstrap(a, b, 500, 0)
    assert abs(r["delta"] - 0.3) < 1e-12 and r["ci95"][0] > 0.29 and r["p_two_sided"] == 1 / 500


def test_compare_aligns_queries_and_refuses_different_sets():
    sr = load_script("sig_reason")
    q = [f"reasonseg:i{k}:0" for k in range(20)]
    a = frame(q, np.arange(20), [20] * 20)
    b = frame(q[::-1], np.arange(20)[::-1], [20] * 20)            # same values, other order
    r = sr.compare(a, b, 200, 0)
    assert r["gIoU"]["delta"] == 0 and r["cIoU"]["delta"] == 0
    with pytest.raises(ProtocolError):
        sr.compare(a, frame(q[:-1] + ["reasonseg:zz:0"], np.arange(20), [20] * 20), 10, 0)
