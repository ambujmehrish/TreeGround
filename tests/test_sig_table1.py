import numpy as np
import pandas as pd
import pytest

from test_s2 import load_script
from treeground.run import ProtocolError


def u(uids, images, values):
    return pd.DataFrame({"uid": uids, "image": images, "value": np.asarray(values, float)})


def test_paired_bootstrap_identical_methods_give_zero_delta_and_p_one():
    st = load_script("sig_table1")
    a = np.linspace(0, 1, 40)
    r = st.paired_bootstrap(a, a.copy(), np.repeat(np.arange(10), 4), 500, 0)
    assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0] and r["p_two_sided"] == 1.0 and r["images"] == 10


def test_paired_bootstrap_a_consistent_gain_is_significant():
    st = load_script("sig_table1")
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 0.8, 300)
    a = b + 0.1 + rng.normal(0, 0.02, 300)
    r = st.paired_bootstrap(a, b, np.repeat(np.arange(100), 3), 2000, 0)
    assert 0.09 < r["delta"] < 0.11 and r["ci95"][0] > 0 and r["p_two_sided"] == 1 / 2000


def test_bootstrap_resamples_images_not_queries():
    # one image carries the whole gain: an image-level bootstrap must not call that certain
    st = load_script("sig_table1")
    a = np.r_[np.ones(50), np.zeros(5)]
    b = np.zeros(55)
    r = st.paired_bootstrap(a, b, np.r_[np.zeros(50), np.arange(1, 6)], 2000, 0)
    assert r["ci95"][0] == 0.0 and r["p_two_sided"] > 0.05


def test_compare_uses_the_best_baseline_on_the_same_queries():
    st = load_script("sig_table1")
    per = {"ours": (u(["q1", "q2", "q3"], ["i1", "i1", "i2"], [0.9, 0.5, 0.7]), 0.7),
           "sam3_text": (u(["q3", "q1", "q2"], ["i2", "i1", "i1"], [0.1, 0.2, 0.3]), 0.2),
           "qwen_box_sam": (u(["q2", "q3", "q1"], ["i1", "i2", "i1"], [0.5, 0.6, 0.8]), 0.6333333333333333)}
    r = st.compare(per, 200, 0)
    assert r["best_baseline"] == "qwen_box_sam" and abs(r["delta"] - (2.1 - 1.9) / 3) < 1e-12


def test_compare_refuses_a_reported_number_it_cannot_rebuild_or_other_queries():
    st = load_script("sig_table1")
    ok = u(["q1", "q2"], ["i1", "i2"], [0.4, 0.6])
    with pytest.raises(ProtocolError, match="differs from the reported"):
        st.compare({"ours": (ok, 0.55), "sa2va": (ok, 0.5)}, 10, 0)
    with pytest.raises(ProtocolError, match="different queries"):
        st.compare({"ours": (ok, 0.5), "sa2va": (u(["q1", "q9"], ["i1", "i2"], [0.4, 0.6]), 0.5)}, 10, 0)
