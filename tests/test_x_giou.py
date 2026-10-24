import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from treeground import tree as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("x_giou", ROOT / "scripts" / "x_giou.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
spec4 = importlib.util.spec_from_file_location("v4_tree_value", ROOT / "scripts" / "v4_tree_value.py")
V4 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(V4)


def item(sig, ious, parents):
    return {"qid": "q", "sig": np.asarray(sig, float), "ious": np.asarray(ious, float),
            "tree": T.from_parents(np.asarray(parents)), "part_ratio": 0.2}


def test_picks_flat_tree_and_rejection():
    # node 0 = object, nodes 1, 2 its children, node 3 elsewhere
    it = item([[0.9, 0.1, 0.1, 0.2], [0.1, 0.3, 0.6, 0.8]], [[0.8, 0.1, 0.1, 0.0], [0.0, 0.2, 0.7, 0.1]],
              [-1, 0, 0, -1])
    p = G.picks(it, 0.85, V4.decide)
    assert p["object_norej"] == 0.8 and p["part_flat_norej"] == 0.1 and p["part_tree_norej"] == 0.7
    assert p["object_calib"] == 0.8 and p["part_flat_calib"] == 0.0 and p["part_tree_calib"] == 0.0


def test_empty_node_set_scores_zero():
    it = {"qid": "q", "sig": np.zeros((2, 0)), "ious": np.zeros((2, 0)), "tree": T.from_parents(np.array([-1])),
          "part_ratio": 0.1}
    assert all(v == 0.0 for v in G.picks(it, 0.5, V4.decide).values())


def test_summary_dedups_objects_per_image():
    df = pd.DataFrame({"image": ["a", "a", "b"], "object": ["cup", "cup", "cup"], "single_level": [False] * 3,
                       **{f"{k}_{m}": [1.0, 1.0, 0.0] for k in ("object", "part_flat", "part_tree")
                          for m in ("norej", "calib")}})
    s = G.summarize(df, {"n": 50, "seed": 0})
    assert s["chains"] == 3 and s["objects"] == 2
    assert np.isclose(s["norej"]["object_gIoU"]["mean"], 0.5)
    assert np.isclose(s["norej"]["part_gIoU_tree"]["mean"], 2 / 3)


def test_label_and_approvals():
    cfg = {"inputs": {"selector_stage": "s1"}, "s2_eval": {}}
    assert G.label(cfg) == "s2_from_s1" and "D17" in G.approvals(cfg, "openimages_hier")
    assert G.label({"inputs": {"selector_stage": "v2"}}) == "v2"
