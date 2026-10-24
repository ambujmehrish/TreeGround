import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("e2e", Path(__file__).resolve().parents[1] / "scripts" / "e2e_variants.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
BS = {"n": 50, "seed": 0}


def frame():
    return pd.DataFrame({
        "image": ["a", "a", "b", "c", "d"], "kind": ["object", "part", "part", "absent", "absent"],
        "present": [True, True, True, False, False], "max_sigma": [0.9, 0.3, 0.6, 0.95, 0.2],
        "sel_iou": [0.8, 0.6, 0.0, 0.0, 0.0], "sel_inter": [80, 30, 0, 0, 0], "sel_union": [100, 50, 0, 0, 0],
        "gt_area": [90, 40, 20, 0, 0]})


def test_calibrated_vs_no_rejection():
    out, curve = E.variants(frame(), 0.5, BS)
    assert np.isclose(out["calibrated"]["gIoU"]["mean"], 0.8 / 3)
    assert np.isclose(out["no_rejection_present"]["gIoU"]["mean"], 1.4 / 3)
    assert np.isclose(out["calibrated"]["no_target_accuracy"], 0.5)
    assert np.isclose(out["calibrated"]["present_accept_rate"], 2 / 3)
    assert np.isclose(out["no_rejection_present"]["cIoU"], 110 / 170)
    assert np.isclose(out["calibrated"]["cIoU"], 80 / 160)
    assert set(out["no_rejection_present"]["by_kind"]) == {"object", "part"}
    assert len(curve) == len(E.GRID) and curve.present_gIoU.iloc[0] >= curve.present_gIoU.iloc[-1]


def test_cious_count_wrong_answered_predictions():
    df = pd.DataFrame({"image": ["a", "b"], "kind": ["part", "part"], "present": [True, True],
                       "max_sigma": [0.9, 0.9], "sel_iou": [0.5, 0.0], "sel_inter": [50, 0], "sel_union": [100, 70],
                       "gt_area": [60, 20]})
    out, _ = E.variants(df, 0.5, BS)
    assert np.isclose(out["no_rejection_present"]["cIoU"], 50 / 170)
    assert np.isclose(out["calibrated"]["cIoU"], 50 / 170)
