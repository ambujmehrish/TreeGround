import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treeground import masks as M
from treeground.run import ProtocolError

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BT = module("b_transfer")
MT = module("make_tables")
BS = {"n": 200, "seed": 0}


def rle(y0, y1, x0, x1, h=20, w=20):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return M.encode(m)[0]


def test_queries_share_objects_and_use_d13_phrases():
    car, wheel, door = rle(0, 10, 0, 10), rle(8, 10, 0, 3), rle(2, 8, 2, 8)
    hub = rle(8, 9, 0, 1)
    chains = [("k|chain|car:wheel", "car", "wheel", car, wheel), ("k|chain|car:door", "car", "door", car, door),
              ("k|chain|car:wheel:hub", "wheel of the car", "hub", wheel, hub)]
    q = BT.image_queries("k", chains)
    assert [(r["qid"], r["phrase"]) for r in q] == [
        ("k|object|car", "the car"), ("k|part|car:wheel", "the wheel of the car"),
        ("k|part|car:door", "the door of the car"), ("k|object|wheel of the car", "the wheel of the car"),
        ("k|part|car:wheel:hub", "the hub of the wheel of the car")]
    assert q[1]["object_qid"] == "k|object|car" and q[1]["chain_qid"] == "k|chain|car:wheel"
    assert q[2]["part_ratio"] == pytest.approx(36 / 100)
    with pytest.raises(ProtocolError):
        BT.image_queries("k", chains + [("k|chain|car:roof", "car", "roof", door, hub)])


def test_box_gt_scores_the_predicted_mask_bounding_box_as_x_transfer():
    gt = np.array([0.0, 0.0, 10.0, 10.0])
    pred = rle(0, 5, 5, 15)  # box x 5..15, y 0..5
    s = BT.score(pred, gt)
    assert s["iou"] == pytest.approx(float(BT.XT.gt_ious([gt], [pred], M.boxes_xyxy([pred]))[0, 0]))
    assert s["iou"] == pytest.approx(25 / 125) and s["inter"] / s["union"] == pytest.approx(s["iou"])
    assert s["pred_area"] == 50 and s["gt_area"] == 100
    assert BT.score(None, gt)["iou"] == 0.0
    m = rle(0, 10, 0, 10)
    assert BT.score(pred, m) == BT.B.overlap(pred, m)


def table():
    rows = []
    for img, (oi, ious) in {"a": (0.9, [0.6, 0.1]), "b": (0.2, [0.7])}.items():
        rows.append({"qid": f"{img}|object|car", "kind": "object", "image": img, "object_qid": f"{img}|object|car",
                     "chain_qid": None, "part_ratio": np.nan, "iou": oi})
        for k, v in enumerate(ious):
            rows.append({"qid": f"{img}|part|car:p{k}", "kind": "part", "image": img, "object_qid": f"{img}|object|car",
                         "chain_qid": f"{img}|chain|car:p{k}", "part_ratio": 0.6 if k else 0.1, "iou": v})
    df = pd.DataFrame(rows)
    df["iou_any"], df["inter"], df["union"] = df.iou, df.iou * 10, 10
    df["pred_area"] = 1
    return df


def test_chain_table_and_summary():
    df = table()
    ch = BT.chain_table(df)
    assert list(ch.qid) == ["a|chain|car:p0", "a|chain|car:p1", "b|chain|car:p0"]
    assert list(ch.chain) == [True, False, False] and list(ch.step2) == [True, False, True]
    assert list(ch.single_level) == [False, True, False]
    s = BT.summarize(df, ch, BS)
    assert s["chains"]["all"]["chain_success_end_to_end"]["mean"] == pytest.approx(1 / 3)
    assert s["chains"]["all"]["final_part_success@0.5"]["mean"] == pytest.approx(2 / 3)
    assert s["by_kind"]["part"]["gIoU"]["mean"] == pytest.approx(1.4 / 3)
    assert s["by_kind"]["object"]["n"] == 2 and s["chains"]["single_level"]["chains"] == 1
    with pytest.raises(ProtocolError):
        BT.chain_table(df[df.qid != "b|object|car"])


def test_common_subset_from_x_transfer_chains_csv(tmp_path):
    rows = [{"qid": q, "image": q[0], "structure": s, "feasible": f}
            for q, fs in {"a|chain|car:p0": (1, 1, 1), "a|chain|car:p1": (1, 0, 1), "b|chain|car:p0": (1, 1, 1)}.items()
            for s, f in zip(("flat", "true", "rand0"), fs)]
    pd.DataFrame(rows).assign(feasible=lambda d: d.feasible.astype(bool)).to_csv(tmp_path / "c.csv", index=False)
    common, listed = BT.common_chains(tmp_path / "c.csv")
    assert common == {"a|chain|car:p0", "b|chain|car:p0"} and len(listed) == 3
    sub = BT.restrict(table(), {"a|chain|car:p1"})
    assert sorted(sub.qid) == ["a|object|car", "a|part|car:p1"]


def test_deviations_follow_x_transfer():
    assert BT.deviations("pascal_part") == ["D7", "D13", "D16"]
    assert BT.deviations("openimages_hier") == ["D7", "D13", "D16", "D8", "D9"]


def test_transfer_table_adds_published_baseline_rows(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    ci = {"mean": 0.2, "ci95": [0.1, 0.3], "n": 9}
    blk = {"chains": 9, "chain_success_end_to_end": ci}
    (m / "b_transfer_sa2va_ade20k_merge.json").write_text(json.dumps(
        {"by_kind": {"part": {"gIoU": ci}, "object": {"gIoU": {**ci, "mean": 0.5}}}, "chains": {"all": blk},
         "common_eligible_subset": {"size": 4,
                                    "chains": {"all": {**blk, "chain_success_end_to_end": {**ci, "mean": 0.25}}}}}))
    MT.write(tmp_path / "t", "transfer", *MT.transfer(m))
    with open(tmp_path / "t" / "transfer.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    base = [r for r in rows if r["search"] == "independent"]
    assert len(base) == 1 and base[0]["selector"] == "Sa2VA" and base[0]["dataset"] == "ADE20K"
    assert (base[0]["chains"], base[0]["common"], base[0]["independent"], base[0]["end_to_end"],
            base[0]["object_gIoU"], base[0]["flat"]) == ("9", "4", "0.250", "0.200", "0.500", "–")
