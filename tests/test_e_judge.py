import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e_judge", ROOT / "scripts" / "e_judge.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

S = {"variants": ["full", "inside"], "target_area": 512 * 512, "batch_size": 3,
     "crop": {"pad_frac": 0.25, "min_side": 64}}


def test_prompts_match_viescore_tie_assembly():
    # sha256 of the prompts VIEScore(task="tie") assembles from vie_prompts.py at Step1X-Edit@4cca05df
    assert hashlib.sha256(J.SC_PROMPT.encode()).hexdigest() == \
        "8f529562c2f89f8db51a3f7cc4d487686f676909c21010b67b0e65ec78150de5"
    assert hashlib.sha256(J.PQ_PROMPT.encode()).hexdigest() == \
        "2473ff2c7406e06545c5c039ceb039ff2604ebacf150868a48f0bf07c9e733b0"
    sc = J.prompt("sc", "make the handle of the mug red")
    assert "<instruction>" not in sc and sc.endswith("Editing instruction: make the handle of the mug red\n")
    assert J.prompt("pq", "ignored") == J.PQ_PROMPT
    m = J.messages(2, sc)[0]
    assert m["role"] == "user" and [c["type"] for c in m["content"]] == ["image", "image", "text"]


@pytest.mark.parametrize("text, expected", [
    ('{\n"score" : [7, 9],\n"reasoning" : "fine"\n}', [7, 9]),
    ('```json\n{"score": [3, 10], "reasoning": "the handle is red"}\n```', [3, 10]),
    ('||V^=^V||\n{"score": [5, 5], "reasoning": "x"}\n||V^=^V||', [5, 5]),
    ('{score: [6, 8], reasoning: unquoted}', [6, 8]),
    ('{"score": [6, 8], "reasoning": "unterminated}', [6, 8]),
    ('{"score" : 4, "reasoning": "single"}', [4]),
    ('"score": 7.5 because', [7.5]),
    ("[2, 4]", [2, 4]),
    ("8", [8]),
    ('{"score": [11, 3]}', None),
    ('{"score": [-1, 3]}', None),
    ('{"score": [], "reasoning": "none"}', None),
    ('{"score": ["high", 3]}', None),
    ("I cannot evaluate this image.", None),
    ("", None),
])
def test_parse_scores(text, expected):
    assert J.parse_scores(text) == expected


def test_gedit_dims():
    assert J.gedit_dims(512, 512, 512 * 512) == (512, 512)
    assert J.gedit_dims(1024, 768, 512 * 512) == (591, 443)
    w, h = J.gedit_dims(640, 427, 512 * 512)
    assert abs(w * h - 512 * 512) < w + h and abs(w / h - 640 / 427) < 0.01


def test_crop_box_pads_clips_and_enforces_min_side():
    m = np.zeros((400, 300), bool)
    m[100:200, 100:140] = True
    assert J.crop_box(m, 0.25, 64) == (88, 75, 152, 225)   # x: 40 + 2*10 = 60 < 64 -> 64 centred
    m = np.zeros((400, 300), bool)
    m[0:100, 250:300] = True                                 # touches the top and right borders
    x0, y0, x1, y1 = J.crop_box(m, 0.25, 64)
    assert (y0, y1) == (0, 150) and x1 == 300 and x1 - x0 == 74
    m = np.zeros((40, 30), bool)
    m[10:12, 10:12] = True
    assert J.crop_box(m, 0.25, 64) == (0, 0, 30, 40)         # min side larger than the image
    with pytest.raises(J.ProtocolError):
        J.crop_box(np.zeros((5, 5), bool), 0.25, 64)


class FakeJudge:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, reqs):
        self.calls.append(reqs)
        return [self.replies[(r["condition"], r["variant"], r["kind"])] for r in reqs]


def test_judge_sample_rows_batches_and_views():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    gt = np.zeros((300, 400), bool)
    gt[100:180, 150:250] = True
    outs = {"free": image.copy(), "gt_blend": image.copy()}
    ok = '{"score": [8, 6], "reasoning": "ok"}'
    replies = {(c, v, k): ok for c in outs for v in S["variants"] for k in J.KINDS}
    replies[("gt_blend", "inside", "pq")] = "no idea"
    replies[("free", "full", "pq")] = '{"score": [9, 4]}'
    fake = FakeJudge(replies)
    rows = {(r["condition"], r["variant"]): r for r in J.judge_sample(fake, "make it red", image, gt, outs, S)}
    assert [len(c) for c in fake.calls] == [3, 3, 2]
    reqs = [r for c in fake.calls for r in c]
    full = next(r for r in reqs if r["variant"] == "full" and r["kind"] == "sc")
    inside = next(r for r in reqs if r["variant"] == "inside" and r["kind"] == "sc")
    assert len(full["images"]) == 2 and full["images"][0].size == J.gedit_dims(400, 300, 512 * 512)
    assert inside["box"] == [125, 80, 275, 200] and inside["images"][1].size == J.gedit_dims(150, 120, 512 * 512)
    assert all(len(r["images"]) == 1 for r in reqs if r["kind"] == "pq")
    r = rows[("free", "full")]
    assert (r["sc"], r["pq"], r["overall"], r["parse_failed"]) == (6, 4, math.sqrt(24), False)
    assert json.loads(r["sc_scores"]) == [8, 6] and r["crop_box"] == "null"
    bad = rows[("gt_blend", "inside")]
    assert bad["parse_failed"] and math.isnan(bad["pq"]) and math.isnan(bad["overall"]) and bad["sc"] == 6
    assert json.loads(bad["crop_box"]) == [125, 80, 275, 200]


def test_judge_sample_rejects_resized_output():
    image = np.zeros((20, 30, 3), np.uint8)
    gt = np.ones((20, 30), bool)
    with pytest.raises(J.ProtocolError):
        J.judge_sample(FakeJudge({}), "x", image, gt, {"free": np.zeros((10, 30, 3), np.uint8)}, S)


def test_summarize_means_failures_and_subsets():
    rows = []
    for cond, base in (("free", 2.0), ("ours_blend", 6.0)):
        for i in range(6):
            sc, pq = base + i % 2, 9.0
            failed = cond == "ours_blend" and i == 0
            rows.append({"sample_id": f"s{i}", "image": f"img{i // 2}", "subset": "a" if i < 3 else "b",
                         "condition": cond, "variant": "full", "sc": sc, "pq": float("nan") if failed else pq,
                         "overall": float("nan") if failed else math.sqrt(sc * pq), "parse_failed": failed})
    df = pd.DataFrame(rows)
    out, table = J.summarize(df, ["free", "ours_blend"], ["full"], {"n": 200, "seed": 0})
    f, o = out["free"]["full"]["all"], out["ours_blend"]["full"]["all"]
    assert f["n"] == 6 and f["parse_failed_rate"] == 0 and f["sc"]["mean"] == pytest.approx(2.5)
    assert f["sc"]["ci95"][0] <= 2.5 <= f["sc"]["ci95"][1]
    assert o["parse_failed_rate"] == pytest.approx(1 / 6) and o["pq"]["n"] == 5 and o["sc"]["n"] == 6
    exp = np.mean([math.sqrt((6 + i % 2) * 9) for i in range(1, 6)])
    assert o["overall"]["mean"] == pytest.approx(exp)
    assert set(out["free"]["full"]["by_subset"]) == {"a", "b"}
    assert list(table.condition) == ["free", "ours_blend"] and table.overall.iloc[1] == pytest.approx(exp)
