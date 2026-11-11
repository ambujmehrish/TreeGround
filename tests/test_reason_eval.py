import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from test_s2 import load_script

H, W = 12, 16


def png_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def make_instructpart(root):
    d = root / "instructpart" / "test"
    (d / "images").mkdir(parents=True)
    (d / "masks").mkdir()
    data = [{"image_path": "kettle_001.jpg", "part_list": [
        {"object": "kettle", "part": "handle", "instruction": ["Add some water.", "Lift it to pour.", "x", "y", "z", "w"]},
        {"object": "kettle", "part": "lid", "instruction": ["Open it.", "Remove the top.", "x", "y", "z", "w"]}]}]
    (d / "data_test.json").write_text(json.dumps(data))
    for part, (r0, r1) in (("handle", (2, 6)), ("lid", (0, 3))):
        Image.fromarray(np.full((H, W, 3), 50, np.uint8)).save(d / "images" / f"kettle_001-kettle-{part}.jpg")
        m = np.zeros((H, W), np.uint8)
        m[r0:r1, 3:9] = 255
        Image.fromarray(m).save(d / "masks" / f"kettle_001-kettle-{part}.png")


def make_reasonseg(root):
    d = root / "reasonseg" / "data"
    d.mkdir(parents=True)
    mask = np.zeros((H, W), bool)
    mask[4:8, 4:12] = True
    rows = [{"image": {"bytes": png_bytes(np.full((H, W, 3), 9, np.uint8)), "path": None},
             "text": f"what can hold water {i}?", "mask": [list(r) for r in mask], "image_id": f"img{i}",
             "ann_id": f"a{i}", "img_height": H, "img_width": W} for i in range(3)]
    pd.DataFrame(rows[:2]).to_parquet(d / "test-00000-of-00002.parquet")
    pd.DataFrame(rows[2:]).to_parquet(d / "test-00001-of-00002.parquet")
    return mask


def test_instructpart_records_pick_the_instruction_and_its_mask(tmp_path):
    re = load_script("reason_eval")
    make_instructpart(tmp_path)
    cfg = {"data": {"instructpart": str(tmp_path / "instructpart")}}
    human, gpt = re.records(cfg, "instructpart_human"), re.records(cfg, "instructpart_gpt")
    assert [r[2] for r in human] == ["Add some water.", "Open it."]
    assert [r[2] for r in gpt] == ["Lift it to pour.", "Remove the top."]
    qid, key, _, load_image, load_gt = human[0]
    assert qid == key == "instructpart:kettle_001-kettle-handle"
    assert load_image().shape == (H, W, 3) and load_gt().sum() == 4 * 6 and load_gt().dtype == bool


def test_reasonseg_records_read_text_image_and_mask_across_files(tmp_path):
    re = load_script("reason_eval")
    mask = make_reasonseg(tmp_path)
    recs = re.records({"data": {"reasonseg": str(tmp_path / "reasonseg")}}, "reasonseg")
    assert [r[0] for r in recs] == ["reasonseg:img0:a0", "reasonseg:img1:a1", "reasonseg:img2:a2"]
    assert recs[2][2] == "what can hold water 2?"
    assert recs[2][3]().shape == (H, W, 3) and (recs[2][4]() == mask).all()


def test_baseline_clips_sam3_text_and_returns_none_on_abstention():
    re = load_script("reason_eval")
    b = object.__new__(re.Baseline)
    tok = lambda s: SimpleNamespace(input_ids=[0] * (len(s.split()) + 2))
    tok.model_max_length = 6
    seen = []
    m = np.ones((H, W), bool)

    def pred(image, phrases):
        seen.append(phrases)
        return [(m, 0.9, "", True)] if len(seen) == 1 else [(m, 0.2, "", False)]
    b.pred, b.clip = pred, re.Baseline._clipper(tok, lambda p: p)
    assert b(None, "k", "one two three four five six seven").all()
    assert seen[0] == ["one two three four"]                 # 4 words + 2 = 6 tokens
    assert b(None, "k", "short") is None


def test_ours_answers_with_the_top_candidate():
    re = load_script("reason_eval")
    o = object.__new__(re.Ours)
    import ta_test_eval as TTE
    from treeground import masks as M
    rles = M.encode(np.stack([np.eye(H, W, dtype=bool), np.ones((H, W), bool)]))

    class S:
        def score(self, image, key, texts, queries):
            assert texts == ["Add some water. <SEL>"] and queries[0].kind == "object"
            return [SimpleNamespace(n=2, rles=rles)], [np.array([[0.1, 2.0]])]
    o.TTE, o.s = TTE, S()
    assert o(None, "k", "Add some water.").all()
