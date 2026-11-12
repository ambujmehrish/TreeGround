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
    assert qid == "instructpart:kettle_001-kettle-handle" and key == "instructpart:kettle_001"   # key = the photo
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
    b.pred, b.clip, b.abstain, b.prompts = pred, re.Baseline._clipper(tok, lambda p: p), True, None
    assert b(None, "k", "one two three four five six seven").all()
    assert seen[0] == ["one two three four"]                 # 4 words + 2 = 6 tokens
    assert b(None, "k", "short") is None
    b.abstain = False                                        # sam3_text_na: below-threshold top mask is kept
    assert b(None, "k", "short").all()


def test_no_abstain_method_uses_the_same_predictor(monkeypatch):
    re = load_script("reason_eval")
    import b_baselines as B
    monkeypatch.setitem(B.PREDICTORS, "sam3_text", lambda cfg: SimpleNamespace())
    b = re.Baseline({}, "sam3_text_na")
    assert b.abstain is False and re.Baseline({}, "sam3_text").abstain is True


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
    o.TTE, o.s, o.max_side, o.rw = TTE, S(), 10 ** 6, None
    assert o(np.zeros((H, W, 3), np.uint8), "k", "Add some water.").all()


def test_reasonseg_images_are_upright_by_exif(tmp_path):
    re = load_script("reason_eval")
    d = tmp_path / "reasonseg" / "data"
    d.mkdir(parents=True)
    img = Image.fromarray(np.zeros((W, H, 3), np.uint8))           # stored sideways (W x H)
    exif = img.getexif()
    exif[0x0112] = 6                                                # rotate 90 CW on display -> H x W
    b = io.BytesIO()
    img.save(b, format="JPEG", exif=exif)
    mask = np.zeros((H, W), bool)
    pd.DataFrame([{"image": {"bytes": b.getvalue(), "path": None}, "text": "q", "mask": [list(r) for r in mask],
                   "image_id": "i", "ann_id": "a", "img_height": H, "img_width": W}]).to_parquet(d / "test-00000-of-00001.parquet")
    (rec,) = re.records({"data": {"reasonseg": str(tmp_path / "reasonseg")}}, "reasonseg")
    assert rec[3]().shape[:2] == rec[4]().shape == (H, W)


def test_clip_cuts_a_single_overlong_word_at_the_token_level():
    re = load_script("reason_eval")

    class Tok:
        model_max_length = 5

        def __call__(self, s, truncation=False, max_length=None):
            ids = list(range(len(s)))                               # one token per character
            return SimpleNamespace(input_ids=ids[:max_length] if truncation else ids)

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)
    clip = re.Baseline._clipper(Tok(), lambda p: p)
    assert clip("abcdefghij") == "xxxxx" and clip("ab cd ef") == "ab cd"


def test_empty_rate_reads_the_column_not_the_dataframe_property():
    df = pd.DataFrame({"empty": [True, False, False, False]})
    assert float(df["empty"].mean()) == 0.25 and df.empty is False


def test_ours_runs_large_images_downscaled_and_returns_a_full_size_mask():
    re = load_script("reason_eval")
    o = object.__new__(re.Ours)
    o.max_side, seen = 8, []

    def select(image, key, text):
        seen.append(image.shape)
        m = np.zeros(image.shape[:2], bool)
        m[:, : image.shape[1] // 2] = True
        return m
    o.select = select
    out = o(np.zeros((H, W, 3), np.uint8), "k", "q")                # 12 x 16 -> 6 x 8
    assert seen[-1] == (6, 8, 3) and out.shape == (H, W) and out[:, :8].all() and not out[:, 8:].any()
    o.max_side = 64
    assert o(np.zeros((H, W, 3), np.uint8), "k", "q").shape == (H, W) and seen[-1] == (H, W, 3)


def test_sentences_get_the_sentence_prompt_and_phrases_keep_the_phrase_prompt(monkeypatch):
    re = load_script("reason_eval")
    import b_baselines as B
    used = []

    class P:
        s = {"prompt": "Please segment {phrase} in this image."}

        def __init__(self, cfg):
            pass

        def __call__(self, image, phrases):
            used.append(self.s["prompt"].format(phrase=phrases[0]))
            return [(np.ones((H, W), bool), float("nan"), "", True)]
    monkeypatch.setitem(B.PREDICTORS, "sa2va", P)
    b = re.Baseline({"reason_prompts": {"sa2va": "{phrase} Please respond with a segmentation mask."}}, "sa2va")
    b(None, "k", "What can hold water?"), b(None, "k", "Lift it to pour."), b(None, "k", "the kettle handle")
    assert used == ["What can hold water? Please respond with a segmentation mask.",
                    "Lift it to pour. Please respond with a segmentation mask.",
                    "Please segment the kettle handle in this image."]


def test_ours_text_clip_matches_the_word_rule_and_cuts_a_single_long_word():
    re = load_script("reason_eval")
    s3p = load_script("s3_pool")

    class Tok:
        model_max_length = 5

        def __call__(self, s, truncation=False, max_length=None):
            ids = list(range(len(s.replace(" ", ""))))              # one token per non-space character
            return SimpleNamespace(input_ids=ids[:max_length] if truncation else ids)

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)
    ti = object.__new__(s3p.TextInstances)
    ti.tok, ti.limit = Tok(), 5
    clip = re.safe_text_clip(Tok())
    for p in ("the ab cd ef", "ab c", "the mug"):
        assert clip(p) == ti.clip(p)                                  # unchanged where the word rule works
    assert clip("abcdefghij") == "xxxxx" and len(ti.clip("abcdefghij")) == 10   # old rule passes 10 tokens on


def test_d51_rewrite_reads_the_question_and_pools_the_named_phrase(monkeypatch):
    re = load_script("reason_eval")
    rd = load_script("reason_diag")
    import json
    from pathlib import Path
    val = Path(__file__).resolve().parents[1] / "results" / "metrics" / "reason_diag_val.json"
    assert re.REWRITE_PROMPT == rd.REWRITE_PROMPT == json.loads(val.read_text())["rewrite_prompt"]   # frozen on val
    o = object.__new__(re.Ours)
    import ta_test_eval as TTE
    from treeground import masks as M
    seen = {}
    named = np.zeros((H, W), bool)
    named[2:6, 3:9] = True

    class S:
        pool, ps, device = True, {"instances_per_phrase": 3, "max_instances": 160, "dup_iou": 0.95}, "cpu"
        import s3_pool as S3P

        def nodes(self, image):
            return M.encode(np.ones((1, H, W), bool)), None

        def text(self, image, phrases, k):
            seen["phrases"] = phrases
            return [[(named, 0.9)] for _ in phrases]

        def logits(self, image, key, pool, queries):
            seen["query"] = queries[0].user
            return [np.arange(pool.n, dtype=float)[None]]
    o.TTE, o.s, o.max_side = TTE, S(), 10 ** 6
    o.rw = lambda image, text: ("the kettle handle", "The kettle handle.")
    out = o(np.zeros((H, W, 3), np.uint8), "k", "What do I hold to pour?")
    assert seen["phrases"] == ["the kettle handle"] and "What do I hold to pour?" in str(seen["query"])
    assert (out == named).all()
    o.rw = lambda image, text: ("", "")
    o(np.zeros((H, W, 3), np.uint8), "k", "What do I hold to pour?")
    assert seen["phrases"] == ["What do I hold to pour?"]          # empty answer: the s3_pool phrase


def test_d51_method_has_its_own_stage_identity_and_approval():
    re = load_script("reason_eval")
    assert re.is_ours("ours_rw") and not re.is_ours("qwen_box_sam")
