from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_s2 import load_script
from treeground import masks as M

H, W = 24, 32


def rect(y0, y1, x0, x1, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_noun_phrase_takes_the_first_line_without_decoration():
    rd = load_script("reason_eval")   # D51's parser lives in the evaluator; the val diagnosis imports it
    assert rd.noun_phrase("The sink.") == "The sink"
    assert rd.noun_phrase("Answer: 'the red cup'\nbecause ...") == "the red cup"
    assert rd.noun_phrase("   ") == ""


def test_downsize_caps_the_long_side_and_masks_come_back_to_size():
    rd = load_script("reason_diag")
    img = np.zeros((300, 600, 3), np.uint8)
    small, f = rd.downsize(img, 150)
    assert small.shape[:2] == (75, 150) and f == 0.25
    assert rd.downsize(img, None)[0] is img and rd.downsize(img, 1000)[1] == 1.0
    back = rd.resize_mask(rect(0, 38, 0, 75, 75, 150), 300, 600)
    assert back.shape == (300, 600) and back[:150, :300].all() and not back[160:, 310:].any()


class FakeText:
    tok = None

    def __call__(self, image, phrases, k):
        # one instance per phrase: the lower-right quarter
        return [[(rect(12, 24, 16, 32), 0.9)] for _ in phrases]


def fake_scorer(rd):
    import s3_pool as S3P
    import ta_test_eval as TTE
    nodes = M.encode(np.stack([rect(0, 24, 0, 32), rect(0, 12, 0, 16)]))
    s = SimpleNamespace(pool=True, text=FakeText(), S3P=S3P, device="cpu",
                        ps={"instances_per_phrase": 3, "max_instances": 160, "dup_iou": 0.95},
                        nodes=lambda image: (nodes, None))
    s.logits = lambda image, key, pool, queries: [np.arange(pool.n, dtype=float)[None]]   # the last candidate wins
    s.pools = lambda image, texts: TTE.Scorer.pools(s, image, texts)
    return s


def test_diag_pool_is_the_scorers_pool_and_the_pick_is_scored_at_full_size():
    rd = load_script("reason_diag")
    d = object.__new__(rd.Diag)
    import ta_test_eval as TTE
    from treeground import tree as T
    from treeground import tree_attn as TA
    d.TTE, d.T, d.TA, d.s = TTE, T, TA, fake_scorer(rd)
    image = np.zeros((H, W, 3), np.uint8)
    text = "what holds the water?"
    ref = d.s.pools(image, [f"{text} <SEL>"])[0]
    mine = d.pool(image, d.s.S3P.phrases(f"{text} <SEL>"))
    assert [M.decode(r).sum() for r in mine.rles] == [M.decode(r).sum() for r in ref.rles] and mine.n_grid == ref.n_grid
    gt = rect(24, 48, 32, 64, 48, 64)                          # the image at twice the scored size
    r = d(image, gt, "k", text, d.s.S3P.phrases(f"{text} <SEL>"))
    assert r["pick_kind"] == "inst" and r["pick_iou"] == 1.0 and r["best_iou"] == 1.0 and r["pool"] == 3


def test_summarize_reports_every_setting():
    rd = load_script("reason_diag")
    df = pd.DataFrame({"gt_frac": [0.01, 0.3], **{f"{s}_{k}": [0.2, 0.8] for s in rd.SETTINGS
                                                  for k in ("pick_iou", "best_iou")}})
    out = rd.summarize(df)
    assert set(out["settings"]) == set(rd.SETTINGS) and out["settings"]["full"]["pick_gIoU"] == 0.5
    assert out["settings"]["rewrite"]["by_size"]["0%-5%"]["n"] == 1
