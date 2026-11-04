import numpy as np
import pandas as pd
import pytest

from test_s2 import H, W, box, load_script
from treeground import masks as M
from treeground import tree as T
from treeground import tree_attn as TA


@pytest.fixture(scope="module")
def pdiag():
    return load_script("ta_pool_diag")


def test_source_labels(pdiag):
    imeta = [{"phrase": 0, "rank": 0, "score": 0.9}, {"phrase": 1, "rank": 0, "score": 0.8},
             {"phrase": 0, "rank": 2, "score": 0.3}]
    assert [pdiag.source_of(2, imeta, i) for i in (-1, 0, 1, 2, 3, 4)] == \
        ["none", "grid", "grid", "part_r1", "object_r1", "part_r3"]
    a, b = box(0, 10, 0, 10), box(0, 20, 0, 20)
    assert pdiag.containment(a, b) == 1.0 and pdiag.containment(b, a) == 0.25 and pdiag.containment(a * 0, b) == 0.0


class FakeScorer:
    # grid: the mug, its handle, a far box; text instances: part phrase -> [handle copy (rank 1), mug (rank 2)],
    # object phrase -> [mug (rank 1)]. augment adds rank by rank over the phrases: handle (part r1), mug (object r1),
    # then drops the part-r2 mug as a duplicate of it; grid duplicates are kept, as in the training pools
    def __init__(self, obj_logits, part_logits, anchor="hard"):
        import s3_pool as S3P
        self.S3P, self.device, self.anchor = S3P, "cpu", anchor
        self.ps = {"instances_per_phrase": 3, "max_instances": 160, "dup_iou": 0.95}
        self.mug, self.handle, self.far = box(10, 90, 10, 70), box(10, 40, 10, 40), box(20, 80, 90, 125)
        self.grid = M.encode(np.stack([self.mug, self.handle, self.far]))
        self.lg = (obj_logits, part_logits)

    def nodes(self, image):
        return self.grid, T.build_tree(self.grid).parent

    def text(self, image, phrases, k):
        out = []
        for p in phrases:
            out.append([(self.handle.copy(), 0.9), (self.mug.copy(), 0.4)] if p.startswith("the handle")
                       else [(self.mug.copy(), 0.95)])
        return out

    def logits(self, image, key, pool, queries):
        assert queries[0].kind == "chain" and pool.n == 5            # 3 grid + handle (part r1) + mug (object r1)
        return [np.array([self.lg[0][:pool.n], self.lg[1][:pool.n]])]


def test_diagnose_reads_picks_and_containment(pdiag):
    text = "Locate the handle of the mug. <SEL>"
    s = FakeScorer([5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 3.0, 2.0, 0.0])    # object: mug; flat part pick: the far box
    inst = {p: s.text(None, [p], 3)[0] for p in s.S3P.phrases(text)}
    row = pdiag.diagnose(s, np.zeros((H, W, 3), np.uint8), "k", "k|part|mug:handle", text, "mug", "handle",
                         s.handle, s.grid, inst)
    assert row["n_grid"] == 3 and row["n_inst"] == 2 and row["n_part_inst"] == 1
    assert row["flat_src"] == "grid" and row["flat_iou"] == 0.0 and row["flat_in_obj"] == 0.0
    assert row["hard_src"] == "part_r1" and row["hard_iou"] == 1.0 and row["hard_in_obj"] == 1.0
    assert row["rank1_present"] and row["rank1_iou"] == 1.0 and row["best_iou"] == 1.0
    assert row["best_src"] in ("grid", "part_r1") and row["obj_src"] == "grid" and not row["hard_fallback"]
    assert row["anchor_iou"] == row["hard_iou"]                          # the checkpoint's anchor is 'hard' here
    s2 = FakeScorer([0.0, 0.0, 9.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0], anchor="none")   # object pick: a leaf
    row2 = pdiag.diagnose(s2, np.zeros((H, W, 3), np.uint8), "k", "q", text, "mug", "handle", s2.handle, s2.grid, inst)
    assert row2["hard_fallback"] and row2["hard_src"] == row2["flat_src"] == "grid" and row2["flat_iou"] == 1.0
    assert row2["anchor_iou"] == row2["flat_iou"]
    empty = pdiag.diagnose(s2, np.zeros((H, W, 3), np.uint8), "k", "q", text, "mug", "handle", s2.handle, [],
                           {p: [] for p in inst})
    assert empty["flat_src"] == "none" and empty["best_iou"] == 0.0


def test_summary_aggregates(pdiag):
    df = pd.DataFrame([
        {"n_grid": 3, "n_inst": 2, "flat_iou": 1.0, "flat_src": "part_r2", "hard_iou": 1.0, "hard_src": "part_r2",
         "anchor_iou": 1.0, "rank1_iou": 0.2, "rank1_present": True, "best_iou": 1.0, "best_src": "part_r2",
         "obj_src": "grid", "flat_in_obj": 1.0, "hard_in_obj": 1.0, "hard_fallback": False},
        {"n_grid": 3, "n_inst": 1, "flat_iou": 0.0, "flat_src": "grid", "hard_iou": 0.6, "hard_src": "part_r1",
         "anchor_iou": 0.0, "rank1_iou": 0.6, "rank1_present": True, "best_iou": 0.7, "best_src": "grid",
         "obj_src": "grid", "flat_in_obj": 0.1, "hard_in_obj": 1.0, "hard_fallback": False}])
    res = pdiag.summarize(df)
    assert res["gIoU"]["flat"] == 0.5 and res["gIoU"]["hard"] == 0.8 and res["gIoU"]["rank1_part_instance"] == 0.4
    assert res["flat_pick"]["part_r2"] == {"share": 0.5, "gIoU": 1.0} and res["best_in_pool_source"]["grid"] == 0.5
    assert res["flat_vs_rank1"]["flat_better_by_0.1"] == 0.5 and res["flat_vs_rank1"]["rank1_better_by_0.1"] == 0.5
    assert res["flat_vs_rank1"]["flat_picks_other_than_rank1_and_wins_0.1"] == 0.5
    assert res["containment"]["flat_in_object_ge_0.8"] == 0.5 and res["containment"]["hard_minus_flat_gIoU"] == pytest.approx(0.3)
