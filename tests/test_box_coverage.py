import types

import numpy as np
import pandas as pd
import pytest
import torch

from test_s2 import load_script
from treeground import masks as M
from treeground.data import paco


@pytest.fixture(scope="module")
def bc():
    return load_script("box_coverage")


def rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def scene():
    h, w = 20, 20
    mug, handle = rect(h, w, 2, 18, 2, 14), rect(h, w, 6, 12, 14, 18)
    img = paco.PacoImage(1, "x.jpg", h, w)
    img.objects.append(paco.Obj(10, "mug", M.encode(mug[None])[0]))
    img.parts.append(paco.Part(11, 10, "mug", "handle", M.encode(handle[None])[0]))
    return img, mug, handle


def test_image_queries_link_part_to_its_object_phrase(bc):
    img, _, _ = scene()
    qs = bc.image_queries(img)
    assert [q["kind"] for q in qs] == ["object", "part"]
    obj, part = qs
    assert obj["obj_phrase"] is None and part["obj_phrase"] == obj["phrase"] and part["gt"] == 11


def test_query_rows_sets(bc):
    img, mug, handle = scene()
    qs = bc.image_queries(img)
    obj, part = qs
    empty = np.zeros_like(mug)
    # part: first box's single mask misses, its best-scored mask is half the handle, a second box hits exactly;
    # the object's box carries the handle as its third mask; the auto pool has only the mug
    half = handle.copy()
    half[:, 16:] = False
    prompted = {
        part["phrase"]: [(np.stack([empty, half, mug]), np.array([0.1, 0.9, 0.2]), empty),
                         (np.stack([handle, empty, empty]), np.array([0.5, 0.4, 0.3]), handle)],
        obj["phrase"]: [(np.stack([mug, mug | handle, handle]), np.array([0.9, 0.5, 0.1]), mug)],
    }
    # SAM 3 text: the part phrase's top instance is the mug, its second the half handle; the object phrase's
    # top instance is the mug alone
    text = {part["phrase"]: [(mug, 0.8), (half, 0.3)], obj["phrase"]: [(mug, 0.9)]}
    rows = bc.query_rows("paco_lvis:1", qs, paco.gt_rles(img), M.encode(mug[None]), prompted, text)
    o, p = rows
    assert o["auto"] == 1.0 and o["box1"] == 1.0 and o["obj3"] == o["box3"] == 1.0
    assert o["text1"] == o["text3"] == o["objtext3"] == o["union_text"] == o["union_all"] == 1.0
    assert p["auto"] == 0.0 and p["box1"] == 0.0
    assert p["box_top"] == pytest.approx(half.sum() / handle.sum())
    assert p["box3"] == 1.0 and p["obj3"] == 1.0 and p["boxes"] == 1.0 and p["union"] == 1.0
    assert p["n_boxes"] == 2 and p["n_obj_boxes"] == 1
    assert p["text1"] == 0.0 and p["text3"] == pytest.approx(half.sum() / handle.sum()) and p["objtext3"] == 0.0
    assert p["union_text"] == p["text3"] and p["union_all"] == 1.0


def test_query_rows_without_boxes_or_nodes(bc):
    img, _, _ = scene()
    qs = bc.image_queries(img)
    rows = bc.query_rows("paco_lvis:1", qs, paco.gt_rles(img), [], {}, {})
    for r in rows:
        assert all(r[s] == 0.0 for s in bc.SETS) and r["n_boxes"] == 0


def test_summarize_by_image(bc):
    df = pd.DataFrame([{"qid": f"q{i}", "kind": k, "image": f"im{i % 2}", "area_bin": "small", "n_boxes": i % 3,
                        "n_obj_boxes": 1, **{s: v for s in bc.SETS}}
                       for i, (k, v) in enumerate([("part", 0.6), ("part", 0.2), ("object", 0.9), ("part", 0.5)])])
    res = bc.summarize(df, {"n": 50, "seed": 0})
    assert res["part"]["queries"] == 3 and res["object"]["queries"] == 1
    assert res["part"]["union"]["hit@0.5"]["mean"] == pytest.approx(2 / 3)
    assert res["part"]["auto"]["mean_best_iou"]["mean"] == pytest.approx(1.3 / 3)
    assert res["part"]["no_box"] == pytest.approx(2 / 3)   # parts i = 0, 1, 3 have 0, 1, 0 boxes
    assert "small" in res["part"]["by_area_bin"]


def test_prompter_masks_keeps_box_order_and_three_masks(bc):
    h, w, H = 8, 10, 16

    class Sam:
        def get_image_embeddings(self, pix):
            return "emb"

        def __call__(self, image_embeddings, input_boxes, multimask_output):
            n, k = input_boxes.shape[1], 3 if multimask_output else 1
            pm = torch.full((1, n, k, H, H), -1.0)
            for i in range(n):          # mask i, j: box index marker in row i, column j
                for j in range(k):
                    pm[0, i, j, i, j] = 1.0
            return types.SimpleNamespace(pred_masks=pm, iou_scores=torch.arange(n * k, dtype=torch.float).view(1, n, k))

    class Proc:
        target_size = H

        def __call__(self, images, return_tensors):
            return types.SimpleNamespace(pixel_values=torch.zeros(1))

        def post_process_masks(self, masks, sizes, binarize):
            return [masks[0][..., :sizes[0][0], :sizes[0][1]]]

    p = bc.Prompter.__new__(bc.Prompter)
    p.pred = types.SimpleNamespace(sam=Sam(), sproc=Proc(), device="cpu", dtype=torch.float32,
                                   s={"sam_batch": 2, "mask_threshold": 0.0})
    boxes = [[[0, 0, 500, 500], [100, 100, 900, 900]], [], [[0, 0, 1000, 1000]]]
    out = p.masks(np.zeros((h, w, 3), np.uint8), boxes)
    assert [len(o) for o in out] == [2, 0, 1]
    for flat_i, (k, b) in enumerate([(0, 0), (0, 1), (2, 0)]):
        three, scores, single = out[k][b]
        row = flat_i % 2                       # index inside its SAM batch of 2
        assert three.shape == (3, h, w) and single.shape == (h, w)
        assert all(three[j, row, j] and three[j].sum() == 1 for j in range(3))
        assert single[row, 0] and single.sum() == 1
        assert scores.shape == (3,) and scores[0] == 3 * row
