import base64
import io
import json

import numpy as np
from PIL import Image

from test_s2 import box, load_script
from treeground import masks as M
from treeground import tree as T
from treeground.data import paco


class FakeScorer:
    # the Scorer pieces record() uses: grid nodes, SAM 3 text instances, the s3_pool helpers, logits
    def __init__(self, s3p, h, w):
        self.S3P, self.device = s3p, "cpu"
        self.ps = {"instances_per_phrase": 2, "max_instances": 10, "dup_iou": 0.95}
        self.h, self.w = h, w
        self.seen = []

    def nodes(self, image):
        return M.encode(np.stack([box(0, 40, 0, 60), box(0, 40, 0, 20)])), None

    def text(self, image, phrases, k):
        handle, mug = np.zeros((self.h, self.w), bool), np.zeros((self.h, self.w), bool)
        handle[10:20, 45:58] = True
        mug[0:40, 0:60] = True
        return [[(handle, 0.4), (mug, 0.9)], [(mug, 0.95)]]

    def logits(self, image, key, pool, queries):
        self.seen.append((pool, queries))
        lg = np.full((1, pool.n), -5.0)
        lg[0, 2] = 3.0                        # our pick: the handle instance (pool index n_grid + 0)
        return [lg]


def test_record_stores_pool_scores_and_sam3_top1(monkeypatch):
    fa = load_script("fig_assets")
    s3p = load_script("s3_pool")
    h, w = 64, 80
    image = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    hm = np.zeros((h, w), bool)
    hm[10:20, 45:58] = True
    img = paco.PacoImage(7, "x.jpg", h, w,
                         [paco.Obj(1, "mug", M.encode(box(0, 40, 0, 60)[None])[0])],
                         [paco.Part(2, 1, "mug", "handle", M.encode(hm[None])[0])])
    q = next(q for q in paco.i1_queries(img)[0] if q.kind == "part")
    s = FakeScorer(s3p, h, w)
    r = fa.record(s, image, img, q)
    assert r["phrases"] == ["the handle of the mug", "the mug"] and r["n_grid"] == 2
    # rank-major order: handle (phrase 0, rank 0), mug (phrase 1, rank 0); mug (phrase 0, rank 1) duplicates it
    assert [(m["phrase"], m["rank"]) for m in r["instances"]] == [(0, 0), (1, 0)]
    assert r["pick"] == 2 and r["pick_iou"] == 1.0 and r["sam3_top1"] == 2 and r["best"] == 2
    assert len(r["pool"]) == len(r["logits"]) == len(r["parents"]) == 4
    assert s.seen[0][1][0].kind == "part" and s.seen[0][0].n_grid == 2
    Image.open(io.BytesIO(base64.b64decode(r["image_jpeg_b64"])))
    json.dumps(r)
    back = M.decode({"size": r["gt_part"]["size"], "counts": r["gt_part"]["counts"].encode()})
    assert back.sum() == hm.sum()
