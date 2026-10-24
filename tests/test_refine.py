import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("refine", ROOT / "scripts" / "refine.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402

RCFG = {"variants": ["r1", "r2", "r3", "r4"], "oracle_variants": ["r1", "r2"], "batch": 4,
        "mask_prompt": {"scale": 20.0, "bias": -10.0}, "negatives": {"max_points": 2, "min_distance": 2.0}}
BS = {"n": 50, "seed": 0}


def rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_box_matches_rle_boxes_and_interior_point_is_central():
    m = rect(40, 60, 5, 25, 10, 50)
    assert np.allclose(R.mask_box(m), M.boxes_xyxy(M.encode(m))[0])
    x, y = R.interior_point(m)
    assert m[int(y), int(x)] and abs(x - 30) <= 1 and abs(y - 15) <= 1


def test_negative_points_lie_in_parent_outside_node_and_are_separated():
    parent = rect(60, 80, 0, 60, 0, 80)
    node = rect(60, 80, 20, 40, 30, 50)
    pts = R.negative_points(node, parent, 2, 2.0)
    assert pts.shape == (2, 2)
    for x, y in pts:
        assert parent[int(y), int(x)] and not node[int(y), int(x)]
    assert np.linalg.norm(pts[0] - pts[1]) > 5
    assert np.array_equal(pts, R.negative_points(node, parent, 2, 2.0))
    assert R.negative_points(node, None, 2, 2.0).shape == (0, 2)
    # a one-pixel ring around the node has clearance 1 < min_distance: no negatives
    ring = rect(60, 80, 19, 41, 29, 51)
    assert R.negative_points(node, ring, 2, 2.0).shape == (0, 2)


def test_prompt_variants():
    parent = rect(60, 80, 0, 60, 0, 80)
    node = rect(60, 80, 20, 40, 30, 50)
    p = {v: R.make_prompt(v, node, parent, RCFG) for v in RCFG["variants"]}
    assert all(np.allclose(q.box, [30, 20, 50, 40]) for q in p.values())
    assert len(p["r1"].points) == 0 and p["r1"].mask is None
    assert p["r2"].labels.tolist() == [1] and node[int(p["r2"].points[0, 1]), int(p["r2"].points[0, 0])]
    assert p["r3"].mask is node and len(p["r3"].points) == 0
    assert p["r4"].labels.tolist() == [1, 0, 0] and np.allclose(p["r4"].points[0], p["r2"].points[0])
    assert R.make_prompt("r4", node, None, RCFG).labels.tolist() == [1]


def test_mask_prompt_is_logit_scaled_low_res_mask():
    lp = R.mask_prompt(rect(100, 200, 0, 100, 0, 100), (16, 16), 20.0, -10.0)
    assert lp.shape == (1, 1, 16, 16)
    assert torch.allclose(lp[0, 0, :, :7], torch.tensor(10.0)) and torch.allclose(lp[0, 0, :, 9:], torch.tensor(-10.0))


def test_choose_takes_highest_predicted_iou_first_on_ties():
    assert R.choose(torch.tensor([[0.1, 0.9, 0.3], [0.5, 0.5, 0.2]])).tolist() == [1, 0]


@pytest.fixture(scope="module")
def tiny():
    # synthetic SAM 3 tracker (random weights, 1-layer ViT): checks prompt plumbing only
    from transformers import Sam3TrackerConfig, Sam3TrackerModel
    c = Sam3TrackerConfig()
    bc = c.vision_config.backbone_config
    bc.hidden_size, bc.num_hidden_layers, bc.intermediate_size, bc.num_attention_heads = 32, 1, 64, 2
    torch.manual_seed(0)
    model = Sam3TrackerModel(c).eval()
    g = torch.Generator().manual_seed(1)
    emb = [torch.randn(1, ch, s, s, generator=g) for ch, s in ((32, 288), (64, 144), (256, 72))]
    return model, emb


def post(masks, sizes, binarize=False):
    return [F.interpolate(m, size=tuple(s), mode="bilinear", align_corners=False) for m, s in zip(masks, sizes)]


def test_run_prompts_groups_without_padding_and_is_batch_invariant(tiny):
    model, emb = tiny
    h, w = 60, 80
    parent = rect(h, w, 0, 60, 0, 80)
    nodes = [rect(h, w, 20, 40, 30, 50), rect(h, w, 5, 30, 5, 30), rect(h, w, 30, 58, 40, 78)]
    prompts = [R.make_prompt(v, n, parent, RCFG) for n in nodes for v in RCFG["variants"]]
    a = R.run_prompts(model, post, emb, prompts, (h, w), 1008, RCFG, 0.0, torch.float32)
    b = R.run_prompts(model, post, emb, prompts, (h, w), 1008, {**RCFG, "batch": 1}, 0.0, torch.float32)
    assert len(a) == len(prompts)
    for (ma, sa), (mb, sb) in zip(a, b):
        assert ma.shape == (h, w) and ma.dtype == bool
        assert abs(sa - sb) < 1e-4 and (ma != mb).mean() < 1e-3


def test_chain_pick_follows_natural_search():
    tree = T.from_parents(np.array([3, 0, 3, -1]))  # 0 -> {1}, 2 leaf, root 3
    sig = np.array([[0.9, 0.2, 0.8], [0.1, 0.7, 0.95]])
    assert R.chain_pick(sig, tree, 0.5) == (0, 1)
    assert R.chain_pick(sig, tree, 0.92) == (-1, 2)  # step 1 rejected: step 2 searches the whole tree
    assert R.chain_pick(sig, tree, 0.96) == (-1, -1)
    assert R.chain_pick(np.zeros((2, 0)), T.from_parents(np.array([-1])), -np.inf) == (-1, -1)


def test_single_block_applies_rejection_like_v2():
    df = pd.DataFrame({"kind": ["object", "part", "part"], "image": ["a", "a", "b"], "gt_area": [10, 20, 40],
                       "inter": [5, 10, 0], "union": [10, 20, 40]})
    acc = np.array([True, False, True])
    blk = R.single_block(df, df.inter, df.union, acc, BS)
    assert blk["gIoU"]["mean"] == pytest.approx(0.5 / 3)
    assert blk["cIoU"]["value"] == pytest.approx(5 / 70)
    assert blk["by_kind"]["part"]["mean"] == 0.0 and blk["by_kind"]["object"]["mean"] == 0.5
    assert blk["success@0.5"]["mean"] == pytest.approx(1 / 3)
    nr = R.single_block(df, df.inter, df.union, np.ones(3, bool), BS)
    assert nr["gIoU"]["mean"] == pytest.approx(1 / 3) and nr["cIoU"]["value"] == pytest.approx(15 / 70)


def test_chain_summary_requires_both_steps():
    ch = pd.DataFrame({"image": ["a", "b", "c"], "single_level": [False, True, False]})
    for mode in R.MODES:
        for v in ("r0", "r1"):
            ch[f"{mode}_obj_{v}"] = [0.6, 0.7, 0.2]
            ch[f"{mode}_part_{v}"] = [0.5, 0.1, 0.9] if v == "r0" else [0.6, 0.6, 0.9]
    out = R.chain_summary(ch, {"variants": ["r1"]}, BS)["no_rejection"]
    assert out["r0"]["all"]["chain_success_end_to_end"]["mean"] == pytest.approx(1 / 3)
    assert out["r1"]["all"]["chain_success_end_to_end"]["mean"] == pytest.approx(2 / 3)
    assert out["r1"]["two_level"]["chains"] == 2 and out["r1"]["single_level"]["chains"] == 1


class FakeProc:
    target_size = 1008
    post_process_masks = staticmethod(post)

    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


def test_refine_image_rows(tiny):
    model, emb = tiny
    h, w = 60, 80
    masks = [rect(h, w, 0, 60, 0, 80), rect(h, w, 20, 40, 30, 50), rect(h, w, 5, 30, 5, 30)]
    rles = M.encode(np.stack(masks))
    tree = T.build_tree(rles)
    gts = {7: M.encode(rect(h, w, 21, 40, 30, 50))[0], 8: M.encode(rect(h, w, 0, 60, 0, 80))[0]}
    s = R.ImageSel("sha", "paco_lvis:1", rles, tree,
                   [{"qid": "q1", "kind": "part", "gt": 7, "a": 2, "sigma": 0.9, "accept": True},
                    {"qid": "q2", "kind": "object", "gt": 8, "a": 0, "sigma": 0.1, "accept": False}],
                   [{"qid": "c1", "gt_obj": 8, "gt_part": 7, "cal": (0, -1), "nr": (0, 1)}])
    model.get_image_embeddings = lambda pix: emb
    single, chains = R.refine_image(model, FakeProc(), np.zeros((h, w, 3), np.uint8), s, gts, RCFG,
                                    R.P.ProposalConfig(dtype="float32"))
    q1, q2 = single
    assert q1["best"] == 1 and q2["best"] == 0
    assert q1["inter_r0"] == 0 and q1["union_r0"] == 25 * 25 + 19 * 20
    assert q1["inter_best_r0"] / q1["union_best_r0"] == pytest.approx(19 / 20)
    assert q2["inter_r0"] == q2["union_r0"] == h * w
    assert all(f"inter_{v}" in q1 for v in RCFG["variants"]) and "inter_gt_box" in q1 and "inter_best_r2" in q1
    c = chains[0]
    assert c["cal_obj_r0"] == 1.0 and c["cal_part_r0"] == 0.0 and c["nr_part_r0"] == pytest.approx(19 / 20)
    assert c["cal_part_r3"] == 0.0 and c["single_level"] is False


class Store:
    def __init__(self, z):
        self.z = z

    def load(self, sha1, keys):
        return {keys[0]: self.z}


class Probes:
    def __init__(self, p):
        self.p = p

    def get(self, sha1, offset, n):
        return self.p[offset:offset + n, 0].clone()


def test_select_image_matches_argmax_and_rejection():
    torch.manual_seed(0)
    sel = R.Selector(8, 6).eval()
    z, p = torch.randn(4, 8), torch.randn(5, 1, 6)
    tree = T.from_parents(np.array([4, 0, 4, 4, -1]))
    src = R.Source("v2", sel, Store(z), Path("."), 0, 3, 0.5, Path("run"), Path("run"), None)
    single = pd.DataFrame({"qid": ["q0", "q1"], "kind": ["object", "part"], "gt": ["[11]", "[12]"], "offset": [0, 1]})
    chains = pd.DataFrame({"qid": ["c0"], "gt": ["[11, 12]"], "offset": [2]})
    s = R.select_image(src, "sha", "k", [None] * 4, tree, single, chains, Probes(p), "cpu")
    with torch.no_grad():
        sig = torch.sigmoid(sel.scores(sel.queries(p[:, 0]), sel.nodes(z))).numpy()
    for i, q in enumerate(s.single):
        assert q["a"] == int(np.argmax(sig[i])) and q["sigma"] == pytest.approx(sig[i].max())
        assert q["accept"] == (sig[i].max() >= 0.5) and q["gt"] == [11, 12][i]
    c = s.chains[0]
    assert c["nr"] == R.chain_pick(sig[2:4], tree, -np.inf) and c["cal"] == R.chain_pick(sig[2:4], tree, 0.5)
    assert (c["gt_obj"], c["gt_part"]) == (11, 12)
    src.store = Store(torch.zeros(0, 8))
    empty = R.select_image(src, "sha", "k", [], T.from_parents(np.array([-1])), single, chains, Probes(p), "cpu")
    assert all(q["a"] == -1 and not q["accept"] for q in empty.single) and empty.chains[0]["cal"] == (-1, -1)


def test_single_summary_and_consistency_with_saved_table(tmp_path):
    rcfg = {"variants": ["r1"], "oracle_variants": ["r1"]}
    df = pd.DataFrame({"qid": ["a", "b"], "kind": ["object", "part"], "image": ["i1", "i2"], "gt_area": [10, 10],
                       "accept": [True, False], "sigma": [0.9, 0.2], "inter_r0": [5, 8], "union_r0": [10, 10],
                       "inter_r1": [9, 10], "union_r1": [10, 10], "pred_iou_r1": [0.8, 0.9]})
    absent = pd.DataFrame({"image": ["i3", "i3"], "accept": [False, True]})
    out = R.single_summary(df, absent, rcfg, BS)
    assert out["r0"]["calibrated"]["gIoU"]["mean"] == pytest.approx(0.25)
    assert out["r1"]["no_rejection"]["gIoU"]["mean"] == pytest.approx(0.95)
    assert out["r1"]["calibrated"]["gIoU_present_and_no_target"]["mean"] == pytest.approx((0.9 + 0 + 1 + 0) / 4)
    assert out["r1"]["mean_predicted_iou"] == pytest.approx(0.85)
    saved = pd.DataFrame({"qid": ["a", "b", "x"], "present": [True, True, False], "sel_iou": [0.5, 0.7, 0.0],
                          "max_sigma": [0.9, 0.2, 0.1]})
    ch = pd.DataFrame({"qid": ["c"], "cal_obj_r0": [0.6], "cal_part_r0": [0.7]})
    nat = tmp_path / "chains_natural.csv"
    pd.DataFrame({"qid": ["c", "c"], "structure": ["flat", "true"], "chain": [False, True]}).to_csv(nat, index=False)
    c = R.consistency(df, ch, saved, nat)
    assert c["single_sel_iou_agreement"] == 0.5 and c["single_sigma_max_abs_diff"] == 0.0
    assert c["chain_success_agreement"] == 1.0 and c["chains_missing_in_saved"] == 0
