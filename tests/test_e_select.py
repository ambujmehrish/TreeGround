import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from test_s2 import load_script
from treeground import masks as M
from treeground import tree as T
from treeground.data import edit_bench as EB

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def es():
    return load_script("e_select")


@pytest.fixture(scope="module")
def ee2():
    return load_script("e_edit_s2")


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def configs(monkeypatch):
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    return {n: load_config(ROOT / "configs" / f"{n}.yaml") for n in
            ("edit", "edit_s1", "edit_s2", "edit_s2_s1", "edit_s2_long")}


def test_d23_configs_keep_the_d14_cache_identities(es, monkeypatch):
    cfgs = configs(monkeypatch)
    E = es.EE
    monkeypatch.setattr(E.load_script("x_transfer"), "node_cap", lambda: {"k": 256, "order": "stability"})

    def hashes(cfg):
        s = E.select_identity(cfg, "partedit", Path("r"), ["a", "b"], False)
        return s, [E.edit_identity(cfg, ed, "h") for ed in E.EDITORS], E.stage_tag(cfg)
    assert hashes(cfgs["edit_s2"]) == hashes(cfgs["edit"])
    assert hashes(cfgs["edit_s2_s1"]) == hashes(cfgs["edit_s1"]) == hashes(cfgs["edit_s2_long"])
    s2t = es.X2.S2T
    assert [s2t.s2_stage(cfgs[n]) for n in ("edit_s2", "edit_s2_s1", "edit_s2_long")] == \
        ["s2", "s2_from_s1", "s2_from_s1_long"]
    assert cfgs["edit_s2_long"]["edit_select"]["rerank"] == "none"
    assert cfgs["edit_s2_s1"]["edit_select"]["rerank"] == "rule"


def test_chain_referents_and_queries(es):
    assert es.chain_parts("the handle of the mug") == ("mug", "handle")
    assert es.chain_parts("the left wing of the toy plane") == ("toy plane", "left wing")
    assert es.chain_parts("the red car") is None
    cfg = {"edit_select": {"answer": "{referent} <SEL>"}}
    q = es.referent_query(cfg, "the handle of the mug")
    assert (q.user, q.answer) == ("Locate the handle of the mug.", "the handle of the mug <SEL>")
    c = es.X2.chain_query("edit|chain", "mug", "handle")
    assert c.answer == "mug <SEL> handle <SEL>" and "<SEL>" not in c.user


def test_tree_pick_stays_inside_the_object_pick(es):
    h, w = 40, 40
    masks = [box(h, w, 0, 20, 0, 40), box(h, w, 0, 10, 0, 10), box(h, w, 20, 40, 0, 40), box(h, w, 20, 30, 0, 10)]
    tree = T.build_tree(M.encode(np.stack(masks)))
    flat = np.array([0.1, 0.2, 0.3, 0.9])
    obj, part = np.array([0.9, 0.0, 0.5, 0.0]), np.array([0.0, 0.4, 0.0, 0.8])
    assert es.picks(tree, flat, None) == (3, -1, 3)
    assert es.picks(tree, flat, [obj, part]) == (3, 0, 1)          # node 3 is outside object 0: flat != tree
    assert es.picks(tree, flat, [np.array([0, 0.9, 0, 0]), part]) == (3, 1, 1)  # leaf object: itself


class FakeSam3:
    def __init__(self, logits, masks):
        self.logits, self.masks = logits, masks

    def get_vision_features(self, pixel_values):
        return None

    def __call__(self, vision_embeds, input_ids, attention_mask):
        return SimpleNamespace(pred_logits=self.logits[None], presence_logits=torch.zeros(1, 1),
                               pred_masks=self.masks[None])


class FakeProc:
    def __call__(self, images=None, text=None, return_tensors=None):
        out = SimpleNamespace(pixel_values=torch.zeros(1), input_ids=torch.zeros(1), attention_mask=torch.zeros(1))
        out.to = lambda _: out
        return out


def test_text_instances_extend_the_sam3text_top_instance(es):
    B = es.B
    logits = torch.tensor([0.1, 2.0, -1.0, 1.0])
    masks = torch.full((4, 8, 8), -10.0)
    for i in range(4):
        masks[i, i:i + 2] = 10.0
    st = object.__new__(B.Sam3Text)
    st.model, st.proc, st.device, st.dtype = FakeSam3(logits, masks), FakeProc(), "cpu", torch.float32
    st.s = {"mask_threshold": 0.5, "score_threshold": 0.5}
    image = np.zeros((16, 16, 3), np.uint8)
    inst = es.text_instances(st, image, "the cup", 3)
    top = st(image, ["the cup"])[0]
    assert [round(s, 4) for _, s in inst] == [round(float(torch.sigmoid(logits[i]) * 0.5), 4) for i in (1, 3, 0)]
    assert np.array_equal(inst[0][0], top[0]) and inst[0][1] == pytest.approx(top[1])


def sel_rows(n=6):
    rows = []
    for i in range(n):
        rows.append({"sample_id": f"s{i}", "image": f"im{i // 2}", "subset": "a", "chain": i < 4,
                     "ours2_iou": 0.6, "ours2tree_iou": 0.7 if i < 4 else 0.6, "nodes_only_iou": 0.3,
                     "nodes_rule_iou": 0.35, "sam3_iou": 0.5, "best_node_iou": 0.65, "best_pool_iou": 0.8,
                     "ours2_kind": "ref0" if i % 2 else "node", "ours2tree_kind": "node", "tree_object_kind": "obj0",
                     "flat_outside_object": i == 0, "flat_in_object_frac": 0.5, "tree_step2_candidates": 4,
                     "ours2_rle": "a", "ours2tree_rle": "a" if i >= 4 else "b", "ours2_above_tau_rej": False,
                     "instances": 3, "nodes_capped": 256})
    return pd.DataFrame(rows)


def test_localization_reports_both_picks_and_instance_rates(es):
    out = es.localization(sel_rows(), {"n": 50, "seed": 0})
    assert out["ours2tree_success@0.5"]["mean"] == 1.0 and out["s2_nodes_only_success@0.5"]["mean"] == 0.0
    assert out["best_pool_mask_iou"]["mean"] == pytest.approx(0.8)
    assert out["ours2_instance_pick_rate"] == 0.5 and out["ours2_pick_kind"] == {"node": 0.5, "ref": 0.5}
    assert out["flat_outside_tree_object_rate"] == 0.25 and out["chain_referents"] == 4
    assert out["flat_tree_same_pick_rate"] == pytest.approx(2 / 6)


def test_d23_conditions_blend_with_the_d14_raw_edit(ee2, tmp_path):
    E = ee2.E
    h = w = 32
    image = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    gt, flat, tree = box(h, w, 0, 10, 0, 10), box(h, w, 20, 30, 20, 30), box(h, w, 0, 10, 0, 10)
    s = EB.EditSample("t:0", "t:0", "x", "make it red", "the cup", "a red cup", "a cup", "a red cup",
                      lambda: (image, gt))
    base = tmp_path / "base"
    (base / "t_0").mkdir(parents=True)
    Image.fromarray(255 - image).save(base / "t_0" / "raw.png")
    sel = pd.DataFrame([{"sample_id": "t:0", "ours_rle": json.dumps(None), "sam3_rle": json.dumps(None),
                         "ours2_rle": json.dumps(M.encode(flat)[0]), "ours2tree_rle": json.dumps(M.encode(tree)[0])}])
    cfg = {"edit": {"mask": {"dilate_px": 2, "feather_px": 1.0}, "seed": 0, "editors": {"qwen_image_edit": {
        "conditions": ["ours2_blend", "ours2tree_blend"]}}}}
    classes = dict(E.EDITOR_CLASSES)
    E.EDITOR_CLASSES["qwen_image_edit"] = lambda _: ee2.RawFromBase(base)
    try:
        rows = E.run_edits(cfg, "qwen_image_edit", [s], sel, tmp_path / "out")
    finally:
        E.EDITOR_CLASSES.update(classes)
    assert list(rows.condition) == ["ours2_blend", "ours2tree_blend"]
    out = {c: np.asarray(Image.open(tmp_path / "out" / "t_0" / f"{c}.png")) for c in rows.condition}
    assert (out["ours2_blend"][flat] == 255 - image[flat]).all() and (out["ours2_blend"][gt] == image[gt]).all()
    assert (out["ours2tree_blend"][gt] == 255 - image[gt]).all()
    m = ee2.sample_masks(sel.set_index("sample_id").loc["t:0"], gt)
    assert set(m) == {"ours", "gt", "sam3text", "ours2", "ours2tree"} and not m["ours"].any()
    with pytest.raises(Exception):
        ee2.new_conditions({"edit_s2": {"conditions": {"flux_fill": ["gt_fill"]}}}, "flux_fill")


def test_condition_block_scores_the_new_sources(ee2):
    g = pd.DataFrame({"image": ["a", "b"], "mask": ["ours2tree"] * 2, "empty_mask": [False] * 2,
                      "mask_iou": [0.6, 0.2], "mask_area_frac": [0.1, 0.2], "gt_area_frac": [0.1, 0.1],
                      **{k: [0.1, 0.2] for k in ee2.E.METRICS}})
    b = ee2.condition_block(g, {"n": 20, "seed": 0})
    assert b["mask_success@0.5"]["mean"] == 0.5 and b["mask_area_over_gt"] == pytest.approx(1.5)


def test_select_sample_runs_both_modes_on_a_tiny_s2_model(es, monkeypatch):
    from test_s2 import H, LAYER, W, wrapped
    from treeground.selector import Selector
    model, proc, extra = wrapped(es.X2.S2T)
    model.eval()
    rgb = (np.random.default_rng(1).random((H, W, 3)) * 255).astype(np.uint8)
    gt = box(H, W, 10, 30, 10, 40)
    nodes = [box(H, W, 0, 60, 0, 80), box(H, W, 10, 30, 10, 40), box(H, W, 60, 96, 0, 128)]
    monkeypatch.setattr(es, "candidates", lambda *a: (M.encode(np.stack(nodes)), np.array([0.9, 0.8, 0.7]), 5))
    inst = {"the handle of the mug": [(box(H, W, 12, 30, 10, 40), 0.6), (np.zeros((H, W), bool), 0.1)],
            "the mug": [(box(H, W, 0, 62, 0, 80), 0.7)], "the red mug": [(box(H, W, 0, 62, 0, 80), 0.7)]}
    monkeypatch.setattr(es, "text_instances", lambda st, image, phrase, k: inst[phrase][:k])
    cfg = {"edit_select": {"sam3_instances": 3, "answer": "{referent} <SEL>"}}
    models = (None, None, model, proc, Selector(64, 64).eval(), extra["sel_id"], LAYER, "cpu")
    s = EB.EditSample("p:0", "p:0", "x", "make it red", "the handle of the mug", "", "", "", lambda: (rgb, gt))
    r = es.select_sample(models, None, cfg, s, {}, (2, 0.25, "run"))
    assert r["chain"] and r["nodes_capped"] == 3 and r["instances"] == 2   # the empty instance is dropped
    assert r["best_node_iou"] == 1.0 and r["sam3_iou"] == pytest.approx(0.9)
    for k in ("ours2", "ours2tree"):
        pick = M.decode(json.loads(r[f"{k}_rle"])).astype(bool)
        assert r[f"{k}_iou"] == pytest.approx((pick & gt).sum() / (pick | gt).sum())
    assert np.isfinite(r["nodes_rule_iou"]) and r["tree_object_kind"] != "none"
    plain = EB.EditSample("p:1", "p:1", "x", "make it red", "the red mug", "", "", "", lambda: (rgb, gt))
    q = es.select_sample(models, None, cfg, plain, {}, None)
    assert not q["chain"] and q["instances"] == 1 and q["ours2tree_rle"] == q["ours2_rle"]
    assert np.isnan(q["nodes_rule_iou"]) and not q["flat_outside_object"]


def test_edit_localization_table_lists_d23_rows(es, tmp_path):
    loc = es.localization(sel_rows(), {"n": 20, "seed": 0})
    (tmp_path / "e_select_gedit_s2_from_s1_merge.json").write_text(json.dumps({"localization": loc}))
    cols, rows = load_script("make_tables").edit_localization(tmp_path)
    got = {r["source"]: r for r in rows}
    assert got["S2 (from S1) flat (D23)"]["instance_rate"] == 0.5 and got["S2 (from S1) tree (D23)"]["n"] == 6
    assert got["best of pool (nodes + SAM 3 inst.)"]["iou"]["mean"] == pytest.approx(0.8)
