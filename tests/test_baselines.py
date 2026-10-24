import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treeground import masks as M
from treeground.data import paco

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b_baselines", ROOT / "scripts" / "b_baselines.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

BS = {"n": 200, "seed": 0}


def rle(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return M.encode(m)[0]


def test_phrases_come_from_the_i1_templates():
    obj = paco.OBJECT_TEMPLATE.format(object="mug")
    part = paco.PART_TEMPLATE.format(object="mug", part="handle")
    assert B.phrase(obj) == "the mug" and B.phrase(part) == "the handle of the mug"
    assert B.concept(B.phrase(obj)) == "mug" and B.concept(B.phrase(part)) == "handle of the mug"
    assert B.phrase("man on the left") == "man on the left"


@pytest.mark.parametrize("text,box", [
    ('```json\n[\n\t{"bbox_2d": [100, 200, 300, 400], "label": "mug"}\n]\n```', [100, 200, 300, 400]),
    ('[{"bbox_2d": [10.5, 20, 30, 40]}, {"bbox_2d": [1, 2, 3, 4]}]', [10.5, 20, 30, 40]),
    ('{"bbox_2d": [-5, 0, 1200, 999], "label": "x"}', [0, 0, 1000, 999]),
    ('The mug is at [100, 200, 300, 400].', [100, 200, 300, 400]),
    ('```json\n[{"bbox_2d": [100, 200, 300', None),
    ('[{"bbox_2d": [300, 200, 100, 400]}]', None),
    ('[{"bbox_2d": [1, 2, 3]}]', None),
    ("There is no mug in the image.", None),
    ("[]", None),
])
def test_parse_box(text, box):
    assert B.parse_box(text) == box


def test_box_to_pixels_uses_relative_1000_grid():
    np.testing.assert_allclose(B.box_to_pixels([0, 500, 1000, 1000], 640, 480), [0, 240, 640, 480])


def test_shards_partition_images():
    keys = [f"k{i}" for i in range(23)] + ["k1"]
    parts = [B.shard_keys(keys, s, 4) for s in range(4)]
    flat = sum(parts, [])
    assert sorted(flat) == sorted(set(keys)) and len(flat) == len(set(flat))


def test_overlap_counts_and_no_target():
    gt, pred = rle(10, 10, 0, 4, 0, 5), rle(10, 10, 2, 6, 0, 5)
    o = B.overlap(pred, gt)
    assert (o["inter"], o["union"], o["pred_area"], o["gt_area"]) == (10, 30, 20, 20)
    assert o["iou"] == pytest.approx(1 / 3)
    assert B.overlap(None, gt)["iou"] == 0.0 and B.overlap(None, gt)["union"] == 20
    assert B.overlap(None, None)["iou"] == 1.0 and B.overlap(pred, None)["iou"] == 0.0


def row(qid, kind, image, inter, union, gt_area, present=True, pred_area=1):
    return {"qid": qid, "kind": kind, "image": image, "present": present, "inter": inter, "union": union,
            "gt_area": gt_area, "pred_area": pred_area, "iou": inter / union if union else float(pred_area == 0),
            "iou_any": inter / union if union else float(pred_area == 0)}


def test_summary_gIoU_cIoU_and_chains():
    df = pd.DataFrame([
        row("a|object|mug", "object", "a", 90, 100, 100),
        row("a|part|mug:handle", "part", "a", 1, 10, 8),
        row("b|object|cup", "object", "b", 0, 50, 50, pred_area=0),
        row("b|part|cup:rim", "part", "b", 30, 40, 35),
        row("g1", "absent", "c", 0, 0, 0, present=False, pred_area=0),
        row("g2", "absent", "c", 0, 7, 0, present=False, pred_area=7),
    ])
    out, ch = B.summarize(df, BS)
    pres = out["present"]
    assert pres["gIoU"]["mean"] == pytest.approx((0.9 + 0.1 + 0.0 + 0.75) / 4)
    assert pres["cIoU"]["value"] == pytest.approx(121 / 200)
    assert pres["success@0.5"]["mean"] == pytest.approx(0.5)
    assert out["by_kind"]["part"]["gIoU"]["mean"] == pytest.approx(0.425)
    assert out["no_target"]["abstention_rate"] == 0.5
    assert out["no_target"]["gIoU_present_and_no_target"]["mean"] == pytest.approx((1.75 + 1.0) / 6)
    assert list(ch.qid) == ["a|chain|mug:handle", "b|chain|cup:rim"]
    assert list(ch.step2) == [False, True] and list(ch.chain) == [False, False]
    assert list(ch.single_level) == [False, True]
    assert out["chains"]["all"]["final_part_success@0.5"]["mean"] == 0.5
    assert out["chains"]["two_level"]["chains"] == 1
    lo, hi = pres["cIoU"]["ci95"]
    assert lo <= pres["cIoU"]["value"] <= hi


def test_real_device_init_loads_exact_weights(tmp_path):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, modeling_utils
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, vocab_size=100)
    ref = Qwen2ForCausalLM(cfg)
    ref.save_pretrained(tmp_path)
    before = modeling_utils.PreTrainedModel.get_init_context
    with B.real_device_init():
        assert torch.linspace(0, 1, 3).device.type == "cpu"
        model = B.checked_load(Qwen2ForCausalLM, str(tmp_path), "main", None, dtype=torch.float32)
    assert modeling_utils.PreTrainedModel.get_init_context == before
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))


def test_real_device_init_handles_class_without_post_init(tmp_path):
    import torch
    from transformers import PretrainedConfig, PreTrainedModel, Qwen2Config, Qwen2ForCausalLM

    class WrapConfig(PretrainedConfig):
        model_type = "tg_wrap"

    class Wrap(PreTrainedModel):  # like Sa2VA's remote class: never calls post_init()
        config_class = WrapConfig

        def __init__(self, config):
            super().__init__(config)
            self.language_model = Qwen2ForCausalLM(Qwen2Config(
                hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, vocab_size=100, tie_word_embeddings=True))
            self.head = torch.nn.Linear(32, 3)
            self.scale = torch.nn.Parameter(torch.full((1,), torch.linspace(0, 1, 3)[1].item()))  # .item() in __init__

    ref = Wrap(WrapConfig())
    ref.save_pretrained(tmp_path)
    with pytest.raises((AttributeError, RuntimeError)):  # meta .item(), then the missing post_init() state
        Wrap.from_pretrained(tmp_path)
    with B.real_device_init():
        model = B.checked_load(Wrap, str(tmp_path), "main", None, dtype=torch.float32)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))
    assert model.language_model.lm_head.weight.data_ptr() == model.language_model.model.embed_tokens.weight.data_ptr()
