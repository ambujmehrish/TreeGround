import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from treeground import mllm

REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def qwen_processor():
    from transformers import AutoConfig, AutoProcessor
    try:
        kw = dict(revision=REVISION, cache_dir=os.environ.get("TG_HF_CACHE"), local_files_only=True)
        return AutoProcessor.from_pretrained(REPO_ID, **kw), AutoConfig.from_pretrained(REPO_ID, **kw)
    except OSError:
        pytest.skip("pinned Qwen3-VL processor files are not available locally (set TG_HF_CACHE)")


def test_candidate_layers():
    assert mllm.candidate_layers(36) == [9, 12, 18, 24]


def test_coverage_is_raster_and_area_weighted():
    gh, gw = 3, 4
    mask = np.zeros((30, 40), dtype=bool)
    mask[0:10, 30:40] = True      # token (0, 3)
    mask[20:25, 0:10] = True      # half of token (2, 0)
    cov = mllm.coverage(mask, gh, gw)
    assert cov.shape == (12,)
    assert cov[3] == pytest.approx(1.0) and cov[8] == pytest.approx(0.5)
    assert np.count_nonzero(cov) == 2
    ptr, idx, w = mllm.sparse_coverages([mask], gh, gw)
    pooled = mllm.pool(torch.arange(12, dtype=torch.float32)[:, None], ptr, idx, w)
    assert pooled.item() == pytest.approx((3 * 1.0 + 8 * 0.5) / 1.5)


def test_processor_patch_order_is_raster_over_merged_grid():
    proc, cfg = qwen_processor()
    patch, merge = cfg.vision_config.patch_size, cfg.vision_config.spatial_merge_size
    cell = patch * merge
    gh, gw = 10, 14  # 320x448 px: above min_pixels and a multiple of 32, so no resize
    img = np.zeros((gh * cell, gw * cell, 3), dtype=np.uint8)
    for r in range(gh):
        for c in range(gw):
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (10 * r, 10 * c, 0)
    out = proc.image_processor(images=[Image.fromarray(img)], return_tensors="pt")
    thw = out["image_grid_thw"][0].tolist()
    assert mllm.merged_grid(out["image_grid_thw"], merge) == (gh, gw) and thw == [1, gh * merge, gw * merge]
    t = cfg.vision_config.temporal_patch_size
    pix = out["pixel_values"].reshape(gh * gw, merge * merge, 3, t, patch, patch)
    mean, std = proc.image_processor.image_mean, proc.image_processor.image_std
    red = (pix[:, :, 0].mean(dim=(-1, -2, -3)) * std[0] + mean[0]) * 255
    green = (pix[:, :, 1].mean(dim=(-1, -2, -3)) * std[1] + mean[1]) * 255
    for k in range(gh * gw):
        assert torch.allclose(red[k], torch.full((merge * merge,), 10.0 * (k // gw)), atol=1.0)
        assert torch.allclose(green[k], torch.full((merge * merge,), 10.0 * (k % gw)), atol=1.0)


def test_rope_index_places_image_tokens_in_raster_order():
    proc, cfg = qwen_processor()
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    merge = cfg.vision_config.spatial_merge_size
    img = Image.fromarray(np.zeros((3 * 32, 5 * 32, 3), dtype=np.uint8))
    text = mllm.query_prompt(proc, "Locate the mug. <SEL>", True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    pos, _ = Qwen3VLModel.get_rope_index(SimpleNamespace(config=cfg), inputs["input_ids"], inputs["image_grid_thw"])
    gh, gw = mllm.merged_grid(inputs["image_grid_thw"], merge)
    where = mllm.image_token_positions(inputs["input_ids"], cfg.image_token_id)
    assert len(where) == gh * gw
    hw = pos[1:, 0, where] - pos[1:, 0, where[0]][:, None]
    k = torch.arange(gh * gw)
    assert torch.equal(hw[0], k // gw) and torch.equal(hw[1], k % gw)


def test_real_images_token_count():
    root = os.environ.get("TG_TEST_REAL_IMAGES")
    if not root:
        pytest.skip("set TG_TEST_REAL_IMAGES to a directory of real images")
    proc, cfg = qwen_processor()
    files = sorted(Path(root).glob("*.jpg"))
    assert files
    for f in files:
        img = Image.open(f).convert("RGB")
        text = mllm.image_prompt(proc)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
        gh, gw = mllm.merged_grid(inputs["image_grid_thw"], cfg.vision_config.spatial_merge_size)
        assert len(mllm.image_token_positions(inputs["input_ids"], cfg.image_token_id)) == gh * gw
