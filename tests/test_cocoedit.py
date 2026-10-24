import hashlib
import importlib.util
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from treeground import cocoedit_metrics as CM
from treeground.data import cocoedit_bench as CB

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = script("e_cocoedit")
rng = np.random.default_rng(0)


def png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def test_psnr_hand_computed_cap_and_empty_mask():
    src = np.zeros((2, 2, 3), np.float32)
    out = src.copy()
    out[0, 0] = 0.1                  # inside the mask: squared error 0.01 on 3 channels
    out[1, 1] = 1.0                  # outside the mask: ignored
    mask = np.array([[1, 1], [0, 0]], np.float32)
    mse = 3 * 0.01 / (2 * 3)
    assert CM.psnr(out, src, mask) == pytest.approx(10 * math.log10(1 / mse), rel=1e-5)
    assert CM.psnr(src, src, mask) == CM.PSNR_CAP
    assert math.isnan(CM.psnr(out, src, np.zeros((2, 2), np.float32)))
    num, den = CM.sq_error(out, src, mask)
    assert (num, den) == (pytest.approx(0.03, rel=1e-5), 6.0)
    # pooled: summed errors over samples, one identical sample adds only pixels
    assert CM.pooled_psnr(np.array([0.03, 0.0]), np.array([6.0, 6.0])) == pytest.approx(10 * math.log10(12 / 0.03),
                                                                                          rel=1e-5)
    assert CM.pooled_psnr(np.zeros(2), np.array([6.0, 6.0])) == CM.PSNR_CAP


def test_sample_metrics_cap_and_clip():
    src = rng.random((24, 24, 3)).astype(np.float32)
    mask = np.zeros((24, 24), bool)
    mask[:12] = True
    out = src.copy()
    out[12:] = 0                      # changes only in the edit region
    r = CM.sample_metrics(out, src, mask, None)
    assert r["psnr"] == CM.PSNR_CAP and r["at_cap"] and r["psnr_capped"] == CM.PSNR_CLIP
    assert r["sq_err"] == 0 and r["sq_count"] == 24 * 12 * 3 and math.isnan(r["lpips"])
    e = CM.sample_metrics(out, src, np.zeros((24, 24), bool), None)
    assert math.isnan(e["psnr"]) and math.isnan(e["psnr_capped"]) and not e["at_cap"]


def test_released_ssim_ignores_mask_and_masked_ssim_does_not():
    pytest.importorskip("skimage")
    src = rng.random((40, 40, 3)).astype(np.float32)
    out = src.copy()
    out[:, 25:] = rng.random((40, 15, 3))   # change far from the left part
    left = np.zeros((40, 40), bool)
    left[:, :12] = True
    full = np.ones((40, 40), bool)
    from skimage.metrics import structural_similarity
    plain = sum(structural_similarity(out[..., c], src[..., c], data_range=1.0, win_size=11, gaussian_weights=True,
                                      use_sample_covariance=False) for c in range(3)) / 3
    (r_left, m_left), (r_full, m_full) = CM.ssim(out, src, left), CM.ssim(out, src, full)
    assert r_left == r_full == pytest.approx(plain, abs=1e-12) and r_left < 0.9
    assert m_left == pytest.approx(1.0, abs=1e-6)  # the 11-tap window around columns < 12 never reaches column 25
    assert m_full == pytest.approx(r_full, abs=1e-7)
    assert math.isnan(CM.ssim(out, src, np.zeros((40, 40), bool))[1])


def test_preserved_composite_and_resize():
    src, out = np.zeros((4, 4, 3)), np.ones((4, 4, 3))
    keep = np.zeros((4, 4), bool)
    keep[0] = True
    c = CM.preserved_composite(out, src, keep)
    assert (c[0] == 1).all() and (c[1:] == 0).all()
    small = np.full((2, 2, 3), 0.5)
    big = CM.to_source_size(small, (4, 6))
    assert big.shape == (4, 6, 3) and np.allclose(big, 127 / 255)   # uint8 truncation then bilinear
    assert CM.to_source_size(src, (4, 4)) is src


def test_category_overall_and_imgedit_parse():
    scores = {"a": [1.0, 3.0], "b": [4.0, float("nan")], "c": [2.0, 2.5, 3.0]}
    overall, per = CM.category_overall(scores, ("a", "b", "c"))
    assert per == {"a": 2.0, "b": 4.0, "c": 2.5} and overall == pytest.approx(17 / 6)
    assert math.isnan(CM.category_overall(scores, ("a", "d"))[0])          # a missing category is not skipped
    o2, per2 = CM.category_overall({"a": [1.0, 1.0, 2.0]}, ("a",), 2)
    assert per2["a"] == 1.33 and o2 == 1.33
    text = ("Brief reasoning: No fur change: wrong subject.\nPrompt Compliance: 1\nVisual Seamlessness: 5\n"
            "Physical & Detail Fidelity: 4")
    assert CM.imgedit_score(text) == 3.33
    assert math.isnan(CM.imgedit_score("Prompt Compliance: high"))


def test_mask_inversion_and_splits():
    raw = np.array([[255, 0], [129, 128]], np.uint8)
    assert (CB.preserved_mask(raw, (2, 2), False) == np.array([[1, 0], [1, 0]], bool)).all()
    assert (CB.preserved_mask(raw, (2, 2), True) == np.array([[0, 1], [0, 1]], bool)).all()
    assert CB.preserved_mask(raw, (4, 4), False).shape == (4, 4)
    g = {"ref_image_path": "fullset/background_change/en/x.png"}
    assert CB.describe("gedit_full", g) == ("background_change", True, True, False)
    assert CB.describe("gedit_full", {"ref_image_path": "fullset/subject-add/en/x.png"})[1:3] == (False, True)
    assert CB.describe("gedit_full", {"ref_image_path": "fullset/tone_transfer/en/x.png"})[1:3] == (False, False)
    assert CB.describe("imgedit_full", {"edit_type": "background"}) == ("background", True, True, True)
    assert CB.describe("imgedit_full", {"edit_type": "add"})[1:3] == (False, True)
    assert CB.describe("imgedit_full", {"edit_type": "extract"})[1:3] == (False, False)


@pytest.fixture
def bench(tmp_path):
    img = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    mask = np.full((6, 8), 255, np.uint8)
    mask[1:3, 2:5] = 0
    meta, z = {}, zipfile.ZipFile(tmp_path / "singleturn.zip", "w")
    for i, t in enumerate(("adjust", "background", "add", "style")):
        meta[str(i)] = {"ref_image_path": f"singleturn/animal/{i}.jpg", "mask_image_path": f"singleturn/animal/{i}_m.png",
                        "caption": f"edit {t}", "edited_object": "a dog", "edit_type": t}
        z.writestr(meta[str(i)]["ref_image_path"], png(img))
        z.writestr(meta[str(i)]["mask_image_path"], png(mask))
    z.close()
    (tmp_path / "imgedit").mkdir()
    (tmp_path / "singleturn.zip").rename(tmp_path / "imgedit" / "singleturn.zip")
    (tmp_path / "imgedit" / "metafile_imgeditbench.json").write_text(json.dumps(meta))
    return tmp_path, img, mask


def test_load_imgedit_edit_regions(bench):
    root, img, mask = bench
    items = {s.category: s for s in CB.load("imgedit_full", root)}
    assert items["adjust"].referent == "the dog" and items["adjust"].sample_id == "imgedit_full:0"
    image, edit = items["adjust"].load()
    assert (image == img).all() and (edit == (mask <= 128)).all()
    _, bg = items["background"].load()
    assert (bg == (mask > 128)).all()                                     # background: the white part is edited
    assert [items[k].localized for k in ("adjust", "background", "add", "style")] == [True, True, False, False]
    assert [items[k].mllm for k in ("adjust", "background", "add", "style")] == [True, True, True, False]
    smoke = C.evaluated(list(items.values()) * 1, True, 1)
    assert len(smoke) == 4


def test_conditions_and_bbox():
    cfg = {"cocoedit": {"sources": {"ours": {}, "sam3text": {}},
                        "editors": {"qwen_image_edit": {"conditions": ["free", "ours_blend", "bbox_blend", "gt_blend"]},
                                    "flux_fill": {"conditions": ["free"]}}}}
    assert C.used_sources(cfg, "qwen_image_edit") == ["ours"]
    with pytest.raises(Exception):
        C.conditions(cfg, "flux_fill")
    cfg["cocoedit"]["editors"]["flux_fill"]["conditions"] = ["ours_blend"]
    with pytest.raises(Exception):
        C.conditions(cfg, "flux_fill")
    m = np.zeros((5, 6), bool)
    m[1, 1] = m[3, 4] = True
    b = C.bbox_mask(m)
    assert b.sum() == 12 and b[1:4, 1:5].all() and not C.bbox_mask(np.zeros((2, 2), bool)).any()
    assert C.mask_source("qwen_box_sam_blend") == "qwen_box_sam" and C.mask_source("free") is None


def test_qwen_conditions_composite_from_raw(bench, tmp_path):
    root, img, mask = bench
    items = sorted(CB.load("imgedit_full", root), key=lambda s: s.sample_id)
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    edited = np.zeros_like(img)
    for s in items:
        C.save_png(edited, raw / "images" / f"{C.safe_name(s.sample_id)}.png")
    loc = [s for s in items if s.localized]
    empty = json.dumps(None)
    sel = {"ours": pd.DataFrame({"sample_id": [s.sample_id for s in loc], "rle": [empty] * len(loc)}).set_index("sample_id")}
    cfg = {"cocoedit": {"mask": {"dilate_px": 0, "feather_px": 0.0}, "seed": 0},
           "edit": {"editors": {"qwen_image_edit": {}}}}
    for cond in ("free", "gt_blend", "ours_blend"):
        d = tmp_path / cond
        (d / "images").mkdir(parents=True)
        rows = {r["sample_id"]: r for r in C.run_condition(cfg, "qwen_image_edit", cond, items, sel, raw, None, d)}
        for s in items:
            out = C.load_png(d / "images" / f"{C.safe_name(s.sample_id)}.png")
            _, gt = s.load()
            if not s.localized or cond == "free":
                assert rows[s.sample_id]["mode"] == "raw" and (out == edited).all()
            elif cond == "ours_blend":
                assert rows[s.sample_id]["mode"] == "empty_mask" and (out == img).all()
            else:
                assert (out[gt] == 0).all() and (out[~gt] == img[~gt]).all()


def test_judge_requests_and_dedup(tmp_path):
    J = script("e_cocoedit_judge")
    prompts = J.imgedit_prompts()
    assert sorted(prompts) == ["action", "add", "adjust", "background", "compose", "extract", "remove", "replace",
                               "style"]
    # byte-identical copy of PKU-YuanGroup/ImgEdit@c14480ac Benchmark/Basic/prompts.json (git blob ce940910)
    data = J.PROMPTS_FILE.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == "ce9409100b71f6dcecfc34483ce009268f67f157"

    class S:
        category, instruction = "remove", "Remove the cat."
    img = np.zeros((4, 4, 3), np.uint8)
    r = J.imgedit_request(prompts, S, img, img, "gt_blend")
    assert "editing instruction is : Remove the cat.." in r["text"] and len(r["images"]) == 2
    msg = J.chat(r)[0]["content"]
    assert msg[0]["type"] == "text" and [m["type"] for m in msg[1:]] == ["image", "image"]
    assert J.chat(dict(r, text_first=False))[0]["content"][-1]["type"] == "text"
    for name, v in (("a", 1), ("b", 2), ("c", 1)):
        C.save_png(np.full((2, 2, 3), v, np.uint8), tmp_path / f"{name}.png")
    groups = J.unique_outputs({k: tmp_path / f"{k}.png" for k in "abc"})
    assert sorted(map(sorted, groups.values())) == [["a", "c"], ["b"]]
