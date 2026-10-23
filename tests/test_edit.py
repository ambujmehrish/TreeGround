import importlib.util
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from treeground import edit_metrics as E
from treeground import masks as M
from treeground.data import edit_bench as EB
from treeground.data import paco

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e_edit", ROOT / "scripts" / "e_edit.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

rng = np.random.default_rng(0)


def img(h=40, w=50):
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def test_masked_psnr_known_mse_and_identity():
    a = np.full((10, 10, 3), 100, np.uint8)
    b = a.copy()
    region = box(10, 10, 0, 5, 0, 10)
    b[~region] = 0  # changes outside the region do not count
    assert E.masked_psnr(a, b, region) == E.PSNR_CAP
    b[region] = 110
    assert E.masked_psnr(a, b, region) == pytest.approx(10 * math.log10(1 / (10 / 255) ** 2))
    assert math.isnan(E.masked_psnr(a, b, np.zeros((10, 10), bool)))


def test_ssim_identity_and_constant_images():
    a = img()
    np.testing.assert_allclose(E.ssim_map(a, a), 1.0, atol=1e-12)
    x, y = np.full((20, 20, 3), 51, np.uint8), np.full((20, 20, 3), 153, np.uint8)
    mx, my = 0.2, 0.6
    expected = (2 * mx * my + E.SSIM_C1) / (mx ** 2 + my ** 2 + E.SSIM_C1)  # zero variance: luminance term only
    np.testing.assert_allclose(E.ssim_map(x, y), expected, atol=1e-9)
    smooth = np.repeat(np.linspace(0, 255, 40).astype(np.uint8)[None, :, None], 30, 0).repeat(3, 2)
    noisy = np.clip(smooth.astype(int) + rng.integers(-40, 40, smooth.shape), 0, 255).astype(np.uint8)
    assert E.ssim_map(smooth, noisy).mean() < 0.5


def test_blend_alpha_and_composite_keep_outside_pixels():
    m = box(60, 60, 20, 30, 20, 30)
    a = E.blend_alpha(m, 4, 2.0)
    assert a.min() >= 0 and a.max() <= 1 and (a[m] == 1).all()
    far = ~E.dilate(m, 4 + 8)
    assert np.abs(a[far]).max() < 1e-4
    orig, edited = img(60, 60), img(60, 60)
    out = E.composite(orig, edited, a)
    assert (out[far] == orig[far]).all() and (out[m] == edited[m]).all()
    assert E.dilate(m, 3).sum() > m.sum() and (E.dilate(m, 0) == m).all()


def test_directional_similarity():
    s, t0 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert E.directional(s, s, t0, np.array([0, 0, 1.0])) == 0.0
    assert E.directional(s, np.array([0, 0, 2.0]), t0, np.array([1.0, -1.0, 1.0])) == pytest.approx(
        np.dot([-1, 0, 1], [1, -2, 1]) / (np.sqrt(2) * np.sqrt(6)))
    assert E.cosine([1, 0], [2, 0]) == pytest.approx(1.0)


def test_mask_iou_and_change():
    a, b = box(10, 10, 0, 4, 0, 5), box(10, 10, 2, 6, 0, 5)
    assert E.mask_iou(a, b) == pytest.approx(10 / 30)
    assert E.mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 0.0
    x = np.zeros((4, 4, 3), np.uint8)
    y = x.copy()
    y[0, 0] = 255
    assert E.abs_change(x, y)[0, 0] == pytest.approx(1.0) and E.abs_change(x, y)[1:].sum() == 0


def test_referent_template_reproduces_i1_part_query():
    assert X.REFERENT_TEMPLATE.format(referent="the handle of the mug") == \
        paco.PART_TEMPLATE.format(part="handle", object="mug")
    assert EB.referent("A mirror frame.") == "the mirror frame"
    assert EB.referent("person's right arm") == "the person's right arm"
    assert EB.NON_LOCAL.search("the entire photo") and not EB.NON_LOCAL.search("mirror frame")


def test_conditions_map_to_masks():
    cfg = yaml.safe_load((ROOT / "configs" / "edit.yaml").read_text())
    for editor in X.EDITORS:
        for c in cfg["edit"]["editors"][editor]["conditions"]:
            assert X.mask_source(c) in (None, "ours", "gt", "sam3text")
    assert "free" not in cfg["edit"]["editors"]["flux_fill"]["conditions"]
    with pytest.raises(Exception):
        X.mask_source("box_fill")


def test_fit_dims_and_seeds():
    w, h = X.fit_dims(640, 480, 1024 * 1024, 16)
    assert w % 16 == 0 and h % 16 == 0 and abs(w * h / 2 ** 20 - 1) < 0.05 and abs(w / h - 4 / 3) < 0.02
    assert X.sample_seed(0, "a") == X.sample_seed(0, "a") != X.sample_seed(0, "b")
    assert X.safe_name("gedit:fullset/a b.png") == "gedit_fullset_a_b.png"


def test_partedit_loader_polarity(tmp_path):
    (tmp_path / "data").mkdir()
    image = img(32, 32)
    gt = box(32, 32, 4, 10, 4, 12)
    for split in EB.PARTEDIT_SPLITS:
        pd.DataFrame([{"id": 0, "original_image": {"bytes": png(image), "path": None},
                       "gt_mask": {"bytes": png(gt.astype(np.uint8) * 255), "path": "0_gt.png"},
                       "subject": "racing_car", "part": "car_hood", "edit": "rusted",
                       "instructp2p_edit2": "Change the racing_car car_hood to rusted car_hood",
                       "prompt_original": "a racing_car", "p2p_prompt": "a racing_car with rusted car_hood"}]
                     ).to_parquet(tmp_path / "data" / f"{split}-00000-of-00001.parquet")
    items = EB.partedit(tmp_path)
    assert [s.sample_id for s in items] == ["partedit:real:0", "partedit:synth:0"]
    s = items[0]
    assert s.referent == "the car hood of the racing car" and "_" not in s.instruction + s.tgt_caption
    im, m = s.load()
    assert (im == image).all() and (m == gt).all()


def test_masked_bench_polarity_and_filters(tmp_path):
    (tmp_path / "imgedit").mkdir()
    image = img(30, 40)
    edited = box(30, 40, 5, 15, 5, 20)
    keep = np.where(edited, 0, 255).astype(np.uint8)  # white = preserved
    meta, z = {}, zipfile.ZipFile(tmp_path / "imgedit" / "singleturn.zip", "w")
    rows = [("1", "adjust", "the suit", keep), ("2", "style", "suit", keep), ("3", "adjust", "entire image", keep),
            ("4", "remove", "dog", np.full((30, 40), 255, np.uint8)), ("5", "remove", "dog", np.zeros((30, 40), np.uint8))]
    for key, et, obj, mk in rows:
        z.writestr(f"singleturn/animal/{key}.jpg", png(image))
        z.writestr(f"singleturn/animal/{key}_mask.png", png(mk))
        meta[key] = {"ref_image_path": f"singleturn/animal/{key}.jpg", "mask_image_path": f"singleturn/animal/{key}_mask.png",
                     "caption": f"Change the {obj}.", "edited_object": obj, "edit_type": et}
    z.close()
    (tmp_path / "imgedit" / "metafile_imgeditbench.json").write_text(json.dumps(meta))
    items = EB.imgedit(tmp_path, ["adjust", "remove"], 0.9)
    # style excluded by type, 'entire image' as non-local, empty and full-image masks by the area filter
    assert [s.sample_id for s in items] == ["imgedit:1"]
    im, m = items[0].load()
    assert (m == edited).all() and items[0].referent == "the suit" and items[0].subset == "adjust"


def paco_image(iid, objs):
    im = paco.PacoImage(iid, f"{iid}.jpg", 64, 64)
    aid = iid * 100
    for name, parts in objs:
        aid += 1
        oid = aid
        im.objects.append(paco.Obj(oid, name, M.encode(box(64, 64, 0, 64, 0, 64))[0]))
        for pname, area_side in parts:
            aid += 1
            im.parts.append(paco.Part(aid, oid, name, pname, M.encode(box(64, 64, 0, area_side, 0, area_side))[0]))
    return im


def test_paco_edit_pairs_are_seeded_and_eligible():
    images = {i: paco_image(i, [("mug", [("handle", 40), ("rim", 10)]), ("cup", [("base", 40)]),
                                ("cup", [("base", 40)])]) for i in range(1, 30)}
    a = EB.paco_edit_pairs(images, 10, 0, 32 * 32, 1)
    b = EB.paco_edit_pairs(images, 10, 0, 32 * 32, 1)
    assert [(p.img.image_id, p.part.ann_id, p.kind, p.value) for p in a] == \
        [(p.img.image_id, p.part.ann_id, p.kind, p.value) for p in b]
    assert len(a) == 10 and len({p.img.image_id for p in a}) == 10
    # cup has two instances (not an I1 query) and the rim is below min_area
    assert all(p.part.obj_name == "mug" and p.part.part_name == "handle" for p in a)
    assert all(p.value in EB.PACO_EDIT_VALUES[p.kind] for p in a)
    assert len(EB.paco_edit_pairs(images, 100, 0, 32 * 32, 2)) == 29


class FakeEditor:
    def __init__(self, cfg):
        self.calls = []

    def __call__(self, image, s, mask, seed):
        self.calls.append(None if mask is None else mask.sum())
        return 255 - image


def edit_cfg(editor):
    cfg = yaml.safe_load((ROOT / "configs" / "edit.yaml").read_text())
    cfg["edit"]["mask"] = {"dilate_px": 2, "feather_px": 1.0}
    return cfg


@pytest.mark.parametrize("editor", ["qwen_image_edit", "flux_fill"])
def test_run_edits_conditions_with_a_stub_editor(tmp_path, monkeypatch, editor):
    monkeypatch.setitem(X.EDITOR_CLASSES, editor, FakeEditor)
    image, gt = img(48, 48), box(48, 48, 10, 20, 10, 20)
    ours = box(48, 48, 30, 40, 30, 40)
    s = EB.EditSample("t:0", "t:0", "x", "make it red", "the cup", "a red cup", "a cup", "a red cup",
                      lambda: (image, gt))
    sel = pd.DataFrame([{"sample_id": "t:0", "ours_rle": json.dumps(M.encode(ours)[0]), "sam3_rle": json.dumps(None)}])
    cfg = edit_cfg(editor)
    rows = X.run_edits(cfg, editor, [s], sel, tmp_path)
    conds = cfg["edit"]["editors"][editor]["conditions"]
    assert list(rows.condition) == conds
    out = {c: np.asarray(Image.open(tmp_path / "t_0" / f"{c}.png")) for c in conds}
    far_gt = ~E.dilate(gt, 8)
    first = "gt_blend" if editor == "qwen_image_edit" else "gt_fill"
    assert (out[first][far_gt] == image[far_gt]).all() and (out[first][gt] == 255 - image[gt]).all()
    ours_c = first.replace("gt", "ours")
    assert (out[ours_c][ours] == 255 - image[ours]).all() and (out[ours_c][gt] == image[gt]).all()
    sam = first.replace("gt", "sam3text")
    assert (out[sam] == image).all() and rows.set_index("condition").empty_mask[sam]
    if editor == "qwen_image_edit":
        assert (out["free"] == 255 - image).all()


def test_summarize_reports_conditions_and_localization():
    rows = []
    for i in range(6):
        for c, mask in (("free", "none"), ("ours_blend", "ours")):
            rows.append({"sample_id": f"s{i}", "image": f"im{i // 2}", "subset": "a", "condition": c, "mask": mask,
                         "empty_mask": False, "psnr_out": 20.0 + i, "ssim_out": 0.9, "lpips_out": np.nan,
                         "l1_in": 0.1, "lpips_in": np.nan, "clip_dir": 0.2, "clip_t": 0.25,
                         "mask_iou": 0.6 if mask == "ours" else np.nan, "mask_area_frac": 0.1, "gt_area_frac": 0.1,
                         "ours_iou": 0.6, "sam3_iou": 0.4, "best_node_iou": 0.8, "ours_above_tau_rej": i % 2 == 0,
                         "sam3_above_threshold": True})
    m, table = X.summarize(pd.DataFrame(rows), ["free", "ours_blend"], {"n": 100, "seed": 0})
    assert m["conditions"]["free"]["all"]["psnr_out"]["mean"] == pytest.approx(22.5)
    assert m["conditions"]["free"]["all"]["lpips_out"] is None and "mask_iou" not in m["conditions"]["free"]["all"]
    assert m["conditions"]["ours_blend"]["all"]["mask_success@0.5"]["mean"] == 1.0
    assert m["localization"]["sam3text_success@0.5"]["mean"] == 0.0
    assert m["localization"]["ours_above_tau_rej_rate"] == 0.5
    assert list(table.condition) == ["free", "ours_blend"]
