import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from treeground.data.edit_bench import EditSample, _png, _zip, referent

# Full GEdit-Bench-EN (606) and ImgEdit-Bench single-turn (737) with the wyh6666 masks, split as in CoCoEdit
# (arXiv 2602.14068) and the released geditbench/imgedit run_psnr_score.py (deviation D25)

DATASETS = ("gedit_full", "imgedit_full")
GEDIT_PSNR_SKIP = ("style_change", "tone_transfer", "subject-add")      # substring of ref_image_path
IMGEDIT_PSNR_SKIP = ("style", "extract", "add")                         # substring of edit_type
GEDIT_MLLM = ("background_change", "color_alter", "material_alter", "motion_change", "ps_human", "subject-add",
              "subject-remove", "subject-replace", "text_change")
IMGEDIT_MLLM = ("add", "adjust", "background", "replace", "remove", "action", "compose")
MLLM_CATEGORIES = {"gedit_full": GEDIT_MLLM, "imgedit_full": IMGEDIT_MLLM}


@dataclass
class BenchSample(EditSample):
    key: str            # metafile key (ImgEdit judge prompts and released result file names use it)
    category: str       # GEdit group / ImgEdit edit_type
    scored: bool        # in the PSNR/SSIM set
    mllm: bool          # in the MLLM Overall categories
    invert: bool        # ImgEdit background: the metric region is the black (object) part of the mask

    @property
    def localized(self) -> bool:
        # mask-guided conditions apply exactly to the PSNR/SSIM set; add/style/tone/extract edits use the free edit
        return self.scored


def preserved_mask(mask_png: np.ndarray, shape: tuple[int, int], invert: bool) -> np.ndarray:
    m = mask_png > 128
    if m.shape != shape:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(shape[::-1], Image.NEAREST)) > 127
    return ~m if invert else m


def _meta(root: Path, dataset: str) -> tuple[dict, str]:
    if dataset == "gedit_full":
        return json.loads((root / "geditbench/metafile_geditbench.json").read_text()), str(root / "geditbench/fullset.zip")
    if dataset == "imgedit_full":
        return json.loads((root / "imgedit/metafile_imgeditbench.json").read_text()), str(root / "imgedit/singleturn.zip")
    raise ValueError(f"unknown dataset {dataset}")


def describe(dataset: str, v: dict) -> tuple[str, bool, bool, bool]:
    if dataset == "gedit_full":
        cat = v["ref_image_path"].split("/")[1]
        return cat, not any(g in v["ref_image_path"] for g in GEDIT_PSNR_SKIP), cat in GEDIT_MLLM, False
    t = v["edit_type"]
    return t, not any(g in t for g in IMGEDIT_PSNR_SKIP), t in IMGEDIT_MLLM, "background" in t


def load(dataset: str, root: str | Path) -> list[BenchSample]:
    meta, archive = _meta(Path(root), dataset)
    out = []
    for key in sorted(meta):
        v = meta[key]
        cat, scored, mllm, invert = describe(dataset, v)
        sid = f"{dataset}:{key}"

        def ld(v=v, invert=invert):
            z = _zip(archive)
            image = _png(z.read(v["ref_image_path"]), "RGB")
            # GT edit region = complement of the preserved (metric) region
            return image, ~preserved_mask(_png(z.read(v["mask_image_path"]), "L"), image.shape[:2], invert)
        cap = v["caption"].strip()
        ref = referent(v["edited_object"])
        out.append(BenchSample(sid, f"{dataset}:{v['ref_image_path']}", cat, cap, ref, cap, f"a photo of {ref}", cap,
                               ld, key, cat, scored, mllm, invert))
    return out
