import io
import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from treeground import masks as M
from treeground.data import paco
from treeground.data.images import load_rgb

# Edit sets for task 3 (deviation D14). Downloaded files are untrusted: parquet via pandas, metadata via json,
# images and masks decoded by PIL from the parquet bytes or read from the zip archives without extraction.

PARTEDIT_SPLITS = ("real", "synth")
NON_LOCAL = re.compile(r"\b(entire|whole|overall|image|photo|picture|background|scene)\b", re.I)
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)

PACO_EDIT_TEMPLATES = {"color": "make the {part} of the {object} {value}",
                       "material": "make the {part} of the {object} look like {value}"}
PACO_EDIT_VALUES = {"color": ("red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black"),
                    "material": ("gold", "wood", "marble", "glass", "copper", "stone", "rubber", "silver")}
PACO_FILL = {"color": "a {value} {part} of a {object}", "material": "a {part} of a {object} made of {value}"}
PACO_TARGET = {"color": "a photo of a {object} with a {value} {part}",
               "material": "a photo of a {object} with a {part} made of {value}"}


@dataclass
class EditSample:
    sample_id: str
    image_key: str      # bootstrap group: one source image
    subset: str
    instruction: str    # editor input (instruction editors)
    referent: str       # target phrase for our selector and SAM 3 text
    fill_prompt: str    # FLUX Fill prompt
    src_caption: str    # CLIP directional similarity captions
    tgt_caption: str
    load: Callable[[], tuple[np.ndarray, np.ndarray]]  # RGB uint8 image, bool GT edit mask


def referent(phrase: str) -> str:
    return "the " + _ARTICLE.sub("", phrase.strip().rstrip("."))


def _png(data: bytes, mode: str) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert(mode))


def _checked(image: np.ndarray, mask: np.ndarray, sid: str) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != image.shape[:2]:
        raise ValueError(f"{sid}: mask {mask.shape} differs from the image {image.shape[:2]}")
    return image, mask


def partedit(root: str | Path) -> list[EditSample]:
    out = []
    for split in PARTEDIT_SPLITS:
        files = sorted(Path(root).glob(f"data/{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"PartEdit-Bench {split} parquet missing under {root}/data")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for r in df.itertuples():
            sid = f"partedit:{split}:{int(r.id)}"
            subject, part = r.subject.replace("_", " "), r.part.replace("_", " ")

            def load(r=r, sid=sid):
                # gt_mask: white = the edited part
                return _checked(_png(r.original_image["bytes"], "RGB"), _png(r.gt_mask["bytes"], "L") > 127, sid)
            out.append(EditSample(sid, sid, split, r.instructp2p_edit2.replace("_", " "),
                                  f"the {part} of the {subject}", r.p2p_prompt.replace("_", " "),
                                  r.prompt_original.replace("_", " "), r.p2p_prompt.replace("_", " "), load))
    return out


@lru_cache(maxsize=4)
def _zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _masked_bench(root: Path, name: str, meta_file: str, archive: str, subset, keep, max_frac: float
                  ) -> list[EditSample]:
    meta = json.loads((root / meta_file).read_text())
    z = _zip(str(root / archive))
    out = []
    for key in sorted(meta):
        v = meta[key]
        if not keep(v) or NON_LOCAL.search(v["edited_object"]):
            continue
        # mask PNG: white = preserved, black = edited region (the official PSNR scripts score the white region)
        edited = _png(z.read(v["mask_image_path"]), "L") <= 128
        if not 0 < edited.mean() <= max_frac:
            continue
        sid = f"{name}:{key}"

        def load(v=v, sid=sid):
            return _checked(_png(z.read(v["ref_image_path"]), "RGB"), _png(z.read(v["mask_image_path"]), "L") <= 128,
                            sid)
        ref = referent(v["edited_object"])
        out.append(EditSample(sid, f"{name}:{v['ref_image_path']}", subset(v), v["caption"].strip(), ref,
                              v["caption"].strip(), f"a photo of {ref}", v["caption"].strip(), load))
    return out


def _group(v: dict) -> str:
    return v["ref_image_path"].split("/")[1]


def gedit(root: str | Path, groups: list[str], max_frac: float) -> list[EditSample]:
    return _masked_bench(Path(root), "gedit", "geditbench/metafile_geditbench.json", "geditbench/fullset.zip",
                         _group, lambda v: _group(v) in groups, max_frac)


def imgedit(root: str | Path, edit_types: list[str], max_frac: float) -> list[EditSample]:
    return _masked_bench(Path(root), "imgedit", "imgedit/metafile_imgeditbench.json", "imgedit/singleturn.zip",
                         lambda v: v["edit_type"], lambda v: v["edit_type"] in edit_types, max_frac)


@dataclass
class PacoEditPair:
    img: paco.PacoImage
    part: paco.Part
    kind: str
    value: str


def paco_edit_pairs(images: dict[int, paco.PacoImage], n: int, seed: int, min_area: int, per_image: int
                    ) -> list[PacoEditPair]:
    # pairs eligible as I1 part queries (one instance of the object category, one mask of the part)
    pool = []
    for iid in sorted(images):
        img = images[iid]
        parts = {p.ann_id: p for p in img.parts}
        for q in paco.i1_queries(img)[0]:
            if q.kind == "part" and M.areas([parts[q.gt[0]].rle])[0] >= min_area:
                pool.append((img, parts[q.gt[0]]))
    rng = np.random.default_rng(seed)
    kinds = sorted(PACO_EDIT_TEMPLATES)
    taken: dict[int, int] = defaultdict(int)
    out = []
    for i in rng.permutation(len(pool)):
        img, part = pool[i]
        if taken[img.image_id] >= per_image:
            continue
        taken[img.image_id] += 1
        kind = kinds[int(rng.integers(len(kinds)))]
        vals = PACO_EDIT_VALUES[kind]
        out.append(PacoEditPair(img, part, kind, vals[int(rng.integers(len(vals)))]))
        if len(out) == n:
            break
    return out


def paco_edit(ann_root: str | Path, coco_root: str | Path, n: int, seed: int, min_area: int, per_image: int,
              resolve: Callable[[Path], Path] = Path) -> list[EditSample]:
    out = []
    for p in paco_edit_pairs(paco.load(ann_root, "val"), n, seed, min_area, per_image):
        fmt = {"object": paco.display(p.part.obj_name), "part": paco.display(p.part.part_name), "value": p.value}
        sid = f"paco_edit:{p.img.image_id}:{p.part.ann_id}"

        def load(p=p, sid=sid):
            image = load_rgb(resolve(paco.image_path(coco_root, p.img)))
            if image.shape[:2] != (p.img.h, p.img.w):
                raise ValueError(f"{sid}: image {image.shape[:2]} differs from the annotation {(p.img.h, p.img.w)}")
            return image, M.decode(p.part.rle).astype(bool)
        out.append(EditSample(sid, p.img.key, p.kind, PACO_EDIT_TEMPLATES[p.kind].format(**fmt),
                              "the {part} of the {object}".format(**fmt), PACO_FILL[p.kind].format(**fmt),
                              "a photo of a {object}".format(**fmt), PACO_TARGET[p.kind].format(**fmt), load))
    return out
