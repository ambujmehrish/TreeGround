import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from treeground import masks as M

# Pascal-Part (Chen et al., CVPR 2014) on PASCAL VOC 2010. Annotations_Part/<image>.mat holds, per object, a
# silhouette mask and named part masks. Following PACO's convention of one mask per part type, numbered parts
# (wheel_1, wheel_2) and left/right parts (lwing, rwing) are merged into one part ("wheel", "wing").

CLASS_NAMES = {"aeroplane": "airplane", "diningtable": "dining table", "pottedplant": "potted plant",
               "tvmonitor": "monitor", "motorbike": "motorbike"}

SIDED = {"wing", "eye", "ear", "leg", "foot", "horn", "hand", "ebrow", "larm", "uarm", "lleg", "uleg",
         "fleg", "bleg", "fpa", "bpa", "fuleg", "flleg", "buleg", "blleg", "fho", "bho"}

PART_NAMES = {
    "body": "body", "stern": "stern", "wing": "wing", "tail": "tail", "engine": "engine", "wheel": "wheel",
    "fwheel": "wheel", "bwheel": "wheel", "saddle": "saddle", "handlebar": "handlebar", "chainwheel": "chain wheel",
    "headlight": "headlight", "head": "head", "eye": "eye", "beak": "beak", "torso": "torso", "neck": "neck",
    "leg": "leg", "foot": "foot", "cap": "cap", "frontside": "front side", "leftside": "side", "rightside": "side",
    "backside": "back side", "roofside": "roof", "leftmirror": "mirror", "rightmirror": "mirror",
    "fliplate": "license plate", "bliplate": "license plate", "door": "door", "window": "window", "ear": "ear",
    "nose": "nose", "fleg": "front leg", "fpa": "front paw", "bleg": "back leg", "bpa": "back paw",
    "muzzle": "muzzle", "horn": "horn", "fuleg": "front upper leg", "flleg": "front lower leg",
    "buleg": "back upper leg", "blleg": "back lower leg", "fho": "front hoof", "bho": "back hoof",
    "ebrow": "eyebrow", "mouth": "mouth", "hair": "hair", "larm": "lower arm", "uarm": "upper arm", "hand": "hand",
    "lleg": "lower leg", "uleg": "upper leg", "pot": "pot", "plant": "plant", "screen": "screen",
    "hfrontside": "front of the locomotive", "hleftside": "side of the locomotive",
    "hrightside": "side of the locomotive", "hbackside": "back of the locomotive", "hroofside": "locomotive roof",
    "coach": "coach", "cfrontside": "front of a coach", "cleftside": "side of a coach",
    "crightside": "side of a coach", "cbackside": "back of a coach", "croofside": "coach roof",
}


def part_name(raw: str) -> str:
    base = re.sub(r"_\d+$", "", raw)
    if base[:1] in ("l", "r") and base[1:] in SIDED:
        base = base[1:]
    if base not in PART_NAMES:
        raise ValueError(f"unknown Pascal-Part part name {raw!r}")
    return PART_NAMES[base]


@dataclass
class PPObject:
    index: int
    name: str
    rle: M.Rle
    parts: dict[str, M.Rle] = field(default_factory=dict)


@dataclass
class PPImage:
    image_id: str
    h: int
    w: int
    objects: list[PPObject]

    @property
    def key(self) -> str:
        return f"pascal_part:{self.image_id}"


def split_ids(voc_root: str | Path, split: str) -> list[str]:
    path = Path(voc_root) / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def image_path(voc_root: str | Path, image_id: str) -> Path:
    return Path(voc_root) / "JPEGImages" / f"{image_id}.jpg"


def load_image(ann_root: str | Path, image_id: str) -> PPImage | None:
    import scipy.io as sio
    path = Path(ann_root) / f"{image_id}.mat"
    if not path.exists():
        return None
    anno = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["anno"]
    objs = []
    h = w = None
    for k, o in enumerate(np.atleast_1d(anno.objects)):
        silh = np.asarray(o.mask, dtype=bool)
        if h is None:
            h, w = silh.shape
        elif silh.shape != (h, w):
            raise ValueError(f"{image_id}: object masks differ in size")
        cls = str(getattr(o, "class"))
        merged: dict[str, np.ndarray] = {}
        for p in (np.atleast_1d(o.parts) if np.size(o.parts) else []):
            pm = np.asarray(p.mask, dtype=bool)
            if pm.shape != (h, w):
                raise ValueError(f"{image_id}: part mask size differs")
            name = part_name(str(p.part_name))
            merged[name] = merged[name] | pm if name in merged else pm
        objs.append(PPObject(k, CLASS_NAMES.get(cls, cls), M.encode(silh[None])[0],
                             {n: M.encode(m[None])[0] for n, m in sorted(merged.items()) if m.any()}))
    if h is None:
        return None
    return PPImage(image_id, h, w, objs)


def chains(img: PPImage) -> list[tuple[str, str, str, M.Rle, M.Rle]]:
    # I1 analogue: object classes with exactly one instance in the image, each of their (merged) part types
    counts: dict[str, int] = {}
    for o in img.objects:
        counts[o.name] = counts.get(o.name, 0) + 1
    out = []
    for o in img.objects:
        if counts[o.name] != 1:
            continue
        for part, rle in o.parts.items():
            out.append((f"{img.key}|chain|{o.name}:{part}", o.name, part, o.rle, rle))
    return out
