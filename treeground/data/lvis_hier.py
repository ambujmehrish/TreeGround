import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from treeground import masks as M

# Box-containment object hierarchy on LVIS v1 val, rebuilt with the rule of Wang et al., "Learning Visual
# Hierarchies in Hyperbolic Space for Image Retrieval" (arXiv 2411.17490, Sec. 3.1 and App. A/B.3): a box b is a
# child of a larger box a when >= 80% of b lies inside a; a (parent class, child class) pair is kept when it occurs
# at least MIN_FREQ times and in at least MIN_PROP of the parent class's instances (LVIS: 5 and 5%); boxes under
# 1% of the image are dropped. Chains: parent instances whose class occurs once in the image, each kept child
# class inside them (instances of one class merged into one mask, PACO convention). GT masks are LVIS masks.

CONTAIN = 0.8
MIN_FREQ = 5
MIN_PROP = 0.05
MIN_AREA_FRAC = 0.01
# a child box covering >= 90% of its parent box is the same entity labelled twice (OpenImages hypernyms such as
# person / man / human body); such pairs are dropped. For a box inside its parent the IoU equals the area ratio.
MAX_CHILD_FRAC = 0.9


@dataclass
class LVISObject:
    ann_id: int
    name: str
    box: np.ndarray  # x0, y0, x1, y1
    segmentation: object


@dataclass
class LVISImage:
    image_id: int
    file_name: str
    h: int
    w: int
    objects: list[LVISObject]

    @property
    def key(self) -> str:
        return f"lvis_val:{self.image_id}"


def display(name: str) -> str:
    # LVIS names carry a synonym after a space and a sense in parentheses: "monitor_(computer_equipment)
    # computer_monitor" -> "monitor"
    head = name.split(" ")[0]
    head = head.split("_(")[0]
    return head.replace("_", " ")


def load(path: str | Path) -> dict[int, LVISImage]:
    data = json.loads(Path(path).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    # file_name keeps the COCO folder (LVIS v1 val draws images from both train2017 and val2017)
    images = {im["id"]: LVISImage(im["id"], "/".join(im["coco_url"].rsplit("/", 2)[-2:]), im["height"], im["width"],
                                  [])
              for im in data["images"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        images[a["image_id"]].objects.append(
            LVISObject(a["id"], display(cats[a["category_id"]]), np.array([x, y, x + w, y + h], float),
                       a["segmentation"]))
    for im in images.values():
        im.objects.sort(key=lambda o: o.ann_id)
    return images


def _kept(img: LVISImage) -> list[LVISObject]:
    return [o for o in img.objects
            if (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]) >= MIN_AREA_FRAC * img.h * img.w]


def _contained(objs: list[LVISObject]) -> list[tuple[int, int]]:
    b = np.stack([o.box for o in objs]) if objs else np.zeros((0, 4))
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return [(i, j) for i in range(len(objs)) for j in range(len(objs))
            if i != j and area[j] < MAX_CHILD_FRAC * area[i] and inter[i, j] >= CONTAIN * area[j]
            and objs[i].name != objs[j].name]


def class_pairs(images: dict[int, LVISImage]) -> dict[tuple[str, str], dict]:
    freq: dict[tuple[str, str], int] = {}
    parents_with: dict[tuple[str, str], set[int]] = {}
    instances: dict[str, int] = {}
    for im in images.values():
        objs = _kept(im)
        for o in objs:
            instances[o.name] = instances.get(o.name, 0) + 1
        for i, j in _contained(objs):
            pair = (objs[i].name, objs[j].name)
            freq[pair] = freq.get(pair, 0) + 1
            parents_with.setdefault(pair, set()).add(objs[i].ann_id)
    out = {}
    for pair, f in freq.items():
        prop = len(parents_with[pair]) / instances[pair[0]]
        if f >= MIN_FREQ and prop >= MIN_PROP:
            out[pair] = {"frequency": f, "proportion": prop}
    return out


def chains(img: LVISImage, pairs: dict) -> list[tuple[str, str, str, M.Rle, M.Rle]]:
    objs = _kept(img)
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.name] = counts.get(o.name, 0) + 1
    children: dict[int, dict[str, list[int]]] = {}
    for i, j in _contained(objs):
        if (objs[i].name, objs[j].name) in pairs and counts[objs[i].name] == 1:
            children.setdefault(i, {}).setdefault(objs[j].name, []).append(j)
    out = []
    for i in sorted(children):
        parent = objs[i]
        prle = M.from_annotation(parent.segmentation, img.h, img.w)
        for name in sorted(children[i]):
            masks = [M.decode(M.from_annotation(objs[j].segmentation, img.h, img.w)) for j in children[i][name]]
            crle = M.encode(np.any(masks, 0)[None].astype(bool))[0]
            if M.areas([prle, crle]).min() == 0:
                continue
            out.append((f"{img.key}|chain|{parent.name}:{name}", parent.name, name, prle, crle))
    return out
