import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# HierOpenImages-style box hierarchy (Wang et al., arXiv 2411.17490, Sec. 3.1 and App. A/B.2) on OpenImages
# boxes: a box b is a child of a larger box a when >= 80% of b lies inside a; a (parent class, child class) pair is
# kept when it occurs at least MIN_FREQ times and in at least MIN_PROP of the parent class's instances (OpenImages:
# 50 and 10%); IsGroupOf boxes and boxes under 1% of the image are dropped. Ground truth is boxes only, so chains
# are scored box-matched (node box vs GT box, IoU >= 0.5), as the spec does for SceneParser (I10).

CONTAIN = 0.8
MIN_FREQ = 50
MIN_PROP = 0.10
MIN_AREA_FRAC = 0.01
# a child box covering >= 90% of its parent box is the same entity labelled twice (OpenImages hypernyms such as
# person / man / human body); such pairs are dropped. For a box inside its parent the IoU equals the area ratio.
MAX_CHILD_FRAC = 0.9


@dataclass
class OIBox:
    name: str
    box: np.ndarray  # normalised x0, y0, x1, y1


def class_names(path: str | Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["LabelName"]: r["DisplayName"].lower() for r in csv.DictReader(f)}


def load_boxes(path: str | Path, names: dict[str, str], image_ids: set[str] | None = None) -> dict[str, list[OIBox]]:
    out: dict[str, list[OIBox]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if image_ids is not None and r["ImageID"] not in image_ids:
                continue
            if r.get("IsGroupOf") == "1":
                continue
            b = np.array([float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])])
            if (b[2] - b[0]) * (b[3] - b[1]) < MIN_AREA_FRAC:
                continue
            out.setdefault(r["ImageID"], []).append(OIBox(names.get(r["LabelName"], r["LabelName"]), b))
    return out


def _contained(objs: list[OIBox]) -> list[tuple[int, int]]:
    if not objs:
        return []
    b = np.stack([o.box for o in objs])
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return [(i, j) for i in range(len(objs)) for j in range(len(objs))
            if i != j and area[j] < MAX_CHILD_FRAC * area[i] and inter[i, j] >= CONTAIN * area[j]
            and objs[i].name != objs[j].name]


def class_pairs(images: dict[str, list[OIBox]]) -> dict[tuple[str, str], dict]:
    freq: dict[tuple[str, str], int] = {}
    parents_with: dict[tuple[str, str], set[tuple[str, int]]] = {}
    instances: dict[str, int] = {}
    for iid, objs in images.items():
        for o in objs:
            instances[o.name] = instances.get(o.name, 0) + 1
        for i, j in _contained(objs):
            pair = (objs[i].name, objs[j].name)
            freq[pair] = freq.get(pair, 0) + 1
            parents_with.setdefault(pair, set()).add((iid, i))
    return {pair: {"frequency": f, "proportion": len(parents_with[pair]) / instances[pair[0]]}
            for pair, f in freq.items()
            if f >= MIN_FREQ and len(parents_with[pair]) / instances[pair[0]] >= MIN_PROP}


def chains(image_id: str, objs: list[OIBox], pairs: dict, w: int, h: int):
    # parents whose class occurs once in the image; child classes with exactly one instance inside the parent
    # (boxes cannot be merged like masks); boxes returned in pixels
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.name] = counts.get(o.name, 0) + 1
    inside: dict[int, dict[str, list[int]]] = {}
    for i, j in _contained(objs):
        if (objs[i].name, objs[j].name) in pairs and counts[objs[i].name] == 1:
            inside.setdefault(i, {}).setdefault(objs[j].name, []).append(j)
    scale = np.array([w, h, w, h], float)
    out = []
    for i in sorted(inside):
        for name in sorted(inside[i]):
            if len(inside[i][name]) != 1:
                continue
            j = inside[i][name][0]
            out.append((f"openimages:{image_id}|chain|{objs[i].name}:{name}", objs[i].name, name,
                        objs[i].box * scale, objs[j].box * scale))
    return out
