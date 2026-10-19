from dataclasses import dataclass

import numpy as np

from treeground import masks as M
from treeground.data.paco import PacoImage

IOU_THR = 0.5


@dataclass
class NodeLabels:
    label: np.ndarray        # "object:part" for the best match of some GT part with IoU >= 0.5, else ""
    part: np.ndarray         # part name of that label, else ""
    high_overlap: np.ndarray # IoU >= 0.5 with some GT part without being its best match (and unlabelled)
    part_iou: np.ndarray     # nodes x GT parts


def paco_part_labels(node_rles: list[M.Rle], img: PacoImage) -> NodeLabels:
    n = len(node_rles)
    ious = M.iou(node_rles, [p.rle for p in img.parts])
    label = np.full(n, "", dtype=object)
    part = np.full(n, "", dtype=object)
    best_iou = np.zeros(n)
    high = np.zeros(n, dtype=bool)
    for k, p in enumerate(img.parts):
        if n == 0:
            break
        col = ious[:, k]
        b = int(np.argmax(col))
        if col[b] < IOU_THR:
            continue
        if col[b] > best_iou[b]:
            label[b], part[b], best_iou[b] = p.label, p.part_name, col[b]
        high |= col >= IOU_THR
    high &= label == ""
    return NodeLabels(label, part, high, ious)
