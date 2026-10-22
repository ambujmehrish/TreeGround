import numpy as np
from pycocotools import mask as mu

Rle = dict


def encode(masks: np.ndarray) -> list[Rle]:
    if masks.ndim == 2:
        masks = masks[None]
    arr = np.asfortranarray(np.transpose(masks.astype(np.uint8), (1, 2, 0)))
    rles = mu.encode(arr)
    return [{"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")} for r in rles]


def _coco(rle: Rle) -> dict:
    counts = rle["counts"]
    return {"size": list(rle["size"]), "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def decode(rle: Rle) -> np.ndarray:
    return mu.decode(_coco(rle)).astype(bool)


def from_annotation(segmentation, h: int, w: int) -> Rle:
    if isinstance(segmentation, list):
        rle = mu.merge(mu.frPyObjects(segmentation, h, w))
    elif isinstance(segmentation["counts"], list):
        rle = mu.frPyObjects(segmentation, h, w)
    else:
        rle = _coco(segmentation)
    counts = rle["counts"]
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def areas(rles: list[Rle]) -> np.ndarray:
    if not rles:
        return np.zeros(0, dtype=np.int64)
    # pycocotools 2.0.11 overflows in batched area() for >= 256 masks; per-mask calls are exact
    return np.array([int(mu.area(_coco(r))) for r in rles], dtype=np.int64)


def boxes_xyxy(rles: list[Rle]) -> np.ndarray:
    if not rles:
        return np.zeros((0, 4), dtype=np.float64)
    b = mu.toBbox([_coco(r) for r in rles]).astype(np.float64)
    b[:, 2:] += b[:, :2]
    return b


def iou(a: list[Rle], b: list[Rle]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou([_coco(r) for r in a], [_coco(r) for r in b], [0] * len(b))).reshape(len(a), len(b))


def intersections(a: list[Rle], b: list[Rle]) -> np.ndarray:
    # pycocotools "crowd" IoU is |a ∩ b| / |a|; scaled back to exact integer pixel counts.
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.int64)
    ratio = np.asarray(mu.iou([_coco(r) for r in a], [_coco(r) for r in b], [1] * len(b))).reshape(len(a), len(b))
    return np.rint(ratio * areas(a)[:, None]).astype(np.int64)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
