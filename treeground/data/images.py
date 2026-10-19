import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    # Stored pixel order, no EXIF transpose: annotations of all sources refer to the stored raster.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def pixel_sha1(path: str | Path) -> tuple[str, int, int]:
    arr = load_rgb(path)
    h, w = arr.shape[:2]
    digest = hashlib.sha1()
    digest.update(f"{h}x{w}".encode())
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest(), w, h


def resolve_coco_path(coco_root: str | Path, file_name: str) -> Path:
    root = Path(coco_root)
    direct = root / file_name
    if direct.exists() or "/" in file_name:
        return direct
    for sub in ("train2017", "val2017", "train2014", "val2014"):
        p = root / sub / file_name
        if p.exists():
            return p
    return direct
