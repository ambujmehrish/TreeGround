import json
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mu

from treeground import masks as M
from treeground.run import ProtocolError

# PartImageNet_Seg (He et al., ECCV 2022; turkeyju/PartImageNet), one of PARTONOMY's three training sources
# (Partonomy-PartImageNet). COCO-style part annotations: category name '<Supercategory> <Part>' with the
# supercategory as the object class. The object mask of an image is the union of its part masks (PartImageNet
# annotates one object class per image and has no object masks).

RENAME = {"tier": "tire", "aeroplane": "airplane"}   # dataset spellings


def union(rles: list) -> dict:
    r = mu.merge([M._coco(x) for x in rles])
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def _name(s: str) -> str:
    s = s.strip().lower()
    return RENAME.get(s, s)


def annotation_file(root: str | Path, split: str) -> Path:
    hits = [p for p in Path(root).rglob(f"{split}.json") if "annotations" in p.parts]
    if len(hits) != 1:
        raise ProtocolError(f"expected one annotations/{split}.json under {root}, found {hits}")
    return hits[0]


def load(root: str | Path, split: str = "train") -> list[dict]:
    ann_path = annotation_file(root, split)
    d = json.loads(ann_path.read_text())
    cats = {}
    for c in d["categories"]:
        sup = _name(c["supercategory"])
        part = _name(c["name"][len(c["supercategory"]):] if c["name"].lower().startswith(c["supercategory"].lower())
                     else c["name"])
        cats[c["id"]] = (sup, part)
    files = {p.name: p for p in (ann_path.parents[1] / "images" / split).rglob("*") if p.is_file()} \
        if (ann_path.parents[1] / "images" / split).exists() else \
        {p.name: p for p in Path(root).rglob("*.JPEG")}
    by_image = defaultdict(list)
    for a in d["annotations"]:
        if a.get("segmentation"):
            by_image[a["image_id"]].append(a)
    out = []
    for im in d["images"]:
        anns = by_image.get(im["id"])
        path = files.get(Path(im["file_name"]).name)
        if not anns or path is None:
            continue
        h, w = im["height"], im["width"]
        parts: dict[str, list] = defaultdict(list)
        objs = set()
        for a in anns:
            sup, part = cats[a["category_id"]]
            objs.add(sup)
            parts[part].append(M.from_annotation(a["segmentation"], h, w))
        if len(objs) != 1:
            continue
        out.append({"key": f"partimagenet:{im['id']}", "path": str(path), "h": h, "w": w, "object": objs.pop(),
                    "parts": {p: union(r) if len(r) > 1 else r[0] for p, r in parts.items()}})
    return out
