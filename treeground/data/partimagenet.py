import json
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mu

from treeground import masks as M
from treeground.run import ProtocolError

# PartImageNet_Seg (He et al., ECCV 2022; turkeyju/PartImageNet), one of PARTONOMY's three training sources
# (Partonomy-PartImageNet). COCO-style part annotations: category name '<Supercategory> <Part>' with the
# supercategory as the object class. Object masks from annotations/<split>_whole; an image without one gets part
# queries only (no object or chain query). Polygons with fewer than 3 points are dropped; images with no (valid) part
# annotation or with several object classes are skipped and counted; a listed image file that is missing is an error.

RENAME = {"tier": "tire", "aeroplane": "airplane"}   # dataset spellings


def union(rles: list) -> dict:
    r = mu.merge([M._coco(x) for x in rles])
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def clean(seg):
    # polygons with fewer than 3 points have no area (pycocotools also misreads a 4-number polygon as a box);
    # they are dropped, and an annotation left without a polygon is skipped (counted by the caller)
    if isinstance(seg, dict):
        return seg
    polys = [p for p in seg if len(p) >= 6]
    return polys or None


def _name(s: str) -> str:
    s = s.strip().lower()
    return RENAME.get(s, s)


def dataset_dir(root: str | Path) -> Path:
    hits = [p.parent for p in Path(root).rglob("annotations") if p.is_dir() and "__MACOSX" not in p.parts]
    if len(hits) != 1:
        raise ProtocolError(f"expected one PartImageNet annotations/ directory under {root}, found {hits}")
    return hits[0]


def annotation_file(root: str | Path, split: str, whole: bool = False) -> Path:
    # PartImageNet_Seg: annotations/<split>/<split>.json (parts), annotations/<split>_whole/<split>.json (objects)
    f = dataset_dir(root) / "annotations" / (f"{split}_whole" if whole else split) / f"{split}.json"
    if not f.exists():
        raise ProtocolError(f"missing {f}")
    return f


def whole_masks(root: str | Path, split: str) -> dict[str, dict]:
    # file name -> union of the image's whole-object masks
    d = json.loads(annotation_file(root, split, whole=True).read_text())
    names = {im["id"]: (im["file_name"], im["height"], im["width"]) for im in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        seg = clean(a["segmentation"]) if a.get("segmentation") else None
        if seg is not None:
            fn, h, w = names[a["image_id"]]
            per[fn].append(M.from_annotation(seg, h, w))
    return {Path(fn).name: (union(r) if len(r) > 1 else r[0]) for fn, r in per.items()}


def load(root: str | Path, split: str = "train") -> list[dict]:
    ann_path = annotation_file(root, split)
    d = json.loads(ann_path.read_text())
    cats = {}
    for c in d["categories"]:
        sup = _name(c["supercategory"])
        part = _name(c["name"][len(c["supercategory"]):] if c["name"].lower().startswith(c["supercategory"].lower())
                     else c["name"])
        cats[c["id"]] = (sup, part)
    files = {p.name: p for p in (dataset_dir(root) / "images" / split).rglob("*")
             if p.is_file() and not p.name.startswith("._")}
    objects = whole_masks(root, split)
    by_image = defaultdict(list)
    for a in d["annotations"]:
        if a.get("segmentation"):
            by_image[a["image_id"]].append(a)
    out, skipped = [], defaultdict(int)
    for im in d["images"]:
        anns = by_image.get(im["id"])
        if not anns:
            skipped["no_part_annotation"] += 1
            continue
        path = files.get(Path(im["file_name"]).name)
        if path is None:
            raise ProtocolError(f"PartImageNet {split}: image file {im['file_name']} is missing under {root}")
        h, w = im["height"], im["width"]
        parts: dict[str, list] = defaultdict(list)
        objs = set()
        for a in anns:
            sup, part = cats[a["category_id"]]
            objs.add(sup)
            seg = clean(a["segmentation"])
            if seg is None:
                skipped["degenerate_polygon_annotation"] += 1
                continue
            parts[part].append(M.from_annotation(seg, h, w))
        if len(objs) != 1:
            skipped["several_object_classes"] += 1
            continue
        if not parts:
            skipped["no_valid_part_annotation"] += 1
            continue
        out.append({"key": f"partimagenet:{im['id']}", "path": str(path), "h": h, "w": w, "object": objs.pop(),
                    "object_mask": objects.get(Path(im["file_name"]).name),
                    "parts": {p: union(r) if len(r) > 1 else r[0] for p, r in parts.items()}})
    load.skipped = dict(skipped)   # read by the caller for its counts
    return out
