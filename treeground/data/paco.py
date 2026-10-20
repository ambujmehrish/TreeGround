import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from treeground import masks as M
from treeground.data.images import resolve_coco_path

SPLITS = {"train": "paco_lvis_v1_train.json", "val": "paco_lvis_v1_val.json", "test": "paco_lvis_v1_test.json"}

OBJECT_TEMPLATE = "Locate the {object}. <SEL>"
PART_TEMPLATE = "Locate the {part} of the {object}. <SEL>"
CHAIN_TEMPLATE = "Locate the {object} <SEL> and then its {part} <SEL>."
TEXT_TEMPLATE = "Find the {part} of a {object}. <SEL>"
TEMPLATES = {"object": OBJECT_TEMPLATE, "part": PART_TEMPLATE, "chain": CHAIN_TEMPLATE, "text": TEXT_TEMPLATE}


@dataclass
class Obj:
    ann_id: int
    name: str
    rle: M.Rle


@dataclass
class Part:
    ann_id: int
    obj_ann_id: int
    obj_name: str
    part_name: str
    rle: M.Rle

    @property
    def label(self) -> str:
        return f"{self.obj_name}:{self.part_name}"


@dataclass
class PacoImage:
    image_id: int
    file_name: str
    h: int
    w: int
    objects: list[Obj] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    not_exhaustive: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"paco_lvis:{self.image_id}"


@dataclass
class Query:
    qid: str
    kind: str
    text: str
    gt: tuple[int, ...]   # annotation id per probe (object, part, or object then part)
    label: str | None = None

    @property
    def n_sel(self) -> int:
        return len(self.gt) if self.kind != "text" else 1


def display(name: str) -> str:
    return name.replace("_", " ")


def ann_path(root: str | Path, split: str) -> Path:
    return Path(root) / SPLITS[split]


def image_list(root: str | Path, split: str) -> list[dict]:
    return json.loads(ann_path(root, split).read_text())["images"]


def load(root: str | Path, split: str, image_ids: set[int] | None = None) -> dict[int, PacoImage]:
    # image_ids restricts mask decoding to the images a stage actually uses
    data = json.loads(ann_path(root, split).read_text())
    cats = {c["id"]: c for c in data["categories"]}
    images = {im["id"]: PacoImage(im["id"], im["file_name"], im["height"], im["width"],
                                  not_exhaustive={cats[c]["name"] for c in im.get("not_exhaustive_category_ids", [])
                                                  if c in cats})
              for im in data["images"] if image_ids is None or im["id"] in image_ids}
    obj_names: dict[int, str] = {}
    pending_parts = []
    for ann in data["annotations"]:
        if ann["image_id"] not in images:
            continue
        cat = cats[ann["category_id"]]
        img = images[ann["image_id"]]
        rle = M.from_annotation(ann["segmentation"], img.h, img.w)
        if cat["supercategory"] == "OBJECT":
            img.objects.append(Obj(ann["id"], cat["name"], rle))
            obj_names[ann["id"]] = cat["name"]
        elif cat["supercategory"] == "PART":
            pending_parts.append((ann, cat, rle))
        else:
            raise ValueError(f"unknown PACO supercategory {cat['supercategory']!r}")
    for ann, cat, rle in pending_parts:
        obj_name, part_name = cat["name"].split(":", 1)
        if obj_names.get(ann["obj_ann_id"]) != obj_name:
            raise ValueError(f"part annotation {ann['id']} does not match its object annotation {ann['obj_ann_id']}")
        images[ann["image_id"]].parts.append(Part(ann["id"], ann["obj_ann_id"], obj_name, part_name, rle))
    return images


def image_path(coco_root: str | Path, img: PacoImage) -> Path:
    return resolve_coco_path(coco_root, img.file_name)


def i1_queries(img: PacoImage) -> tuple[list[Query], dict]:
    by_name: dict[str, list[Obj]] = defaultdict(list)
    for o in img.objects:
        by_name[o.name].append(o)
    parts_of: dict[int, dict[str, list[Part]]] = defaultdict(lambda: defaultdict(list))
    for p in img.parts:
        parts_of[p.obj_ann_id][p.part_name].append(p)
    queries: list[Query] = []
    kept_parts = 0
    for name in sorted(by_name):
        insts = by_name[name]
        if len(insts) != 1:
            continue
        o = insts[0]
        queries.append(Query(f"{img.key}|object|{name}", "object",
                             OBJECT_TEMPLATE.format(object=display(name)), (o.ann_id,)))
        for part_name in sorted(parts_of[o.ann_id]):
            masks = parts_of[o.ann_id][part_name]
            if len(masks) != 1:
                continue
            p = masks[0]
            kept_parts += 1
            fmt = {"object": display(name), "part": display(part_name)}
            queries.append(Query(f"{img.key}|part|{p.label}", "part", PART_TEMPLATE.format(**fmt), (p.ann_id,)))
            queries.append(Query(f"{img.key}|chain|{p.label}", "chain", CHAIN_TEMPLATE.format(**fmt),
                                 (o.ann_id, p.ann_id)))
    stats = {
        "objects": len(img.objects),
        "objects_kept": sum(1 for q in queries if q.kind == "object"),
        "parts": len(img.parts),
        "parts_kept": kept_parts,
    }
    return queries, stats


def text_query(label: str) -> Query:
    obj, part = label.split(":", 1)
    return Query(f"text|{label}", "text", TEXT_TEMPLATE.format(object=display(obj), part=display(part)), (), label)


def gt_rles(img: PacoImage) -> dict[int, M.Rle]:
    out = {o.ann_id: o.rle for o in img.objects}
    out.update({p.ann_id: p.rle for p in img.parts})
    return out
