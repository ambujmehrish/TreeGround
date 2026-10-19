import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PLACEHOLDER_ACTION = "__placeholder_action__"
SPLITS = ("train", "val")


@dataclass
class SceneParserImage:
    image_path: str
    objects: list[dict]

    @property
    def key(self) -> str:
        return f"sceneparser:{Path(self.image_path).stem}"


def split_file(root: str | Path, split: str) -> Path:
    return Path(root) / f"{split}.jsonl"


def available(root: str | Path, split: str) -> bool:
    return split_file(root, split).exists()


def iter_split(root: str | Path, split: str) -> Iterator[SceneParserImage]:
    with open(split_file(root, split)) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                yield SceneParserImage(r["image_path"], r.get("objects", []))


def image_path(images_root: str | Path, img: SceneParserImage) -> Path:
    return Path(images_root) / img.image_path


def chains(img: SceneParserImage) -> list[dict]:
    out = []
    for o in img.objects:
        for p in o.get("parts", []):
            if p.get("is_pseudo_part", False):
                continue
            out.append({"names": (o["phrase"], p["part_name"]), "boxes": (o["object_bbox"], p["part_bbox"])})
            for a in p.get("affordances", []):
                if a.get("action") == PLACEHOLDER_ACTION or "affordance_bbox" not in a:
                    continue
                out.append({"names": (o["phrase"], p["part_name"], a["action"]),
                            "boxes": (o["object_bbox"], p["part_bbox"], a["affordance_bbox"])})
    return out
