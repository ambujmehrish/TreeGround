import json
from dataclasses import dataclass, field
from pathlib import Path

from treeground import masks as M

QA_FILE = "test-qa_pairs.json"


@dataclass
class PartonomyImage:
    rel_path: str
    label: str
    roots: list[M.Rle] = field(default_factory=list)
    parts: dict[str, list[M.Rle]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"partonomy:{Path(self.rel_path).stem}"


def available(root: str | Path) -> bool:
    return (Path(root) / QA_FILE).exists()


def load(root: str | Path) -> dict[str, PartonomyImage]:
    records = json.loads((Path(root) / QA_FILE).read_text())
    images: dict[str, PartonomyImage] = {}
    seen: dict[str, set] = {}
    for r in records:
        img = images.setdefault(r["image_path"], PartonomyImage(r["image_path"], r["image_label"]))
        keys = seen.setdefault(r["image_path"], set())
        for name, segs in r.get("segmentations", {}).items():
            for s in segs:
                rle = {"size": list(s["size"]), "counts": s["counts"]}
                k = (name, s["counts"])
                if k in keys:
                    continue
                keys.add(k)
                if s.get("is_root_concept", False):
                    img.roots.append(rle)
                else:
                    img.parts.setdefault(name, []).append(rle)
    return images


def image_path(root: str | Path, img: PartonomyImage) -> Path:
    return Path(root) / img.rel_path


def chains(img: PartonomyImage) -> list[tuple[str, M.Rle, M.Rle]]:
    if len(img.roots) != 1:
        return []
    return [(name, img.roots[0], rle) for name in sorted(img.parts) for rle in img.parts[name]]
