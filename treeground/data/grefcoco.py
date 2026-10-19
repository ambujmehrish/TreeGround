import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "testA", "testB")


@dataclass
class GrefQuery:
    sent_id: int
    ref_id: int
    image_id: int
    file_name: str
    text: str
    split: str

    @property
    def image_key(self) -> str:
        return f"grefcoco:{self.image_id}"

    @property
    def qid(self) -> str:
        return f"grefcoco|{self.split}|{self.sent_id}"


def _refs_path(root: Path) -> Path:
    p = root / "grefs(unc).json"
    if not p.exists():
        raise FileNotFoundError(f"gRefCOCO refs not found: {p}")
    return p


def load(root: str | Path) -> tuple[list[dict], dict[int, dict]]:
    root = Path(root)
    refs = json.loads(_refs_path(root).read_text())
    inst = json.loads((root / "instances.json").read_text())
    return refs, {im["id"]: im for im in inst["images"]}


def is_no_target(ref: dict) -> bool:
    ann = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
    flag = ref.get("no_target")
    empty = all(a == -1 for a in ann)
    if flag is not None and bool(flag) != empty:
        raise ValueError(f"gRefCOCO ref {ref['ref_id']}: no_target flag disagrees with ann_id {ann}")
    return empty


def no_target_queries(refs: list[dict], images: dict[int, dict], split: str) -> list[GrefQuery]:
    out = []
    for ref in refs:
        if ref["split"] != split or not is_no_target(ref):
            continue
        im = images[ref["image_id"]]
        for s in ref["sentences"]:
            out.append(GrefQuery(s["sent_id"], ref["ref_id"], ref["image_id"], im["file_name"], s["sent"], split))
    return sorted(out, key=lambda q: q.sent_id)


def image_splits(refs: list[dict], images: dict[int, dict]) -> list[tuple[int, str, str]]:
    seen = {}
    for ref in refs:
        key = (ref["image_id"], ref["split"])
        seen[key] = images[ref["image_id"]]["file_name"]
    return [(img, split, fn) for (img, split), fn in sorted(seen.items())]


def image_path(coco_root: str | Path, file_name: str) -> Path:
    return Path(coco_root) / "train2014" / file_name
