import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from treeground import masks as M

# ADE20K (ADE20K_2021_17_01 full release, as exported to parquet by the Hugging Face mirror 1aurent/ADE20K).
# Each row holds the image, one mask image per annotated object (255 = visible pixels, 128 = occluded extent) and
# the object list with its part hierarchy (part_level 0 = object, 1 = part, 2 = part of a part; is_part_of = id).
# GT masks are the visible pixels. Parts of the same name under one parent are merged (PACO convention).

VISIBLE = 255


@dataclass
class ADEObject:
    id: int
    name: str
    level: int
    parent: int | None
    rle: M.Rle


@dataclass
class ADEImage:
    image_id: str
    h: int
    w: int
    objects: list[ADEObject]

    @property
    def key(self) -> str:
        return f"ade20k:{self.image_id}"


def short_name(name: str) -> str:
    return name.split(",")[0].strip()


def row_groups(parquet: str | Path) -> list[tuple[int, int]]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet))
    return [(g, pf.metadata.row_group(g).num_rows) for g in range(pf.num_row_groups)]


def iter_rows(parquet: str | Path, indices: set[int] | None = None):
    # yields (row index, image array, ADEImage) in file order; indices restricts to a subset (sharding)
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(str(parquet))
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        wanted = [i for i in range(start, start + n) if indices is None or i in indices]
        if wanted:
            t = pf.read_row_group(g, columns=["image", "instances", "objects", "filename"])
            for i in wanted:
                r = i - start
                with Image.open(io.BytesIO(t.column("image")[r].as_py()["bytes"])) as im:
                    image = np.asarray(im.convert("RGB"))
                h, w = image.shape[:2]
                objs = []
                for o, inst in zip(t.column("objects")[r].as_py(), t.column("instances")[r].as_py()):
                    with Image.open(io.BytesIO(inst["bytes"])) as im:
                        m = np.asarray(im) == VISIBLE
                    if m.shape != (h, w):
                        raise ValueError(f"row {i}: instance mask size {m.shape} differs from the image {(h, w)}")
                    objs.append(ADEObject(int(o["id"]), short_name(o["name"]), int(o["parts"]["part_level"]),
                                          o["parts"]["is_part_of"], M.encode(m[None])[0]))
                name = t.column("filename")[r].as_py().rsplit(".", 1)[0]
                yield i, image, ADEImage(name, h, w, objs)
        start += n


def _merged_children(img: ADEImage, parent: int, level: int) -> dict[str, M.Rle]:
    groups: dict[str, list[M.Rle]] = {}
    for o in img.objects:
        if o.level == level and o.parent == parent:
            groups.setdefault(o.name, []).append(o.rle)
    out = {}
    for name, rles in sorted(groups.items()):
        rle = rles[0] if len(rles) == 1 else M.encode(np.any([M.decode(r) for r in rles], 0)[None])[0]
        if M.areas([rle])[0] > 0:
            out[name] = rle
    return out


def chains(img: ADEImage) -> list[tuple[str, str, str, M.Rle, M.Rle]]:
    # object -> part chains for objects whose name occurs once at level 0, and part -> subpart chains for parts
    # whose name occurs once under their object; the first step names the part with its object
    counts: dict[str, int] = {}
    for o in img.objects:
        if o.level == 0:
            counts[o.name] = counts.get(o.name, 0) + 1
    out = []
    for o in img.objects:
        if o.level != 0 or counts[o.name] != 1 or M.areas([o.rle])[0] == 0:
            continue
        for part, rle in _merged_children(img, o.id, 1).items():
            out.append((f"{img.key}|chain|{o.name}:{part}", o.name, part, o.rle, rle))
        sibling_names: dict[str, int] = {}
        for p in img.objects:
            if p.level == 1 and p.parent == o.id:
                sibling_names[p.name] = sibling_names.get(p.name, 0) + 1
        for p in img.objects:
            if p.level != 1 or p.parent != o.id or sibling_names[p.name] != 1 or M.areas([p.rle])[0] == 0:
                continue
            for sub, rle in _merged_children(img, p.id, 2).items():
                out.append((f"{img.key}|chain|{o.name}:{p.name}:{sub}", f"{p.name} of the {o.name}", sub,
                            p.rle, rle))
    return out


def all_parts(img: ADEImage) -> list[tuple[str, M.Rle]]:
    names = {o.id: o.name for o in img.objects}
    return [(f"{names.get(o.parent, '?')}:{o.name}", o.rle) for o in img.objects
            if o.level >= 1 and M.areas([o.rle])[0] > 0]
