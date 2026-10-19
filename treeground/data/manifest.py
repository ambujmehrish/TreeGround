import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from treeground.data import grefcoco, paco, partonomy, sceneparser
from treeground.data.images import pixel_sha1
from treeground.run import ProtocolError

SPLIT_CLASS = {
    ("paco_lvis", "train"): "train", ("paco_lvis", "val"): "dev", ("paco_lvis", "test"): "eval",
    ("grefcoco", "train"): "train", ("grefcoco", "val"): "dev",
    ("grefcoco", "testA"): "eval", ("grefcoco", "testB"): "eval",
    ("partonomy", "test"): "eval",
    ("sceneparser", "train"): "train", ("sceneparser", "val"): "dev",
}
PRECEDENCE = {"heldout": 3, "eval": 2, "dev": 1, "train": 0}
COLUMNS = ["image_key", "source", "split", "split_class", "coco_id", "sha1", "path", "w", "h"]


def paco_entries(cfg: dict, splits=("train", "val", "test")) -> list[dict]:
    rows = []
    for split in splits:
        for img in paco.load(cfg["data"]["paco_lvis"], split).values():
            rows.append({"image_key": img.key, "source": "paco_lvis", "split": split, "coco_id": img.image_id,
                         "path": str(paco.image_path(cfg["data"]["coco_images"], img))})
    return rows


def grefcoco_entries(cfg: dict) -> list[dict]:
    refs, images = grefcoco.load(cfg["data"]["grefcoco"])
    return [{"image_key": f"grefcoco:{img}", "source": "grefcoco", "split": split, "coco_id": img,
             "path": str(grefcoco.image_path(cfg["data"]["coco_images"], fn))}
            for img, split, fn in grefcoco.image_splits(refs, images)]


def partonomy_entries(cfg: dict) -> list[dict]:
    root = cfg["data"]["partonomy"]
    return [{"image_key": img.key, "source": "partonomy", "split": "test", "coco_id": None,
             "path": str(partonomy.image_path(root, img))} for img in partonomy.load(root).values()]


def sceneparser_entries(cfg: dict) -> list[dict]:
    rows = []
    for split in sceneparser.SPLITS:
        if not sceneparser.available(cfg["data"]["sceneparser"], split):
            continue
        for img in sceneparser.iter_split(cfg["data"]["sceneparser"], split):
            rows.append({"image_key": img.key, "source": "sceneparser", "split": split, "coco_id": None,
                         "path": str(sceneparser.image_path(cfg["data"]["sceneparser_images"], img))})
    return rows


def source_available(cfg: dict, source: str) -> bool:
    d = cfg["data"]
    if source == "paco_lvis":
        return all(paco.ann_path(d["paco_lvis"], s).exists() for s in paco.SPLITS)
    if source == "grefcoco":
        return (Path(d["grefcoco"]) / "grefs(unc).json").exists() and (Path(d["grefcoco"]) / "instances.json").exists()
    if source == "partonomy":
        return partonomy.available(d["partonomy"]) and (Path(d["partonomy"]) / "images").is_dir()
    if source == "sceneparser":
        images = Path(d["sceneparser_images"]) / "datasets" / "EgoObjects" / "images"
        return any(sceneparser.available(d["sceneparser"], s) for s in sceneparser.SPLITS) and images.is_dir()
    raise ValueError(source)


ENTRY_BUILDERS = {"paco_lvis": paco_entries, "grefcoco": grefcoco_entries, "partonomy": partonomy_entries,
                  "sceneparser": sceneparser_entries}


def _hash_one(path: str) -> tuple[str, int, int]:
    if not Path(path).exists():
        raise ProtocolError(f"missing image file: {path}")
    return pixel_sha1(path)


def hash_images(paths: list[str], workers: int) -> list[tuple[str, int, int]]:
    unique = sorted(set(paths))
    with Pool(workers) as pool:
        results = dict(zip(unique, pool.map(_hash_one, unique, chunksize=64)))
    return [results[p] for p in paths]


def apply_precedence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = df["split_class"].map(PRECEDENCE)
    top = rank.groupby(df["sha1"]).transform("max")
    removed = df[rank < top].copy()
    removed["kept_class"] = removed["sha1"].map(df.loc[rank == top].groupby("sha1")["split_class"].first())
    return df[rank == top].reset_index(drop=True), removed.reset_index(drop=True)


def check_disjoint(df: pd.DataFrame) -> None:
    classes = df.groupby("sha1")["split_class"].nunique()
    bad = classes[classes > 1]
    if len(bad):
        raise ProtocolError(f"{len(bad)} images appear in more than one split class after precedence removal")


def manifest_hash(df: pd.DataFrame) -> str:
    canon = df[COLUMNS].sort_values(["image_key", "split"]).astype(str).to_csv(index=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def load_manifest(run_dir: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(run_dir / "manifest.parquet")
    stored = (run_dir / "manifest.sha256").read_text().strip()
    if manifest_hash(df) != stored:
        raise ProtocolError(f"manifest at {run_dir} does not match its recorded hash")
    return df, stored


def seeded_choice(items: list, k: int, seed: int) -> list:
    items = sorted(items)
    if k >= len(items):
        return items
    rng = np.random.default_rng(seed)
    return sorted(items[i] for i in rng.choice(len(items), size=k, replace=False))
