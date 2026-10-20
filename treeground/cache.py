import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from treeground.run import REPO, ProtocolError, canonical_hash, read_json, write_json

# Modules whose code determines cached values. Entry scripts (I/O, logging, sharding) are excluded so that
# cosmetic edits do not invalidate caches.
PRODUCERS = {
    "proposals": ["treeground/proposals.py", "treeground/masks.py"],
    "features": ["treeground/proposals.py", "treeground/masks.py", "treeground/mllm.py", "treeground/features.py"],
    "queries": ["treeground/mllm.py", "treeground/data/paco.py", "treeground/data/grefcoco.py"],
}


def code_version(kind: str) -> str:
    h = hashlib.sha256()
    for rel in PRODUCERS[kind]:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()


def cache_dir(cfg: dict, kind: str, identity: dict) -> Path:
    sub = "smoke" if identity["smoke"] else "full"
    return Path(cfg["cache_root"]) / sub / kind / canonical_hash(identity)[:16]


def write_identity(path: Path, identity: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    target = path / "identity.json"
    if target.exists():
        check_identity(path, identity)
        return
    write_json(target, {"identity": identity, "hash": canonical_hash(identity)})


def check_identity(path: Path, expected: dict) -> None:
    target = path / "identity.json"
    if not target.exists():
        raise ProtocolError(f"cache {path} has no identity.json")
    stored = read_json(target)
    if stored["hash"] != canonical_hash(expected) or stored["identity"] != json.loads(json.dumps(expected)):
        diff = {k: (stored["identity"].get(k), expected.get(k)) for k in set(stored["identity"]) | set(expected)
                if stored["identity"].get(k) != expected.get(k)}
        raise ProtocolError(f"cache identity mismatch at {path}: {diff}")


def mark_complete(path: Path, summary: dict) -> None:
    write_json(path / "complete.json", summary)


def require_complete(path: Path) -> dict:
    if not (path / "complete.json").exists():
        raise ProtocolError(f"cache {path} is not complete (run the merge step)")
    return read_json(path / "complete.json")


def save_tensors(path: Path, tensors: dict[str, np.ndarray | torch.Tensor], meta: dict[str, str]) -> None:
    out = {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v.contiguous().cpu())
           for k, v in tensors.items()}
    tmp = path.with_suffix(".tmp")
    save_file(out, str(tmp), metadata=meta)
    tmp.replace(path)


def load_tensors(path: Path, keys: list[str] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        names = keys if keys is not None else list(f.keys())
        return {k: f.get_tensor(k) for k in names}, meta


class ProposalStore:
    def __init__(self, path: Path, identity: dict, manifest: pd.DataFrame):
        check_identity(path, identity)
        require_complete(path)
        self.path = path
        self.index = pd.read_parquet(path / "index.parquet").set_index("sha1")
        self._shards: dict[str, dict[str, pd.DataFrame]] = {}

    def has(self, sha1: str) -> bool:
        return sha1 in self.index.index

    def rows(self, sha1: str) -> pd.DataFrame:
        if sha1 not in self.index.index:
            raise ProtocolError(f"image {sha1} has no proposal entry in {self.path}")
        shard = self.index.loc[sha1, "shard"]
        if shard not in self._shards:
            df = pd.read_parquet(self.path / "shards" / shard)
            self._shards[shard] = {k: g.sort_values("src_index") for k, g in df.groupby("sha1")}
        empty = pd.DataFrame(columns=["src_index", "rle", "area", "box", "stability", "sam_iou"])
        return self._shards[shard].get(sha1, empty)

    def nodes(self, sha1: str, tau: float) -> dict:
        df = self.rows(sha1)
        if len(df):
            df = df[df[keep_column(tau)].astype(bool)]
        h, w = int(self.index.loc[sha1, "h"]), int(self.index.loc[sha1, "w"])
        return {
            "h": h, "w": w,
            "rles": [{"size": [h, w], "counts": c} for c in df["rle"]],
            "areas": df["area"].to_numpy(np.int64),
            "boxes": np.stack(df["box"].to_numpy()) if len(df) else np.zeros((0, 4)),
            "src_index": df["src_index"].to_numpy(np.int64),
            "stability": df["stability"].to_numpy(np.float64),
        }


def keep_column(tau: float) -> str:
    return f"keep_{int(round(tau * 100)):03d}"


class FeatureStore:
    def __init__(self, path: Path, identity: dict, manifest: pd.DataFrame):
        check_identity(path, identity)
        require_complete(path)
        self.path = path
        self.identity_hash = canonical_hash(identity)
        self.sha1s = set(manifest["sha1"])

    def file(self, sha1: str) -> Path:
        return self.path / "images" / f"{sha1}.safetensors"

    def load(self, sha1: str, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
        if sha1 not in self.sha1s:
            raise ProtocolError(f"image {sha1} is not in the split manifest")
        f = self.file(sha1)
        if not f.exists():
            raise ProtocolError(f"missing feature cache entry {f}")
        tensors, meta = load_tensors(f, keys)
        if meta.get("image_sha1") != sha1 or meta.get("identity") != self.identity_hash:
            raise ProtocolError(f"feature cache entry {f} does not match its image or cache identity")
        return tensors
