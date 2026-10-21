import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s0_proposals as S  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.cache import ProposalStore  # noqa: E402


def _frame(sha1, n, h=8, w=8):
    rles = M.encode(np.stack([np.eye(h, w, k, dtype=bool) for k in range(n)]))
    return pd.DataFrame({"sha1": sha1, "src_index": np.arange(n)[::-1], "rle": [r["counts"] for r in rles],
                         "h": h, "w": w, "area": M.areas(rles), "box": list(M.boxes_xyxy(rles)),
                         "sam_iou": 0.5, "stability": 0.9, "keep_090": True})


def test_row_group_per_image_roundtrip(tmp_path, monkeypatch):
    df = pd.concat([_frame("a", 3), _frame("b", 2), _frame("d", 4)], ignore_index=True)
    shard = tmp_path / "shards" / "00000.parquet"
    shard.parent.mkdir()
    S.write_by_image(df, shard)
    groups = S.row_groups(shard)
    assert groups == {"a": 0, "b": 1, "d": 2}
    index = pd.DataFrame({"sha1": ["a", "b", "c", "d"], "shard": "00000.parquet", "h": 8, "w": 8,
                          "row_group": [groups.get(s, -1) for s in "abcd"]})
    index.to_parquet(tmp_path / "index.parquet", index=False)
    monkeypatch.setattr("treeground.cache.check_identity", lambda *a: None)
    monkeypatch.setattr("treeground.cache.require_complete", lambda *a: {})
    store = ProposalStore(tmp_path, {}, None)
    for sha1, n in (("d", 4), ("a", 3), ("c", 0), ("b", 2)):
        nodes = store.nodes(sha1, 0.9)
        assert len(nodes["rles"]) == n
        assert list(nodes["src_index"]) == sorted(nodes["src_index"])


def test_old_single_group_shards_fall_back(tmp_path):
    df = pd.concat([_frame("a", 3), _frame("b", 2)], ignore_index=True)
    path = tmp_path / "x.parquet"
    df.to_parquet(path, index=False)
    assert S.row_groups(path) == {}


def test_node_cap_keeps_top_k_by_score(tmp_path, monkeypatch):
    df = _frame("a", 5)
    df["stability"] = [0.1, 0.9, 0.5, 0.9, 0.3]
    shard = tmp_path / "shards" / "00000.parquet"
    shard.parent.mkdir()
    S.write_by_image(df, shard)
    pd.DataFrame({"sha1": ["a"], "shard": "00000.parquet", "h": 8, "w": 8, "row_group": [0]}).to_parquet(
        tmp_path / "index.parquet", index=False)
    monkeypatch.setattr("treeground.cache.check_identity", lambda *a: None)
    monkeypatch.setattr("treeground.cache.require_complete", lambda *a: {})
    store = ProposalStore(tmp_path, {}, None, {"k": 3, "order": "stability"})
    kept = store.nodes("a", 0.9)
    # src_index runs 4..0 down the frame; stabilities 0.9 at src 3 and 1, then 0.5 at src 2
    assert list(kept["src_index"]) == [1, 2, 3]
    assert len(store.nodes("a", None)["rles"]) == 5
