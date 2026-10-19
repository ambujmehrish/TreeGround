import pandas as pd
import pytest

from treeground.data import manifest as MF
from treeground.run import ProtocolError


def frame(rows):
    df = pd.DataFrame(rows, columns=["image_key", "source", "split", "split_class", "sha1"])
    for c in ("coco_id", "path", "w", "h"):
        df[c] = 0
    return df


def test_precedence_keeps_highest_split_class():
    df = frame([
        ["paco_lvis:1", "paco_lvis", "train", "train", "a"],
        ["grefcoco:1", "grefcoco", "val", "dev", "a"],
        ["paco_lvis:2", "paco_lvis", "val", "dev", "b"],
        ["grefcoco:2", "grefcoco", "testA", "eval", "b"],
        ["paco_lvis:3", "paco_lvis", "val", "dev", "c"],
        ["grefcoco:3", "grefcoco", "val", "dev", "c"],
    ])
    kept, removed = MF.apply_precedence(df)
    assert set(kept.image_key) == {"grefcoco:1", "grefcoco:2", "paco_lvis:3", "grefcoco:3"}
    assert set(removed.image_key) == {"paco_lvis:1", "paco_lvis:2"}
    assert dict(zip(removed.image_key, removed.kept_class)) == {"paco_lvis:1": "dev", "paco_lvis:2": "eval"}
    MF.check_disjoint(kept)
    with pytest.raises(ProtocolError):
        MF.check_disjoint(df)


def test_manifest_hash_is_order_invariant():
    df = frame([["k1", "s", "train", "train", "a"], ["k2", "s", "val", "dev", "b"]])
    assert MF.manifest_hash(df) == MF.manifest_hash(df.iloc[::-1])
    changed = df.copy()
    changed.loc[0, "sha1"] = "z"
    assert MF.manifest_hash(df) != MF.manifest_hash(changed)


def test_seeded_choice_is_deterministic():
    items = list(range(100))
    assert MF.seeded_choice(items, 10, 0) == MF.seeded_choice(items[::-1], 10, 0)
    assert MF.seeded_choice(items, 200, 0) == items
