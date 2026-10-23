import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_multi_target_positives_and_ignore():
    C = load_script("s1_cache")
    ious = np.array([[0.1, 0.7, 0.6, 0.0], [0.0, 0.2, 0.3, 0.9]])
    t = C.multi_target(ious)
    assert list(t.pos) == [1, 3] and list(t.ignore) == [2]
    assert C.multi_target(np.array([[0.1, 0.2]])) is None


def test_load_records_round_trip(tmp_path):
    S1 = load_script("s1_train")
    V2 = load_script("v2_train_eval")
    d = tmp_path / "full" / "s1" / "paco_train" / "abc"
    (d / "shards").mkdir(parents=True)
    (d / "identity.json").write_text(json.dumps({"identity": {}, "hash": "h"}))
    n = 4
    rec = {"key": "k", "z": torch.randn(n, 8).half(), "parent": np.array([4, 0, 0, 4, -1], np.int32),
           "labels": np.array(["mug:handle", "", "", ""], dtype=object), "parts": np.array(["handle", "", "", ""], dtype=object),
           "high": np.zeros(n, bool),
           "queries": [{"qid": "q1", "kind": "chain", "probes": torch.randn(2, 8).half(),
                        "steps": [(np.array([0], np.int32), np.array([], np.int32)), (np.array([1], np.int32), np.array([2], np.int32))],
                        "path": (0, 1)}],
           "absent": [{"qid": "a1", "probes": torch.randn(1, 8).half()}]}
    torch.save({"records": [rec], "counts": {"queries": 1}, "sel_init": "x", "layer": 12}, d / "shards" / "00000.pt")
    cfg = {"cache_root": str(tmp_path), "s1": {"sources": ["paco_train"], "num_shards": {"paco_train": 1}}}
    recs, absent, counts, used = S1.load_records(cfg, V2, 12)
    assert len(recs) == 1 and len(absent) == 1 and counts == {"paco_train:queries": 1}
    q = recs[0].queries[0]
    assert q.kind == "chain" and q.path == (0, 1) and list(q.steps[1].ignore) == [2]
    assert list(recs[0].tree.strict_descendants(0)) == [1, 2]
    assert absent[0].absent and absent[0].queries[0].steps[0].pos.size == 0
