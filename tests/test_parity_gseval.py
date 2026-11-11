import types

import numpy as np

from test_s2 import load_script
from treeground import masks as M

H, W = 10, 12


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


class FakeSam3Union:
    official = "official_threshold"

    def __init__(self, cfg):
        pass

    def instances(self, image, texts):
        # per text: (score, mask), highest first; 0.9 and 0.6 pass the official 0.5, 0.3 does not
        return [[(0.9, box(0, 5, 0, 6)), (0.6, box(5, 10, 6, 12)), (0.3, box(0, 10, 0, 12))], [(0.4, box(0, 2, 0, 2))]]


def test_gseval_parity_baseline_returns_the_official_union(monkeypatch):
    pe = load_script("parity_eval")
    bp = load_script("b_parity")
    monkeypatch.setitem(bp.PARITY, "sam3_union", FakeSam3Union)
    monkeypatch.setitem(__import__("sys").modules, "b_parity", bp)
    mod = types.SimpleNamespace(FILES=("scripts/gseval_eval.py",), identity=lambda cfg, method: {"method": method})
    cfg = {"parity": {"sam3_union": {"floor": 0.05, "official_threshold": 0.5, "max_instances": 32}}}
    pe.install(mod, "gseval_eval", cfg, "sam3_union")
    assert mod.FILES[-2:] == ("scripts/b_parity.py", "scripts/parity_eval.py")
    assert mod.identity(cfg, "sam3_union")["parity_method"] == "sam3_union"
    hit, miss = mod.Baseline(cfg, "sam3_union")(np.zeros((H, W, 3), np.uint8), ["a", "b"])
    (score, rle), = hit["pred"]
    assert score == 1.0 and (M.decode(rle) == (box(0, 5, 0, 6) | box(5, 10, 6, 12))).all()
    assert miss["pred"] == []                                 # nothing above 0.5: abstain, as the D45 baseline rows
