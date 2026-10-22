import numpy as np
import pytest

from treeground import masks as M
from treeground.data import ade20k, pascal_part


def test_pascal_part_names_merge_sides_and_numbers():
    assert pascal_part.part_name("lwing") == "wing"
    assert pascal_part.part_name("rwing") == "wing"
    assert pascal_part.part_name("wheel_3") == "wheel"
    assert pascal_part.part_name("llleg") == "lower leg"     # person: left lower leg
    assert pascal_part.part_name("lleg") == "leg"            # bird: left leg
    assert pascal_part.part_name("rfho") == "front hoof"
    assert pascal_part.part_name("leftmirror") == "mirror"
    with pytest.raises(ValueError):
        pascal_part.part_name("antenna")


def _rle(h, w, y0, y1, x0, x1):
    m = np.zeros((1, h, w), bool)
    m[0, y0:y1, x0:x1] = True
    return M.encode(m)[0]


def test_pascal_part_chains_single_instance_only():
    h, w = 20, 20
    car = pascal_part.PPObject(0, "car", _rle(h, w, 0, 10, 0, 10), {"wheel": _rle(h, w, 8, 10, 0, 3)})
    p1 = pascal_part.PPObject(1, "person", _rle(h, w, 10, 20, 0, 5), {"head": _rle(h, w, 10, 12, 0, 5)})
    p2 = pascal_part.PPObject(2, "person", _rle(h, w, 10, 20, 10, 15), {"head": _rle(h, w, 10, 12, 10, 15)})
    chains = pascal_part.chains(pascal_part.PPImage("x", h, w, [car, p1, p2]))
    assert [c[1:3] for c in chains] == [("car", "wheel")]


def test_ade20k_chains_object_part_and_subpart():
    h, w = 30, 30
    objs = [ade20k.ADEObject(0, "cabinet", 0, None, _rle(h, w, 0, 20, 0, 20)),
            ade20k.ADEObject(1, "door", 1, 0, _rle(h, w, 0, 20, 0, 9)),
            ade20k.ADEObject(2, "door", 1, 0, _rle(h, w, 0, 20, 10, 20)),
            ade20k.ADEObject(3, "drawer", 1, 0, _rle(h, w, 15, 20, 0, 20)),
            ade20k.ADEObject(4, "knob", 2, 3, _rle(h, w, 16, 18, 8, 10)),
            ade20k.ADEObject(5, "chair", 0, None, _rle(h, w, 22, 30, 0, 5)),
            ade20k.ADEObject(6, "chair", 0, None, _rle(h, w, 22, 30, 10, 15))]
    chains = ade20k.chains(ade20k.ADEImage("x", h, w, objs))
    got = {(c[1], c[2]) for c in chains}
    assert got == {("cabinet", "door"), ("cabinet", "drawer"), ("drawer of the cabinet", "knob")}
    doors = [c for c in chains if c[2] == "door"][0]
    assert M.areas([doors[4]])[0] == 20 * 9 + 20 * 10   # both doors merged into one mask
    assert ade20k.short_name("person, individual, someone") == "person"


def test_box_hierarchy_rules():
    from treeground.data import lvis_hier, openimages_hier
    assert lvis_hier.display("monitor_(computer_equipment) computer_monitor") == "monitor"
    assert lvis_hier.display("teddy_bear") == "teddy bear"
    car = openimages_hier.OIBox("car", np.array([0.0, 0.0, 0.8, 0.8]))
    wheel = openimages_hier.OIBox("wheel", np.array([0.1, 0.5, 0.3, 0.8]))
    dup = openimages_hier.OIBox("vehicle", np.array([0.0, 0.0, 0.79, 0.8]))   # same entity, second label
    pairs = {("car", "wheel"): {}, ("vehicle", "car"): {}, ("car", "vehicle"): {}}
    ch = openimages_hier.chains("x", [car, wheel, dup], pairs, 100, 100)
    assert [c[1:3] for c in ch] == [("car", "wheel")]
    assert np.allclose(ch[0][4], [10, 50, 30, 80])
