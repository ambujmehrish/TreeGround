import numpy as np

from treeground import masks as M
from treeground import tree as T


def box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def fixture_masks():
    h = w = 40
    return [
        box_mask(h, w, 0, 40, 0, 40),    # 0: whole image
        box_mask(h, w, 0, 20, 0, 20),    # 1: quadrant
        box_mask(h, w, 0, 10, 0, 10),    # 2: inside 1
        box_mask(h, w, 0, 30, 0, 30),    # 3: contains 1 and 2
        box_mask(h, w, 20, 40, 20, 40),  # 4: other quadrant
        box_mask(h, w, 25, 30, 25, 30),  # 5: inside 4
    ]


def test_parents_pick_smallest_container_and_tree_is_acyclic():
    rles = M.encode(np.stack(fixture_masks()))
    tree = T.build_tree(rles)
    assert list(tree.parent[:6]) == [6, 3, 1, 0, 0, 4]
    for v in range(tree.n):
        seen, u = set(), v
        while u != tree.root:
            assert u not in seen
            seen.add(u)
            u = tree.parent[u]
    assert list(tree.depth[:6]) == [1, 3, 4, 2, 2, 3]
    assert list(tree.strict_descendants(0)) == [1, 2, 3, 4, 5]
    assert list(tree.strict_descendants(3)) == [1, 2]
    assert list(tree.strict_descendants(tree.root)) == [0, 1, 2, 3, 4, 5]
    assert tree.is_strict_descendant(2, 3) and not tree.is_strict_descendant(3, 3)


def test_containment_threshold_is_exact():
    areas = np.array([10, 100])
    inter = np.array([[10, 8], [8, 100]])
    assert T.parents_from_overlaps(areas, inter)[0] == 1
    inter[0, 1] = 7
    assert T.parents_from_overlaps(areas, inter)[0] == 2


def test_equal_size_containers_tie_break():
    areas = np.array([10, 50, 50])
    inter = np.array([[10, 9, 10], [9, 50, 0], [10, 0, 50]])
    assert T.parents_from_overlaps(areas, inter)[0] == 2
    inter[0, 2] = 9
    assert T.parents_from_overlaps(areas, inter)[0] == 1


def test_randomised_tree_preserves_depth_histogram():
    rles = M.encode(np.stack(fixture_masks()))
    tree = T.build_tree(rles)
    for seed in range(20):
        r = T.randomised_tree(tree, seed)
        assert np.array_equal(r.depth, tree.depth)
        for v in range(r.n):
            assert r.depth[r.parent[v]] == r.depth[v] - 1
        assert np.array_equal(T.randomised_tree(tree, seed).parent, r.parent)


def test_intersections_are_exact_pixel_counts():
    ms = fixture_masks()
    rles = M.encode(np.stack(ms))
    inter = M.intersections(rles, rles)
    for i in range(len(ms)):
        for j in range(len(ms)):
            assert inter[i, j] == (ms[i] & ms[j]).sum()


def test_mask_ops_with_many_masks():
    rng = np.random.default_rng(0)
    ms = rng.random((300, 24, 32)) > 0.5
    rles = M.encode(ms)
    assert np.array_equal(M.areas(rles), ms.reshape(300, -1).sum(1))
    assert M.boxes_xyxy(rles).shape == (300, 4)
    assert M.iou(rles, rles).shape == (300, 300)
