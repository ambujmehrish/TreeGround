import numpy as np

from treeground import tree as T


def chain_tree():
    # root(5) -> 0 -> {1, 2}; root -> 3 -> 4
    return T.from_parents(np.array([5, 0, 0, 5, 3, -1]))


def test_dp_picks_lower_iou_node_inside_subtree_over_higher_outside():
    tree = chain_tree()
    ious = np.array([
        [0.9, 0.0, 0.0, 0.0, 0.0],   # only node 0 matches the object
        [0.0, 0.6, 0.0, 0.0, 0.9],   # part: 0.6 inside sub+(0), 0.9 outside it
    ])
    assert T.max_iou_path(tree, ious) == (0, 1)
    assert T.best_match_path(tree, ious) is None


def test_dp_maximises_summed_iou_over_feasible_paths():
    tree = chain_tree()
    ious = np.array([[0.95, 0.0, 0.0, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0, 0.8]])
    assert T.max_iou_path(tree, ious) == (3, 4)
    assert T.best_match_path(tree, ious) is None
    assert T.flat_feasible(ious)
    ious[1, 1] = 0.6
    assert T.max_iou_path(tree, ious) == (0, 1)   # 0.95 + 0.6 > 0.7 + 0.8


def test_dp_tie_breaks():
    tree = T.from_parents(np.array([4, 0, 4, 2, -1]))  # root(4) -> 0 -> 1; root -> 2 -> 3
    by_min = np.array([[0.75, 0.0, 0.5, 0.0], [0.0, 0.25, 0.0, 0.5]])
    assert T.max_iou_path(tree, by_min) == (2, 3)   # equal sums; higher minimum step IoU
    by_lex = np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]])
    assert T.max_iou_path(tree, by_lex) == (0, 1)   # full tie; lexicographically smaller sequence


def test_no_feasible_path():
    tree = chain_tree()
    ious = np.array([[0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9]])
    assert T.max_iou_path(tree, ious) is None
    assert T.feasible_paths(tree, ious) == []
    assert T.max_iou_path(tree, np.array([[0.4] * 5, [0.9] * 5])) is None
