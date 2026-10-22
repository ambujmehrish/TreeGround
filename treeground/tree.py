from dataclasses import dataclass

import numpy as np

from treeground import masks as M

CONTAIN_NUM, CONTAIN_DEN = 4, 5  # |m_i ∩ m_j| >= 0.8 |m_i|, compared on exact integers


@dataclass
class Tree:
    parent: np.ndarray  # length n + 1; node n is the root r (whole image), parent[r] = -1
    depth: np.ndarray
    children: list[np.ndarray]
    order: np.ndarray   # nodes in DFS pre-order starting at the root
    tin: np.ndarray
    tout: np.ndarray    # position in `order` of the last node of each subtree

    @property
    def n(self) -> int:
        return len(self.parent) - 1

    @property
    def root(self) -> int:
        return self.n

    def strict_descendants(self, v: int) -> np.ndarray:
        return np.sort(self.order[self.tin[v] + 1: self.tout[v] + 1])

    def is_strict_descendant(self, u: int, v: int) -> bool:
        return self.tin[v] < self.tin[u] <= self.tout[v]


def from_parents(parent: np.ndarray) -> Tree:
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent) - 1
    children = [[] for _ in range(n + 1)]
    for v in range(n):
        children[parent[v]].append(v)
    children = [np.array(sorted(c), dtype=np.int64) for c in children]
    depth = np.zeros(n + 1, dtype=np.int64)
    order, tin, tout = [], np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    stack = [(n, False)]
    while stack:
        v, done = stack.pop()
        if done:
            tout[v] = len(order) - 1
            continue
        tin[v] = len(order)
        order.append(v)
        stack.append((v, True))
        for c in children[v][::-1]:
            depth[c] = depth[v] + 1
            stack.append((c, False))
    if len(order) != n + 1:
        raise ValueError("parent array does not describe a tree rooted at r")
    return Tree(parent, depth, children, np.array(order, dtype=np.int64), tin, tout)


def parents_from_overlaps(areas: np.ndarray, inter: np.ndarray, num: int = CONTAIN_NUM,
                          den: int = CONTAIN_DEN) -> np.ndarray:
    areas = np.asarray(areas, dtype=np.int64)
    n = len(areas)
    parent = np.full(n + 1, -1, dtype=np.int64)
    for i in range(n):
        ok = (areas > areas[i]) & (den * inter[i] >= num * areas[i])
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            parent[i] = n
            continue
        # smallest container; ties: higher containment, then lower index
        key = np.lexsort((cand, -inter[i, cand], areas[cand]))
        parent[i] = cand[key[0]]
    return parent


def intersections_dense(rles: list[M.Rle], device: str, block: int = 512) -> np.ndarray:
    # exact pairwise pixel intersections on an accelerator: masks held as bytes, multiplied as float32 blocks.
    # Every partial sum is an integer below 2**24, so the result equals intersections(rles, rles) for images
    # under 16.7 MP. Falls back to the exact CPU route when the masks do not fit in accelerator memory.
    import torch
    n = len(rles)
    if not n:
        return np.zeros((0, 0), dtype=np.int64)
    h, w = rles[0]["size"]
    hw = h * w
    if hw >= 2 ** 24:
        return M.intersections(rles, rles)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"  # e.g. a login node: same exact integers, computed with CPU matmuls
    if str(device).startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        need = n * hw + 2 * min(block, n) * hw * 4 + n * min(block, n) * 4
        if need > 0.8 * free:
            return M.intersections(rles, rles)
    out = np.zeros((n, n), dtype=np.int64)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    dense = None
    try:
        dense = torch.empty((n, hw), dtype=torch.uint8, device=device)
        for s in range(0, n, block):
            chunk = np.stack([M.decode(r).reshape(-1) for r in rles[s:s + block]])
            dense[s:s + len(chunk)] = torch.from_numpy(chunk.astype(np.uint8)).to(device)
        for s in range(0, n, block):
            rows = dense[s:s + block].float()
            for t in range(0, n, block):
                cols = dense[t:t + block].float()
                out[s:s + block, t:t + block] = torch.round(rows @ cols.T).to(torch.int64).cpu().numpy()
                del cols
            del rows
    except torch.OutOfMemoryError:
        dense = None
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return M.intersections(rles, rles)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        del dense
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def box_overlaps(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # integer box areas and pairwise box intersections (RLE boxes are whole-pixel [x0, y0, x1, y1))
    b = np.rint(np.asarray(boxes, dtype=np.float64).reshape(-1, 4)).astype(np.int64)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    return w * h, iw * ih


def build_box_tree(boxes: np.ndarray, num: int = CONTAIN_NUM, den: int = CONTAIN_DEN) -> Tree:
    # same parent rule as build_tree, applied to the nodes' bounding boxes instead of their masks
    areas, inter = box_overlaps(boxes)
    return from_parents(parents_from_overlaps(areas, inter, num, den))


def build_tree(rles: list[M.Rle], device: str | None = None) -> Tree:
    # device: compute pairwise intersections on an accelerator (identical integers, much faster for 1000+ masks)
    areas = M.areas(rles)
    inter = intersections_dense(rles, device) if device and len(rles) > 64 else M.intersections(rles, rles)
    return from_parents(parents_from_overlaps(areas, inter))


def randomised_tree(tree: Tree, seed: int) -> Tree:
    rng = np.random.default_rng(seed)
    n = tree.n
    parent = np.full(n + 1, -1, dtype=np.int64)
    depth = tree.depth[:n]
    by_depth = {d: np.flatnonzero(depth == d) for d in np.unique(depth)}
    for v in range(n):
        d = depth[v]
        parent[v] = n if d == 1 else rng.choice(by_depth[d - 1])
    return from_parents(parent)


def step_candidates(step_ious: np.ndarray, thr: float = 0.5) -> list[np.ndarray]:
    return [np.flatnonzero(row >= thr) for row in np.asarray(step_ious)]


def feasible_paths(tree: Tree, step_ious: np.ndarray, thr: float = 0.5) -> list[tuple[int, ...]]:
    cands = step_candidates(step_ious, thr)
    paths: list[tuple[int, ...]] = [(int(v),) for v in cands[0]]
    for t in range(1, len(cands)):
        allowed = cands[t]
        paths = [p + (int(v),) for p in paths for v in allowed if tree.is_strict_descendant(v, p[-1])]
    return paths


def max_iou_path(tree: Tree, step_ious: np.ndarray, thr: float = 0.5) -> tuple[int, ...] | None:
    # Exact maximiser of the summed IoU over all feasible paths; ties: higher minimum step IoU,
    # then the lexicographically smaller node sequence.
    ious = np.asarray(step_ious)
    paths = feasible_paths(tree, ious, thr)
    if not paths:
        return None

    def key(p: tuple[int, ...]):
        vals = [ious[t, v] for t, v in enumerate(p)]
        return (-sum(vals), -min(vals), p)

    return min(paths, key=key)


def best_match_path(tree: Tree, step_ious: np.ndarray, thr: float = 0.5) -> tuple[int, ...] | None:
    ious = np.asarray(step_ious)
    best = tuple(int(np.argmax(row)) for row in ious)  # argmax ties resolve to the lower index
    if any(ious[t, v] < thr for t, v in enumerate(best)):
        return None
    if any(not tree.is_strict_descendant(best[t], best[t - 1]) for t in range(1, len(best))):
        return None
    return best


def flat_feasible(step_ious: np.ndarray, thr: float = 0.5) -> bool:
    return all(len(c) > 0 for c in step_candidates(step_ious, thr))
