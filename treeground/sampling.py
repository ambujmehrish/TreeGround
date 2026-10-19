from dataclasses import dataclass, field

import numpy as np

from treeground.tree import Tree

IOU_THR = 0.5
N_HARD, N_UNIFORM, N_CROSS = 32, 32, 64


@dataclass
class StepTarget:
    pos: np.ndarray
    ignore: np.ndarray


@dataclass
class Draw:
    ids: np.ndarray
    shortfall: int
    stats: dict = field(default_factory=dict)


def best_index(ious: np.ndarray) -> int:
    return int(np.argmax(ious))  # ties resolve to the lower node index


def single_step_target(ious: np.ndarray) -> StepTarget | None:
    if len(ious) == 0:
        return None
    best = best_index(ious)
    if ious[best] < IOU_THR:
        return None
    others = np.flatnonzero(ious >= IOU_THR)
    return StepTarget(np.array([best]), others[others != best])


def chain_step_targets(path: tuple[int, ...], step_ious: np.ndarray) -> list[StepTarget]:
    # The DP-selected node is the positive even when the single-step rule would ignore it; the
    # individually best node (and any other high-overlap node) is ignored for that step.
    out = []
    for t, v in enumerate(path):
        high = np.flatnonzero(step_ious[t] >= IOU_THR)
        out.append(StepTarget(np.array([v]), high[high != v]))
    return out


def text_target(labels: np.ndarray, part_names: np.ndarray, high_overlap: np.ndarray, label: str
                ) -> StepTarget:
    part = label.split(":", 1)[1]
    pos = np.flatnonzero(labels == label)
    other_object = (part_names == part) & (labels != label)
    ignore = np.flatnonzero((other_object | high_overlap) & (labels != label))
    return StepTarget(pos, ignore)


def eligible(candidates: np.ndarray, *exclude: np.ndarray) -> np.ndarray:
    drop = np.concatenate([np.asarray(e, dtype=np.int64) for e in exclude]) if exclude else np.zeros(0, np.int64)
    return np.setdiff1d(np.asarray(candidates, dtype=np.int64), drop)


def hard_then_uniform(pool: np.ndarray, scores: np.ndarray, rng: np.random.Generator, total: int,
                      n_hard: int) -> Draw:
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= total:
        return Draw(pool.copy(), total - len(pool), {"hard": min(n_hard, len(pool)), "uniform": 0})
    n_hard = min(n_hard, total)
    order = np.lexsort((pool, -scores[pool]))
    hard = pool[order[:n_hard]]
    rest = pool[order[n_hard:]]
    uniform = rng.choice(rest, size=total - n_hard, replace=False) if total > n_hard else np.zeros(0, np.int64)
    return Draw(np.concatenate([hard, np.sort(uniform)]), 0, {"hard": n_hard, "uniform": total - n_hard})


def in_image_negatives(pool: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> Draw:
    return hard_then_uniform(pool, scores, rng, N_HARD + N_UNIFORM, N_HARD)


def chain_negatives(subtree_pool: np.ndarray, outside_pool: np.ndarray, scores: np.ndarray,
                    rng: np.random.Generator) -> Draw:
    total = N_HARD + N_UNIFORM
    first = hard_then_uniform(subtree_pool, scores, rng, total, N_HARD)
    taken = len(first.ids)
    if taken >= total:
        return Draw(first.ids, 0, {"subtree": taken, "topup": 0})
    top = hard_then_uniform(outside_pool, scores, rng, total - taken, N_HARD)
    return Draw(np.concatenate([first.ids, top.ids]), top.shortfall, {"subtree": taken, "topup": len(top.ids)})


def cross_image_negatives(pool: np.ndarray, rng: np.random.Generator, n: int = N_CROSS) -> Draw:
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= n:
        return Draw(pool.copy(), n - len(pool))
    return Draw(np.sort(rng.choice(pool, size=n, replace=False)), 0)


def inference_pool(tree: Tree, anchor: int | None) -> np.ndarray:
    # C_single when anchor is None (all nodes of the image); C_chain = sub+(anchor) otherwise.
    return np.arange(tree.n) if anchor is None else tree.strict_descendants(anchor)


def pre_scoring_sample(pool: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if b > len(pool):
        raise ValueError("sample size exceeds pool size")
    return np.sort(rng.choice(pool, size=b, replace=False))
