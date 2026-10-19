import numpy as np
import torch

AREA_BINS = ((32 ** 2, "small"), (96 ** 2, "medium"))


def area_bin(area: float) -> str:
    for limit, name in AREA_BINS:
        if area < limit:
            return name
    return "large"


def ranked_ap(scores: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-row AP and R@1 over valid gallery items; invalid items are removed from the ranking.
    s = scores.masked_fill(~valid, float("-inf"))
    order = torch.sort(s, dim=1, descending=True, stable=True).indices
    pos = torch.gather(positive & valid, 1, order).float()
    keep = torch.gather(valid, 1, order).float()
    rank = keep.cumsum(1)
    hits = pos.cumsum(1)
    n_pos = pos.sum(1)
    ap = (pos * hits / rank.clamp(min=1)).sum(1) / n_pos.clamp(min=1)
    r1 = pos[:, 0]
    return ap, r1


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05
                 ) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        stats[i] = (w * sums).sum() / max((w * counts).sum(), 1.0)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mean_ci(values, groups, n: int = 1000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = bootstrap_ci(values, groups, n, seed)
    return {"mean": float(values.mean()) if len(values) else float("nan"), "ci95": [lo, hi], "n": int(len(values))}


def describe(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}
    q = np.quantile(v, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"n": int(len(v)), "mean": float(v.mean()), "min": q[0], "p10": q[1], "p25": q[2], "median": q[3],
            "p75": q[4], "p90": q[5], "max": q[6]}
