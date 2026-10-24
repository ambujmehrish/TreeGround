from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from treeground.tree import Tree

FEATURES = ("logit", "sigma", "rank", "sigma_gap", "logit_gap", "sam_iou", "log_sam_iou", "stability",
            "log_area_frac", "depth", "log_descendants", "is_top1", "ancestor_of_top1", "descendant_of_top1",
            "parent_sigma", "parent_is_root", "best_child_sigma")
SAM_IOU_FLOOR = 1e-3  # SAM 3 predicted IoU is an unbounded regression output; floored before the log


def top_k(sig: np.ndarray, k: int) -> np.ndarray:
    # sigma descending, ties to the lower node index (rank 0 = the selector's argmax)
    return np.argsort(-sig, kind="stable")[:k]


def candidate_features(sig: np.ndarray, logit: np.ndarray, tree: Tree, area_frac: np.ndarray, sam_iou: np.ndarray,
                       stability: np.ndarray, k: int) -> pd.DataFrame:
    idx = top_k(sig, k)
    t, root = idx[0], tree.root
    par = tree.parent[idx]
    sig_ext = np.append(sig, 0.0)  # the root (whole image) is not a node: sigma 0
    log_iou = np.log(np.clip(sam_iou, SAM_IOU_FLOOR, None))
    return pd.DataFrame({
        "node": idx, "rank": np.arange(len(idx)), "logit": logit[idx], "sigma": sig[idx],
        "sigma_gap": sig[t] - sig[idx], "logit_gap": logit[t] - logit[idx], "sam_iou": sam_iou[idx],
        "log_sam_iou": log_iou[idx], "stability": stability[idx], "log_area_frac": np.log(area_frac[idx]),
        "depth": tree.depth[idx].astype(np.float64),
        "log_descendants": np.log1p(tree.tout[idx] - tree.tin[idx]).astype(np.float64),
        "is_top1": (idx == t).astype(np.float64),
        "ancestor_of_top1": np.array([tree.is_strict_descendant(t, v) for v in idx], np.float64),
        "descendant_of_top1": np.array([tree.is_strict_descendant(v, t) for v in idx], np.float64),
        "parent_sigma": sig_ext[par], "parent_is_root": (par == root).astype(np.float64),
        "best_child_sigma": np.array([sig[tree.children[v]].max() if len(tree.children[v]) else 0.0 for v in idx])})


def design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # candidate features, their interaction with the query kind, and the kind itself
    f = df[list(FEATURES)].to_numpy(np.float64)
    p = (df["kind"].to_numpy() == "part").astype(np.float64)[:, None]
    names = list(FEATURES) + [f"{n} x part" for n in FEATURES] + ["is_part"]
    return np.hstack([f, f * p, p]), names


def assign_folds(images, n_folds: int, seed: int) -> dict:
    uniq = np.array(sorted(set(images)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[j]: i % n_folds for i, j in enumerate(perm)}


def pick(df: pd.DataFrame, score) -> pd.Series:
    # per query: row label of the highest score, ties to the lower rank
    s = df.assign(_s=np.asarray(score, np.float64)).sort_values(["qid", "_s", "rank"], ascending=[True, False, True],
                                                                kind="stable")
    first = s.drop_duplicates("qid")
    return pd.Series(first.index, index=first.qid)


@dataclass
class Linear:
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float

    def score(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std) @ self.w + self.b


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std, mean, std


def fit_pointwise(x: np.ndarray, y: np.ndarray, l2: float) -> Linear:
    xs, mean, std = standardize(x)
    y = y.astype(np.float64)
    n, d = xs.shape

    def f(theta):
        z = xs @ theta[:d] + theta[d]
        loss = np.logaddexp(0.0, z) - y * z
        r = (1.0 / (1.0 + np.exp(-z)) - y) / n
        return loss.mean() + 0.5 * l2 * theta[:d] @ theta[:d], np.append(xs.T @ r + l2 * theta[:d], r.sum())

    th = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B").x
    return Linear(mean, std, th[:d], float(th[d]))


def fit_listwise(x: np.ndarray, group: np.ndarray, target: np.ndarray, l2: float) -> Linear:
    # conditional logit: softmax over each query's candidates, cross-entropy against its one target row;
    # rows of one group must be contiguous
    xs, mean, std = standardize(x)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    g = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(group)]))
    y = target.astype(np.float64)
    nq = len(starts)

    def f(w):
        s = xs @ w
        m = np.maximum.reduceat(s, starts)
        e = np.exp(s - m[g])
        z = np.add.reduceat(e, starts)
        loss = -(y @ s - (m + np.log(z)).sum()) / nq
        return loss + 0.5 * l2 * w @ w, xs.T @ (e / z[g] - y) / nq + l2 * w

    w = minimize(f, np.zeros(xs.shape[1]), jac=True, method="L-BFGS-B").x
    return Linear(mean, std, w, 0.0)


def listwise_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # training queries need a candidate overlapping the GT; the target is the highest-IoU candidate (lower rank on ties)
    best = df.groupby("qid")["iou"].transform("max")
    d = df[best > 0].sort_values(["qid", "rank"], kind="stable")
    target = d.index.isin(pick(d, d["iou"]).to_numpy())
    return d, target


def rule_score(df: pd.DataFrame, w: float) -> np.ndarray:
    return df["logit"].to_numpy() + w * df["log_sam_iou"].to_numpy()


def mean_pick_iou(df: pd.DataFrame, score) -> float:
    return float(df.loc[pick(df, score).to_numpy(), "iou"].mean())


def fit_rule(df: pd.DataFrame, grid: list[float]) -> float:
    # weight maximising the mean IoU of the picked candidates on the training queries; ties to the smaller weight
    grid = sorted(grid)
    return grid[int(np.argmax([mean_pick_iou(df, rule_score(df, w)) for w in grid]))]


def cross_validate(df: pd.DataFrame, folds: np.ndarray, fit: Callable, predict: Callable) -> tuple[np.ndarray, list]:
    # each row is scored by the model fit on the other folds (folds are assigned per image)
    scores, models = np.full(len(df), np.nan), []
    for f in np.unique(folds):
        test = folds == f
        model = fit(df[~test])
        scores[test] = predict(model, df[test])
        models.append(model)
    return scores, models


def pointwise_fit(l2: float, thr: float) -> Callable:
    def fit(d: pd.DataFrame) -> Linear:
        return fit_pointwise(design(d)[0], d["iou"].to_numpy() >= thr, l2)
    return fit


def listwise_fit(l2: float) -> Callable:
    def fit(d: pd.DataFrame) -> Linear:
        d, target = listwise_rows(d)
        return fit_listwise(design(d)[0], d["qid"].to_numpy(), target, l2)
    return fit


def linear_predict(model: Linear, d: pd.DataFrame) -> np.ndarray:
    return model.score(design(d)[0])


def importances(models: list[Linear], names: list[str]) -> list[dict]:
    w = np.stack([m.w for m in models])
    rows = [{"feature": n, "coef_mean": float(w[:, i].mean()), "coef_std": float(w[:, i].std())}
            for i, n in enumerate(names)]
    return sorted(rows, key=lambda r: -abs(r["coef_mean"]))


def picked_overlap(qs: pd.DataFrame, cand: pd.DataFrame, picks: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    # per query (order of qs): intersection and union of the picked candidate; no candidate -> nothing predicted
    rows = cand.loc[picks.to_numpy(), ["inter", "union"]].set_index(picks.index)
    j = qs[["qid", "gt_area"]].join(rows, on="qid")
    return j["inter"].fillna(0).to_numpy(np.float64), j["union"].fillna(j["gt_area"]).to_numpy(np.float64)
