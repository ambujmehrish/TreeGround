import numpy as np

GRID = np.linspace(0.0, 1.0, 200)


def rejection_objective(grid: np.ndarray, present_max: np.ndarray, present_iou: np.ndarray, absent_max: np.ndarray,
                        lam: float = 1.0) -> np.ndarray:
    # Spec Eq. 2: max sigma over the candidate set; an empty set has max = -inf and always abstains.
    present_max, present_iou, absent_max = map(np.asarray, (present_max, present_iou, absent_max))
    accept_p = present_max[None, :] >= grid[:, None]
    accept_a = absent_max[None, :] >= grid[:, None]
    gain = (accept_p * present_iou[None, :]).sum(1) + (~accept_a).sum(1) - lam * accept_a.sum(1)
    return gain / max(1, len(present_max) + len(absent_max))


def calibrate_rejection(present_max: np.ndarray, present_iou: np.ndarray, absent_max: np.ndarray, lam: float = 1.0,
                        grid: np.ndarray = GRID) -> tuple[float, np.ndarray]:
    obj = rejection_objective(grid, present_max, present_iou, absent_max, lam)
    return float(grid[int(np.argmax(obj))]), obj  # ties resolve to the smallest threshold
