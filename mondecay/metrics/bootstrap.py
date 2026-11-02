"""Episode-clustered bootstrap: resample EPISODES (not rollouts), because k samples of one episode are dependent."""
from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_bootstrap(cluster_ids: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """`stat(idx)` takes an index array into the row-level data and returns a float. Returns (point, lo, hi)."""
    ids = np.asarray(cluster_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(ids)))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        v = stat(idx)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff_bootstrap(cluster_ids, stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                          n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """CI on stat_b - stat_a using the SAME resampled clusters. Both stats index the same row space."""
    return cluster_bootstrap(cluster_ids, lambda idx: stat_b(idx) - stat_a(idx), n_boot, seed, alpha)
