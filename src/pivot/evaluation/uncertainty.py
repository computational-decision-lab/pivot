from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, n_bootstrap: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    array = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True).mean(axis=1)
    return (float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2)))
