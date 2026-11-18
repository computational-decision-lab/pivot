from __future__ import annotations

import math
from statistics import NormalDist


def required_subgaussian_samples(
    sigma: float,
    margin: float,
    candidates: int,
    delta: float,
) -> int:
    """Return paired rollouts needed for best-update identification.

    With sigma-sub-Gaussian paired differences and a union bound across K
    candidates, `n >= 4 sigma^2 m^-2 log(K/delta)` makes every competing
    sample-mean gap have the correct sign with probability at least 1-delta.
    """

    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return max(1, math.ceil(4.0 * sigma * sigma / (margin * margin) * math.log(candidates / delta)))


def best_update_error_bound(sigma: float, margin: float, candidates: int, samples: int) -> float:
    """Return the explicit union-bound failure probability.

    The bound assumes each pairwise rollout-difference estimate is
    sigma-sub-Gaussian. For a true best-update margin `margin`, a candidate can
    overtake it only when its estimated gap error exceeds half the margin;
    the selected normalization yields `K exp(-n m^2/(4 sigma^2))`.
    """

    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sigma == 0.0:
        return 0.0
    return float(candidates * math.exp(-samples * margin * margin / (4.0 * sigma * sigma)))


def required_cluster_samples(
    sigma: float,
    margin: float,
    alpha: float,
    power: float,
    *,
    two_sided: bool = True,
) -> int:
    """Normal-approximation sample size for a clustered mean effect.

    This is separate from :func:`required_subgaussian_samples`: the latter is
    a worst-case per-round best-update identification bound, whereas CTI
    confirmation tests one paired trajectory-level mean.
    """

    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between zero and one")
    tail = 1.0 - alpha / 2.0 if two_sided else 1.0 - alpha
    critical = NormalDist().inv_cdf(tail)
    target = NormalDist().inv_cdf(power)
    return max(1, math.ceil(((critical + target) * sigma / margin) ** 2))
