from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidityGate:
    name: str
    passed: bool
    value: float
    threshold: str
    reason: str


@dataclass(frozen=True)
class ConstructValidityReport:
    valid: bool
    gates: tuple[ValidityGate, ...]


def evaluate_e3b_gates(
    *,
    rewards: Sequence[float],
    max_possible_reward: float,
    response_differences: Sequence[float],
    candidate_true_deltas: Sequence[float],
    proxy_deltas: Sequence[float],
    paired_deltas: Sequence[float],
    ceiling_fraction_limit: float = 0.10,
    ceiling_tolerance: float = 0.02,
    response_variance_min: float = 1e-4,
    candidate_dispersion_min: float = 1e-3,
    proxy_correlation_min: float = 0.05,
    proxy_correlation_max: float = 0.999,
    stability_ratio_max: float = 0.5,
) -> ConstructValidityReport:
    """Evaluate the five preregistered E3b construct-validity gates."""

    numeric_rewards = _finite(rewards, "rewards")
    response = _finite(response_differences, "response_differences")
    true_deltas = _finite(candidate_true_deltas, "candidate_true_deltas")
    proxy = _finite(proxy_deltas, "proxy_deltas")
    paired = _finite(paired_deltas, "paired_deltas")
    if max_possible_reward <= 0.0 or not math.isfinite(max_possible_reward):
        raise ValueError("max_possible_reward must be finite and positive")
    if len(true_deltas) != len(proxy):
        raise ValueError("proxy and true candidate deltas must have equal length")

    near_ceiling = sum(
        value >= max_possible_reward * (1.0 - ceiling_tolerance) for value in numeric_rewards
    ) / len(numeric_rewards)
    response_variance = float(np.var(response, ddof=1)) if len(response) > 1 else 0.0
    candidate_dispersion = float(np.std(true_deltas, ddof=1)) if len(true_deltas) > 1 else 0.0
    if len(true_deltas) > 1 and np.std(true_deltas) > 0.0 and np.std(proxy) > 0.0:
        correlation = float(np.corrcoef(proxy, true_deltas)[0, 1])
    else:
        correlation = 1.0
    paired_se = float(np.std(paired, ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else math.inf
    candidate_range = max(true_deltas) - min(true_deltas) if true_deltas else 0.0
    stability_ratio = paired_se / max(candidate_range, 1e-12)

    gates = (
        ValidityGate(
            "no_premature_ceiling",
            near_ceiling < ceiling_fraction_limit,
            near_ceiling,
            f"fraction < {ceiling_fraction_limit}",
            "fraction of policies within two percent of the reward ceiling",
        ),
        ValidityGate(
            "nontrivial_response",
            response_variance > response_variance_min,
            response_variance,
            f"variance > {response_variance_min}",
            "variance of actor-minus-direct improvement",
        ),
        ValidityGate(
            "candidate_diversity",
            len(true_deltas) >= 4 and candidate_dispersion > candidate_dispersion_min,
            candidate_dispersion,
            f"K >= 4 and dispersion > {candidate_dispersion_min}",
            f"candidate count is {len(true_deltas)}",
        ),
        ValidityGate(
            "nonperfect_proxy",
            proxy_correlation_min < abs(correlation) < proxy_correlation_max,
            correlation,
            f"{proxy_correlation_min} < |correlation| < {proxy_correlation_max}",
            "proxy must be informative but not identical to deployment truth",
        ),
        ValidityGate(
            "stable_measurement",
            math.isfinite(stability_ratio) and stability_ratio < stability_ratio_max,
            stability_ratio,
            f"paired SE / candidate range < {stability_ratio_max}",
            "paired rollout noise must resolve candidate margins",
        ),
    )
    return ConstructValidityReport(all(gate.passed for gate in gates), gates)


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    numeric = tuple(float(value) for value in values)
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError(f"{name} must contain finite values")
    return numeric
