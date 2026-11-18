"""Analytic Gaussian decision bounds, conditional on the registered model.

These are not frequentist confidence sequences or evidence of model calibration.
They bound posterior decision error/regret without using finite fantasy scores as
certificates. Full latent covariance, including shared baseline uncertainty, enters
every contrast. Repeated inspection does not create a frequentist guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def decision_bounds(model: Any) -> dict[str, Any]:
    means = np.asarray(model.decision_means)
    covariance = np.asarray(model.decision_covariance)
    chosen = int(np.argmax(means))

    def bounds(index: int) -> tuple[float, float, float]:
        probabilities, losses = [], []
        for competitor in range(len(means)):
            if competitor == index:
                continue
            gap = float(means[competitor] - means[index])
            variance = float(
                covariance[index, index]
                + covariance[competitor, competitor]
                - 2 * covariance[index, competitor]
            )
            if variance < -1e-10 * max(1.0, float(np.max(np.abs(covariance)))):
                raise ValueError("negative Gaussian contrast variance")
            if variance <= 0:
                probability = float(gap > 0 or (gap == 0 and competitor < index))
                loss = max(0.0, gap)
            else:
                sigma = math.sqrt(variance)
                z = gap / sigma
                probability = 0.5 * math.erfc(-z / math.sqrt(2))
                loss = max(
                    0.0, sigma * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) + gap * probability
                )
            probabilities.append(probability)
            losses.append(loss)
        return max(0.0, 1 - sum(probabilities)), 1 - max(probabilities, default=0.0), sum(losses)

    lower, upper, regret = bounds(chosen)
    incumbent_lower, incumbent_upper, _ = bounds(0)
    return {
        "selection_probability_lower": lower,
        "selection_probability_upper": upper,
        "incumbent_probability_lower": incumbent_lower,
        "incumbent_probability_upper": incumbent_upper,
        "expected_regret_upper": regret,
        "bound_scope": "conditional_on_registered_joint_gaussian_model",
        "frequentist_confidence_certified": False,
    }


@dataclass(frozen=True)
class PosteriorStopRule:
    delta: float
    eta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")

    def acquisition_upper(self, bounds: dict[str, Any], actions: Sequence[Any]) -> float:
        # Any one-step KG is no greater than the value of perfect information,
        # which is no greater than the sum of positive Gaussian contrast losses.
        return max(
            (bounds["expected_regret_upper"] / action.cost for action in actions), default=0.0
        )

    def satisfied(self, bounds: dict[str, Any], actions: Sequence[Any]) -> bool:
        return (
            bounds["selection_probability_lower"] >= 1 - self.delta
            and self.acquisition_upper(bounds, actions) <= self.eta
        )
