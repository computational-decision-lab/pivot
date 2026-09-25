from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _probability_law(values: Sequence[float], name: str, *, strictly_positive: bool) -> tuple[float, ...]:
    law = tuple(float(value) for value in values)
    if not law:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in law):
        raise ValueError(f"{name} must contain finite probabilities")
    if strictly_positive and any(value <= 0.0 for value in law):
        raise ValueError(f"{name} probabilities must be positive")
    if not strictly_positive and any(value < 0.0 for value in law):
        raise ValueError(f"{name} probabilities must be non-negative")
    if not math.isclose(sum(law), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name} probabilities must sum to one")
    return law


def chi_square_divergence(
    operator_weights: Sequence[float], population_weights: Sequence[float]
) -> float:
    """Return discrete chi-square divergence chi2(Q_A || P).

    The population law is required to have positive mass on every represented
    transition, which enforces the declared absolute-continuity assumption.
    """

    operator = _probability_law(operator_weights, "operator", strictly_positive=False)
    population = _probability_law(population_weights, "population", strictly_positive=True)
    if len(operator) != len(population):
        raise ValueError("operator and population laws must have equal length")
    return float(sum((q_value - p_value) ** 2 / p_value for q_value, p_value in zip(operator, population)))


def effective_sample_size(operator_weights: Sequence[float]) -> float:
    """Return the discrete concentration ESS, 1 / sum_tau Q_A(tau)^2."""

    operator = _probability_law(operator_weights, "operator", strictly_positive=False)
    return float(1.0 / sum(value * value for value in operator))


def operator_shift_bound(
    losses: Sequence[float],
    operator_weights: Sequence[float],
    population_weights: Sequence[float],
) -> float:
    """Evaluate the Cauchy--Schwarz upper bound on operator-relative IF."""

    numeric_losses = tuple(float(value) for value in losses)
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_losses):
        raise ValueError("transition losses must be finite and non-negative")
    operator = _probability_law(operator_weights, "operator", strictly_positive=False)
    population = _probability_law(population_weights, "population", strictly_positive=True)
    if len(numeric_losses) != len(operator) or len(operator) != len(population):
        raise ValueError("losses and probability laws must have equal length")
    second_moment = sum(weight * loss * loss for weight, loss in zip(population, numeric_losses))
    divergence = chi_square_divergence(operator, population)
    return float(math.sqrt(second_moment) * math.sqrt(1.0 + divergence))


def operator_shift_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    loss_key: str = "loss",
    population_key: str = "population_weight",
    operator_key: str = "operator_weight",
) -> dict[str, float | int | bool]:
    """Summarize empirical IF and its operator-shift bound on fixed support."""

    if not rows:
        raise ValueError("operator-shift rows must not be empty")
    losses = [float(row[loss_key]) for row in rows]
    population = [float(row[population_key]) for row in rows]
    operator = [float(row[operator_key]) for row in rows]
    divergence = chi_square_divergence(operator, population)
    bound = operator_shift_bound(losses, operator, population)
    observed = float(sum(weight * loss for weight, loss in zip(operator, losses)))
    return {
        "n_transitions": len(rows),
        "operator_if": observed,
        "population_loss": float(sum(weight * loss for weight, loss in zip(population, losses))),
        "population_second_moment": float(
            sum(weight * loss * loss for weight, loss in zip(population, losses))
        ),
        "chi_square_divergence": divergence,
        "effective_sample_size": effective_sample_size(operator),
        "operator_shift_bound": bound,
        "bound_holds": observed <= bound + 1e-12,
    }
