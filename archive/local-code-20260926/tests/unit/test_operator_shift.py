from __future__ import annotations

import math

import pytest

from pivot.theory.operator_shift import (
    chi_square_divergence,
    effective_sample_size,
    operator_shift_bound,
    operator_shift_summary,
)


def test_operator_shift_bound_is_tight_for_equal_laws() -> None:
    losses = [0.2, 0.4, 0.6]
    population = [1 / 3, 1 / 3, 1 / 3]
    operator = [1 / 3, 1 / 3, 1 / 3]

    assert chi_square_divergence(operator, population) == pytest.approx(0.0)
    assert effective_sample_size(operator) == pytest.approx(3.0)
    assert operator_shift_bound(losses, operator, population) == pytest.approx(
        math.sqrt(sum(weight * loss * loss for weight, loss in zip(population, losses)))
    )


def test_operator_shift_summary_reports_if_and_divergence() -> None:
    rows = [
        {"loss": 0.1, "population_weight": 0.5, "operator_weight": 0.8},
        {"loss": 0.3, "population_weight": 0.3, "operator_weight": 0.1},
        {"loss": 0.8, "population_weight": 0.2, "operator_weight": 0.1},
    ]
    summary = operator_shift_summary(rows)

    assert summary["operator_if"] == pytest.approx(0.19)
    assert summary["chi_square_divergence"] == pytest.approx(
        (0.8 - 0.5) ** 2 / 0.5 + (0.1 - 0.3) ** 2 / 0.3 + (0.1 - 0.2) ** 2 / 0.2
    )
    assert summary["effective_sample_size"] == pytest.approx(1 / (0.8**2 + 0.1**2 + 0.1**2))
    assert summary["bound_holds"] is True


def test_operator_shift_rejects_invalid_probability_laws() -> None:
    with pytest.raises(ValueError, match="positive"):
        chi_square_divergence([0.5, 0.5], [0.0, 1.0])
    with pytest.raises(ValueError, match="sum"):
        effective_sample_size([0.2, 0.2])
