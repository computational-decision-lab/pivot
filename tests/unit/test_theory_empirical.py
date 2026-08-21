from __future__ import annotations

from math import isclose

import pytest

from pivot.theory.empirical import (
    evaluate_global_fidelity_case,
    evaluate_response_footprint_case,
)


def test_global_fidelity_blindness_adjacent_swap_has_vanishing_global_error() -> None:
    small = evaluate_global_fidelity_case(32, operator_samples=24, seed=7)
    large = evaluate_global_fidelity_case(256, operator_samples=24, seed=7)

    assert small["improvement_reversal_rate"] == 1.0
    assert small["improvement_sign_consistency"] == 0.0
    assert large["global_mae"] < small["global_mae"]
    assert large["spearman_deficit"] < small["spearman_deficit"]
    assert large["operator_ide"] > large["global_mae"]


def test_global_fidelity_blindness_matches_constructive_errors() -> None:
    n_policies = 64
    result = evaluate_global_fidelity_case(n_policies, operator_samples=10, seed=3)

    expected_mae = 2.0 / (n_policies * (n_policies - 1))
    expected_operator_ide = 2.0 / (n_policies - 1)
    assert result["global_mae"] == pytest.approx(expected_mae)
    assert result["operator_ide"] == pytest.approx(expected_operator_ide)
    assert result["operator_sample_count"] == 10


def test_response_footprint_bound_is_tight_for_analytic_response_map() -> None:
    result = evaluate_response_footprint_case(
        response_strength=0.75,
        update_footprint=0.2,
        value_lipschitz=1.7,
        seed=11,
    )

    assert result["bound_holds"] is True
    assert isclose(float(result["bound_ratio"]), 1.0, rel_tol=1e-12)
    assert result["bound_slack"] == pytest.approx(0.0)


def test_response_footprint_rejects_invalid_grid_values() -> None:
    with pytest.raises(ValueError, match="response_strength"):
        evaluate_response_footprint_case(-0.1, 0.2)
    with pytest.raises(ValueError, match="update_footprint"):
        evaluate_response_footprint_case(0.1, -0.2)
