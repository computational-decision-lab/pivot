from __future__ import annotations

import math

import pytest

from pivot.theory.sample_complexity import (
    best_update_error_bound,
    required_cluster_samples,
    required_subgaussian_samples,
)


def test_required_subgaussian_samples_uses_explicit_union_bound() -> None:
    result = required_subgaussian_samples(sigma=2.0, margin=0.5, candidates=4, delta=0.05)
    expected = math.ceil(4 * 2.0**2 / 0.5**2 * math.log(4 / 0.05))
    assert result == expected


def test_required_subgaussian_samples_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="margin"):
        required_subgaussian_samples(1.0, 0.0, 2, 0.05)
    with pytest.raises(ValueError, match="candidates"):
        required_subgaussian_samples(1.0, 1.0, 1, 0.05)


def test_best_update_error_bound_is_inverted_by_sample_rule() -> None:
    n = required_subgaussian_samples(1.0, 0.5, 4, 0.05)
    bound = best_update_error_bound(1.0, 0.5, 4, n)
    assert bound <= 0.05


def test_cluster_power_rule_is_distinct_and_monotone() -> None:
    small = required_cluster_samples(2.0, 0.5, 0.05, 0.8)
    large_margin = required_cluster_samples(2.0, 1.0, 0.05, 0.8)
    assert small > large_margin
    assert small > 0
