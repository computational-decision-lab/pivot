from __future__ import annotations

from pivot.core.policy import Policy
from pivot.v9.operators import OPERATOR_FAMILIES, chi_square_shift, generate_candidate_batch


def test_all_operator_families_produce_unique_reproducible_batches() -> None:
    incumbent = Policy.from_mapping({"intensity": 0.2, "bias": 0.0})
    for family in OPERATOR_FAMILIES:
        first = generate_candidate_batch(incumbent, family=family, shift_level=1.0, count=8, seed=10)
        second = generate_candidate_batch(incumbent, family=family, shift_level=1.0, count=8, seed=10)
        assert [item.candidate.policy_id for item in first] == [item.candidate.policy_id for item in second]
        assert len({item.candidate.policy_id for item in first}) == 8


def test_shift_diagnostic_is_monotone_and_zero_at_global_reference() -> None:
    levels = [0.0, 0.1, 0.5, 1.0, 2.0]
    values = [chi_square_shift(level) for level in levels]
    assert values[0] == 0.0
    assert values == sorted(values)
