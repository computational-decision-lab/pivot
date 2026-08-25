from __future__ import annotations

from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


def test_policy_transition_round_trips_with_explicit_nulls() -> None:
    transition = PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6}),
        round_id=3,
        candidate_index=1,
        improvement_operator="synthetic",
        edit_type="intensity",
        delta_actor=None,
        delta_strategic=None,
    )
    record = transition.to_record()
    assert record["transition_id"] == transition.transition_id
    assert "delta_actor" in record and record["delta_actor"] is None
    restored = PolicyTransition.from_record(record)
    assert restored == transition


def test_policy_ids_are_stable_and_input_mappings_are_copied() -> None:
    values = {"intensity": 0.2}
    policy = Policy.from_mapping(values)
    values["intensity"] = 0.8
    assert policy.parameters["intensity"] == 0.2
    assert policy.policy_id == Policy.from_mapping({"intensity": 0.2}).policy_id
