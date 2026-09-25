from __future__ import annotations

import pytest

from improve_x.operators import EvolutionaryMutation
from pivot.core.policy import Policy


def test_evolutionary_mutation_generates_seeded_candidate_batch() -> None:
    incumbent = Policy.from_mapping({"intensity": 0.2})
    operator = EvolutionaryMutation(mutation_scale=0.05)

    first = operator.propose(incumbent, round_id="r0", seed=17, num_candidates=3)
    second = operator.propose(incumbent, round_id="r0", seed=17, num_candidates=3)

    assert first.candidate_ids == second.candidate_ids
    assert len(first.candidates) == 3
    assert {transition.incumbent.policy_id for transition in first.candidates} == {incumbent.policy_id}
    assert {transition.improvement_operator for transition in first.candidates} == {"evolutionary-mutation"}
    assert all("parent_policy_id" in transition.candidate.metadata for transition in first.candidates)


def test_evolutionary_mutation_can_use_parent_population() -> None:
    incumbent = Policy.from_mapping({"intensity": 0.2, "risk": 0.1})
    parent = Policy.from_mapping({"intensity": 0.7, "risk": 0.4})
    operator = EvolutionaryMutation(mutation_scale=0.01)

    batch = operator.propose(incumbent, round_id=2, seed=4, num_candidates=5, parents=(parent,))

    assert parent.policy_id in {transition.candidate.metadata["parent_policy_id"] for transition in batch.candidates}
    assert all(transition.candidate.parameters["intensity"] <= 0.95 for transition in batch.candidates)
    assert all(transition.candidate.parameters["intensity"] >= -0.95 for transition in batch.candidates)


def test_evolutionary_mutation_rejects_nonpositive_mutation_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        EvolutionaryMutation(mutation_scale=0.0)
