from __future__ import annotations

from dataclasses import dataclass

import pytest

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult
from pivot.core.transition import PolicyTransition
from pivot.evaluation.unpaired import UnpairedEvaluator


@dataclass
class AdditiveWorld:
    def evaluate(self, policy: Policy, context: RolloutContext) -> RolloutResult:
        return RolloutResult(
            value=policy.parameters["intensity"] + context.exogenous_value,
            environment_steps=2,
            simulator_calls=1,
        )


def test_unpaired_evaluator_uses_disjoint_contexts_and_independent_standard_error() -> None:
    transition = PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6}),
        round_id=0,
        candidate_index=0,
        improvement_operator="test",
    )
    incumbent = [
        RolloutContext(seed=1, scenario_id="i-1", exogenous_value=0.5),
        RolloutContext(seed=2, scenario_id="i-2", exogenous_value=-0.5),
    ]
    candidate = [
        RolloutContext(seed=11, scenario_id="c-1", exogenous_value=0.0),
        RolloutContext(seed=12, scenario_id="c-2", exogenous_value=0.0),
    ]
    result = UnpairedEvaluator(AdditiveWorld()).evaluate(transition, incumbent, candidate)
    assert result.delta == pytest.approx(0.4)
    assert result.incumbent_seed_ids == (1, 2)
    assert result.candidate_seed_ids == (11, 12)
    assert not set(result.incumbent_seed_ids) & set(result.candidate_seed_ids)
    assert result.standard_error == pytest.approx(0.5)
    assert result.paired is False
