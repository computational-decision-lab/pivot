from __future__ import annotations

from dataclasses import dataclass

import pytest

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult
from pivot.core.transition import PolicyTransition
from pivot.evaluation.paired import PairedEvaluator


@dataclass
class AdditiveWorld:
    def evaluate(self, policy: Policy, context: RolloutContext) -> RolloutResult:
        value = policy.parameters["intensity"] + context.exogenous_value
        return RolloutResult(value=value, environment_steps=2, simulator_calls=1)


def test_paired_evaluator_subtracts_inside_each_shared_context() -> None:
    transition = PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6}),
        round_id=0,
        candidate_index=0,
        improvement_operator="test",
    )
    contexts = [
        RolloutContext(seed=1, scenario_id="a", exogenous_value=3.0),
        RolloutContext(seed=2, scenario_id="b", exogenous_value=-1.0),
    ]
    result = PairedEvaluator(AdditiveWorld(), mode="actor").evaluate(transition, contexts)
    assert result.deltas == pytest.approx((0.4, 0.4))
    assert result.delta == pytest.approx(0.4)
    assert result.num_rollouts == 2
    assert result.paired_seed_ids == (1, 2)
    assert result.environment_steps == 8
