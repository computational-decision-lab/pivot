from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from pivot.core.policy import Policy
from pivot.core.result import PairedEvaluation, RolloutContext, RolloutResult
from pivot.core.transition import PolicyTransition


class PairedEvaluator:
    def __init__(self, world: Any, mode: str = "observer") -> None:
        self.world = world
        self.mode = mode

    def evaluate(
        self, transition: PolicyTransition, contexts: Sequence[RolloutContext]
    ) -> PairedEvaluation:
        if not contexts:
            raise ValueError("at least one paired context is required")
        incumbent_values = []
        candidate_values = []
        deltas = []
        steps = 0
        calls = 0
        costs = []
        for context in contexts:
            incumbent = self._evaluate(transition.incumbent, context)
            candidate = self._evaluate(transition.candidate, context)
            incumbent_values.append(float(incumbent.value))
            candidate_values.append(float(candidate.value))
            deltas.append(float(candidate.value - incumbent.value))
            steps += incumbent.environment_steps + candidate.environment_steps
            calls += incumbent.simulator_calls + candidate.simulator_calls
            if incumbent.compute_cost is not None:
                costs.append(incumbent.compute_cost)
            if candidate.compute_cost is not None:
                costs.append(candidate.compute_cost)
        mean_delta = sum(deltas) / len(deltas)
        if len(deltas) > 1:
            variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
            standard_error = math.sqrt(variance / len(deltas))
        else:
            standard_error = 0.0
        margin = 1.96 * standard_error
        return PairedEvaluation(
            incumbent_value=sum(incumbent_values) / len(incumbent_values),
            candidate_value=sum(candidate_values) / len(candidate_values),
            delta=mean_delta,
            standard_error=standard_error,
            confidence_interval=(mean_delta - margin, mean_delta + margin),
            num_rollouts=len(contexts),
            paired_seed_ids=tuple(context.seed for context in contexts),
            deltas=tuple(deltas),
            environment_steps=steps,
            simulator_calls=calls,
            compute_cost=sum(costs) if costs else None,
        )

    def _evaluate(self, policy: Policy, context: RolloutContext) -> RolloutResult:
        try:
            return cast(RolloutResult, self.world.evaluate(policy, context, mode=self.mode))
        except TypeError as error:
            if "unexpected keyword argument 'mode'" not in str(error):
                raise
            return cast(RolloutResult, self.world.evaluate(policy, context))
