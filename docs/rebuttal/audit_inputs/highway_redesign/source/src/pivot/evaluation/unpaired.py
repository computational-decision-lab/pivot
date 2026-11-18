from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult
from pivot.core.transition import PolicyTransition


@dataclass(frozen=True)
class UnpairedEvaluation:
    """Independent-sample candidate-minus-incumbent evaluation."""

    incumbent_value: float
    candidate_value: float
    delta: float
    standard_error: float
    confidence_interval: tuple[float, float]
    num_incumbent_rollouts: int
    num_candidate_rollouts: int
    incumbent_seed_ids: tuple[int, ...]
    candidate_seed_ids: tuple[int, ...]
    environment_steps: int
    simulator_calls: int
    paired: bool = False


class UnpairedEvaluator:
    def __init__(self, world: Any, mode: str = "observer") -> None:
        self.world = world
        self.mode = mode

    def evaluate(
        self,
        transition: PolicyTransition,
        incumbent_contexts: Sequence[RolloutContext],
        candidate_contexts: Sequence[RolloutContext],
    ) -> UnpairedEvaluation:
        if not incumbent_contexts or not candidate_contexts:
            raise ValueError("both independent context sets must be non-empty")
        incumbent_ids = tuple(int(context.seed) for context in incumbent_contexts)
        candidate_ids = tuple(int(context.seed) for context in candidate_contexts)
        if set(incumbent_ids) & set(candidate_ids):
            raise ValueError("unpaired evaluation requires disjoint context seeds")
        incumbent_results = [self._evaluate(transition.incumbent, context) for context in incumbent_contexts]
        candidate_results = [self._evaluate(transition.candidate, context) for context in candidate_contexts]
        incumbent_values = [float(result.value) for result in incumbent_results]
        candidate_values = [float(result.value) for result in candidate_results]
        incumbent_mean = sum(incumbent_values) / len(incumbent_values)
        candidate_mean = sum(candidate_values) / len(candidate_values)
        delta = candidate_mean - incumbent_mean
        standard_error = math.sqrt(
            _sample_variance(incumbent_values) / len(incumbent_values)
            + _sample_variance(candidate_values) / len(candidate_values)
        )
        margin = 1.96 * standard_error
        all_results = incumbent_results + candidate_results
        return UnpairedEvaluation(
            incumbent_value=incumbent_mean,
            candidate_value=candidate_mean,
            delta=delta,
            standard_error=standard_error,
            confidence_interval=(delta - margin, delta + margin),
            num_incumbent_rollouts=len(incumbent_results),
            num_candidate_rollouts=len(candidate_results),
            incumbent_seed_ids=incumbent_ids,
            candidate_seed_ids=candidate_ids,
            environment_steps=sum(result.environment_steps for result in all_results),
            simulator_calls=sum(result.simulator_calls for result in all_results),
        )

    def _evaluate(self, policy: Policy, context: RolloutContext) -> RolloutResult:
        try:
            return cast(RolloutResult, self.world.evaluate(policy, context, mode=self.mode))
        except TypeError as error:
            if "unexpected keyword argument 'mode'" not in str(error):
                raise
            return cast(RolloutResult, self.world.evaluate(policy, context))


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)
