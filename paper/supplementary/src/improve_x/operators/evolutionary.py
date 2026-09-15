from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from improve_x.core.operator import CandidateBatch
from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


@dataclass(frozen=True)
class EvolutionaryMutation:
    """Seeded population mutation operator for transition-level studies.

    The trajectory's selected policy remains the incumbent of every emitted
    transition. Optional parents supply a reproducible population whose
    parameters can seed mutations, while the downstream evaluator supplies the
    selection signal. This keeps generation separate from world evaluation.
    """

    component: str = "intensity"
    mutation_scale: float = 0.08
    lower_bound: float = -0.95
    upper_bound: float = 0.95
    operator_name: str = "evolutionary-mutation"

    def __post_init__(self) -> None:
        if not math.isfinite(self.mutation_scale) or self.mutation_scale <= 0:
            raise ValueError("mutation_scale must be finite and positive")
        if not self.lower_bound < self.upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")

    def propose(
        self,
        incumbent: Policy,
        round_id: int | str,
        seed: int,
        *,
        num_candidates: int = 4,
        parents: Sequence[Policy] = (),
        config_id: str | None = None,
    ) -> CandidateBatch:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if any(not isinstance(parent, Policy) for parent in parents):
            raise TypeError("parents must contain Policy objects")

        population = (incumbent, *parents)
        rng = np.random.default_rng(seed)
        transitions: list[PolicyTransition] = []
        seen_ids: set[str] = set()
        for candidate_index in range(num_candidates):
            parent = population[(seed + candidate_index) % len(population)]
            base_value = parent.parameters.get(self.component, incumbent.parameters.get(self.component, 0.0))
            value = self._bounded(base_value + float(rng.normal(0.0, self.mutation_scale)))
            candidate = self._candidate(parent, value)
            if candidate.policy_id in seen_ids:
                # Bounded mutation can collapse multiple values at an edge. A
                # deterministic interior fallback keeps the batch rankable.
                value = self.lower_bound + (self.upper_bound - self.lower_bound) * (candidate_index + 1) / (
                    num_candidates + 1
                )
                candidate = self._candidate(parent, value)
            seen_ids.add(candidate.policy_id)
            transitions.append(
                PolicyTransition(
                    incumbent=incumbent,
                    candidate=candidate,
                    round_id=round_id,
                    candidate_index=candidate_index,
                    improvement_operator=self.operator_name,
                    edit_type=self.component,
                    seed=seed,
                    config_id=config_id,
                )
            )
        return CandidateBatch(
            incumbent=incumbent,
            candidates=tuple(transitions),
            operator=self.operator_name,
            round_id=round_id,
            seed=seed,
            metadata={"parent_policy_ids": tuple(parent.policy_id for parent in population)},
        )

    def _candidate(self, parent: Policy, value: float) -> Policy:
        return Policy.from_mapping(
            {**dict(parent.parameters), self.component: value},
            metadata={
                **dict(parent.metadata),
                "operator": self.operator_name,
                "component": self.component,
                "parent_policy_id": parent.policy_id,
            },
        )

    def _bounded(self, value: float) -> float:
        return min(self.upper_bound, max(self.lower_bound, value))
