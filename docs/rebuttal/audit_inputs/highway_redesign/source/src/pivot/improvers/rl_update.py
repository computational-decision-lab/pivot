from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition


@dataclass(frozen=True)
class RLUpdateOperator:
    """A tiny, reproducible update operator for the controlled E3 loop.

    It is intentionally a finite-difference-free hill-climbing update.  The
    operator is only a source of candidate transitions; it does not define the
    evaluation world or the PIVOT acquisition policy.
    """

    component: str = "intensity"
    operator_name: str = "rl-update"
    step_size: float = 0.08

    def propose(
        self,
        incumbent: Policy,
        context: RolloutContext,
        num_candidates: int = 1,
        optimization_strength: float = 1.0,
        seed: int = 0,
        round_id: int | str = 0,
        config_id: str | None = None,
    ) -> list[PolicyTransition]:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if optimization_strength < 0 or not np.isfinite(optimization_strength):
            raise ValueError("optimization_strength must be finite and non-negative")
        rng = np.random.default_rng(seed)
        current = incumbent.parameters.get(self.component, 0.0)
        # The direct reward in the controlled world has a positive local
        # gradient around the initial policy.  The state term keeps the update
        # operator context-aware without importing an environment.
        gradient_proxy = 1.5 - 0.8 * current + 0.05 * context.initial_state
        step = self.step_size * optimization_strength * gradient_proxy
        candidates: list[PolicyTransition] = []
        for index in range(num_candidates):
            noise = float(rng.normal(0.0, self.step_size * 0.01))
            value = min(0.95, max(-0.95, current + step + noise))
            candidate = Policy.from_mapping(
                {**dict(incumbent.parameters), self.component: value},
                metadata={"operator": self.operator_name, "component": self.component},
            )
            candidates.append(
                PolicyTransition(
                    incumbent=incumbent,
                    candidate=candidate,
                    round_id=round_id,
                    candidate_index=index,
                    improvement_operator=self.operator_name,
                    edit_type=self.component,
                    seed=seed,
                    config_id=config_id,
                )
            )
        return candidates
