from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


@dataclass(frozen=True)
class SyntheticPerturbation:
    """Deterministic one-component edits for controlling update footprint.

    The operator deliberately does not inspect a world.  This keeps the E1/E2
    phenomenon test independent of a learning algorithm and makes the update
    scale an explicit registered variable.
    """

    component: str = "intensity"
    operator_name: str = "synthetic"

    def propose(
        self,
        incumbent: Policy,
        scale: float,
        num_candidates: int = 1,
        seed: int = 0,
        round_id: int | str = 0,
        config_id: str | None = None,
    ) -> list[PolicyTransition]:
        if scale < 0 or not np.isfinite(scale):
            raise ValueError("scale must be finite and non-negative")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        rng = np.random.default_rng(seed)
        base = incumbent.parameters.get(self.component, 0.0)
        candidates: list[PolicyTransition] = []
        # Small deterministic jitter prevents duplicate policy identities when
        # several candidates share a scale, while preserving the registered
        # scale as the dominant footprint control.
        offsets = np.linspace(-0.5, 0.5, num_candidates) if num_candidates > 1 else [0.0]
        for index, offset in enumerate(offsets):
            jitter = float(rng.normal(0.0, max(scale, 1e-12) * 1e-3))
            value = min(0.95, max(-0.95, base + scale * (1.0 + 0.05 * offset) + jitter))
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
