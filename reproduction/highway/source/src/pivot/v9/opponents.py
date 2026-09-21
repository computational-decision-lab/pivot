from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pivot.core.policy import Policy

from .environments import PerformativeControlWorld

OPPONENT_MODES = ("fixed", "reactive", "best_response", "gradient_adaptive", "rl_evolutionary")


@dataclass(frozen=True)
class Opponent:
    mode: str
    strength: float
    seed: int

    def __post_init__(self) -> None:
        if self.mode not in OPPONENT_MODES:
            raise ValueError(f"unknown opponent mode: {self.mode}")
        if self.strength < 0:
            raise ValueError("opponent strength must be non-negative")

    def strategic_value(self, policy: Policy, *, seed: int) -> float:
        """Evaluate a focal policy under an independently adapting response."""

        world = PerformativeControlWorld()
        actor = world.evaluate(policy, seed=seed + self.seed, mode="actor").value
        if self.mode == "fixed":
            return float(actor)
        action = abs(policy.action(0.0))
        base = self.strength * (0.45 * action + 0.02 * ((self.seed + seed) % 7))
        if self.mode == "reactive":
            response = base
        elif self.mode == "best_response":
            response = base * (1.0 + 0.65 * action)
        elif self.mode == "gradient_adaptive":
            response = base * (1.0 + 0.4 * np.tanh(action + self.strength))
        else:
            rng = np.random.default_rng(self.seed + seed)
            response = base * (1.0 + 0.35 * rng.random()) * (1.0 + 0.2 * action)
        return float(actor - response * (1.0 + 0.55 * action))
