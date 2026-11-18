from __future__ import annotations

from typing import Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult

from .config import PerformativeConfig


class PerformativeWorld:
    """Small known world with observer, actor, and strategic response modes."""

    def __init__(self, config: PerformativeConfig) -> None:
        self.config = config

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "observer",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        if context_or_seed is None:
            if seed is None:
                raise TypeError("either context_or_seed or seed is required")
            context_or_seed = seed
        context = (
            context_or_seed
            if isinstance(context_or_seed, RolloutContext)
            else RolloutContext(seed=int(context_or_seed), scenario_id=f"seed-{context_or_seed}")
        )
        rng = np.random.default_rng(context.seed)
        state = float(context.initial_state)
        impact_stock = 0.0
        opponent_stock = 0.0
        total = 0.0
        for _ in range(self.config.horizon):
            action = policy.action(state)
            exogenous = context.exogenous_value + float(rng.normal(0.0, self.config.noise_scale))
            response = 0.0
            opponent = 0.0
            if mode in ("actor", "strategic"):
                response = self.config.response_strength * abs(action)
            if mode == "strategic":
                opponent = self.config.competition_strength * float(np.tanh(impact_stock))
            next_state = self.config.decay * state + exogenous + response - opponent
            direct_reward = 1.5 * action - 0.4 * action * action + 0.25 * action * state
            mechanical_penalty = (
                self.config.response_strength * 1.8 * impact_stock * abs(action)
                if mode in ("actor", "strategic")
                else 0.0
            )
            competition_penalty = 1.5 * opponent * abs(action)
            reward = direct_reward - mechanical_penalty - competition_penalty
            total += float(np.clip(reward, -self.config.reward_bound, self.config.reward_bound))
            impact_stock += abs(action)
            opponent_stock += opponent
            state = next_state
        return RolloutResult(
            value=float(np.clip(total, -self.config.reward_bound, self.config.reward_bound)),
            environment_steps=self.config.horizon,
            simulator_calls=1,
            metadata={
                "world_id": self.config.config_id,
                "mode": mode,
                "response_strength": self.config.response_strength,
                "competition_strength": self.config.competition_strength,
                "final_state": state,
                "impact_stock": impact_stock,
                "opponent_stock": opponent_stock,
            },
        )

    def response_distance(self, incumbent: Policy, candidate: Policy) -> float:
        return abs(candidate.parameters.get("intensity", 0.0) - incumbent.parameters.get("intensity", 0.0))
