from __future__ import annotations

from typing import Literal

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult
from pivot.environments.interactive_market.world import InteractiveMarketWorld

from .config import StrategicMarketConfig


class StrategicMarketWorld:
    """F4 wrapper with fixed, reactive, or finite-step adaptive opponents."""

    world_id = "F4-strategic-market"

    def __init__(self, config: StrategicMarketConfig) -> None:
        self.config = config
        self.actor_world = InteractiveMarketWorld(config.interactive)

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "strategic",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        base = self.actor_world.evaluate(policy, context_or_seed, mode="actor", seed=seed)
        if mode != "strategic" or self.config.opponent_mode == "fixed" or self.config.market_share_sensitivity == 0.0:
            metadata = dict(base.metadata)
            metadata.update({"world_id": self.world_id, "opponent_mode": self.config.opponent_mode, "opponent_skill": 0.0, "opponent_context": {"mode": "fixed"}})
            return RolloutResult(base.value, base.environment_steps, base.simulator_calls, base.compute_cost, metadata)
        action = abs(policy.action(0.0) * policy.parameters.get("position_size", 1.0))
        response = self.config.market_share_sensitivity * action * self.config.opponent_count
        skill = 0.0
        if self.config.opponent_mode == "adaptive":
            for _ in range(self.config.adaptation_steps):
                skill += self.config.learning_rate * (response - skill)
            response += skill
        # Strategic response is a consequence of the focal update, not an
        # authorization filter.  The penalty is logged separately so the
        # mechanical actor value remains recoverable.
        competition_penalty = response * action * max(1, base.environment_steps)
        metadata = dict(base.metadata)
        metadata.update(
            {
                "world_id": self.world_id,
                "opponent_mode": self.config.opponent_mode,
                "opponent_skill": skill,
                "opponent_response": response,
                "opponent_context": {
                    "opponent_id": self.config.opponent_id,
                    "opponent_count": self.config.opponent_count,
                    "adaptation_steps": self.config.adaptation_steps,
                    "learning_rate": self.config.learning_rate,
                    "market_share_sensitivity": self.config.market_share_sensitivity,
                },
            }
        )
        return RolloutResult(
            value=float(base.value - competition_penalty),
            environment_steps=base.environment_steps,
            simulator_calls=base.simulator_calls,
            compute_cost=base.compute_cost,
            metadata=metadata,
        )

    def response_distance(self, incumbent: Policy, candidate: Policy) -> float:
        return self.actor_world.response_distance(incumbent, candidate)

    def strategic_sensitivity(self, incumbent: Policy, candidate: Policy) -> float:
        footprint = self.response_distance(incumbent, candidate)
        response = self.config.market_share_sensitivity * self.config.opponent_count
        if self.config.opponent_mode == "adaptive":
            response *= 1.0 + self.config.learning_rate * self.config.adaptation_steps
        return float(response * footprint / max(footprint, 1e-12))
