from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult

WorldMode = Literal["observer", "actor", "strategic"]


@dataclass(frozen=True)
class V9WorldConfig:
    horizon: int = 16
    noise_scale: float = 0.04
    response_strength: float = 0.8
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.noise_scale < 0 or self.response_strength < 0:
            raise ValueError("invalid V9 world configuration")


class PerformativeControlWorld:
    """Independent continuous response world for E2C/E3C.

    Observer evaluation freezes response stock. Actor evaluation lets deployed
    actions change the stock and therefore subsequent state/reward. Strategic
    mode adds an adaptive response term but remains optional for E7C.
    """

    environment_id = "performative_control"
    environment_family = "performative_control"

    def __init__(self, config: V9WorldConfig | None = None) -> None:
        self.config = config or V9WorldConfig()

    def evaluate(self, policy: Policy, *, seed: int, mode: WorldMode = "observer") -> RolloutResult:
        rng = np.random.default_rng(int(seed) + self.config.seed_offset)
        state = float(rng.normal(0.0, 0.35))
        response_stock = 0.0
        total = 0.0
        for step in range(self.config.horizon):
            action = policy.action(state)
            shock = float(rng.normal(0.0, self.config.noise_scale))
            response = 0.0
            if mode in ("actor", "strategic"):
                response = self.config.response_strength * (0.55 * abs(action) + 0.1 * response_stock)
            strategic_drag = 0.0
            if mode == "strategic":
                strategic_drag = 0.20 * np.tanh(response_stock / 2.0) * abs(action)
            target = 0.45 + 0.12 * math.sin(0.35 * step) + shock
            direct_reward = 1.7 * action - 0.85 * action * action + 0.32 * action * state
            impact_penalty = 0.0 if mode == "observer" else 0.62 * response_stock * abs(action)
            total += direct_reward - impact_penalty - strategic_drag
            next_state = 0.72 * state + target + response - 0.08 * abs(action)
            response_stock += abs(action) if mode != "observer" else 0.0
            state = float(next_state)
        return RolloutResult(
            value=float(total / self.config.horizon),
            environment_steps=self.config.horizon,
            simulator_calls=1,
            metadata={
                "environment_id": self.environment_id,
                "environment_family": self.environment_family,
                "mode": mode,
                "response_strength": self.config.response_strength,
                "response_stock": response_stock,
            },
        )


class CongestionResourceWorld:
    """Independent queue/load response world for cross-environment tests."""

    environment_id = "congestion_resource"
    environment_family = "congestion_resource"

    def __init__(self, config: V9WorldConfig | None = None) -> None:
        self.config = config or V9WorldConfig(response_strength=0.65)

    def evaluate(self, policy: Policy, *, seed: int, mode: WorldMode = "observer") -> RolloutResult:
        rng = np.random.default_rng(int(seed) + self.config.seed_offset + 17)
        queue = float(rng.uniform(0.15, 0.45))
        total = 0.0
        service = 0.62
        for _ in range(self.config.horizon):
            state = queue - 0.4
            action = max(0.0, policy.action(state))
            noise = float(rng.normal(0.0, self.config.noise_scale))
            load = 0.18 + 0.70 * action
            endogenous_load = load if mode in ("actor", "strategic") else 0.18
            queue = max(0.0, queue + endogenous_load - service + noise)
            throughput = min(service, queue + load)
            latency_cost = queue * queue
            strategic_cost = 0.0
            if mode == "strategic":
                strategic_cost = 0.22 * action * (1.0 + queue)
            total += 1.8 * throughput - 0.75 * latency_cost - 0.12 * action - strategic_cost
        return RolloutResult(
            value=float(total / self.config.horizon),
            environment_steps=self.config.horizon,
            simulator_calls=1,
            metadata={
                "environment_id": self.environment_id,
                "environment_family": self.environment_family,
                "mode": mode,
                "response_strength": self.config.response_strength,
                "final_queue": queue,
            },
        )


class FrozenMPE2World:
    """Version-pinned MPE2 adapter used as the preserved external family.

    The adapter intentionally exposes the V7 observer/actor distinction and
    does not relabel the V7 null as a positive result. It requires the pinned
    optional MPE2/PettingZoo dependency at runtime.
    """

    environment_id = "mpe2_frozen"
    environment_family = "mpe2_frozen_external"

    def __init__(self, *, scenario: str = "simple_adversary_v3", max_cycles: int = 25) -> None:
        from pivot.environments.external_adaptive.mpe2_world import MPE2Config, MPE2World

        self._world = MPE2World(
            MPE2Config(
                scenario=scenario,
                max_cycles=max_cycles,
                observer_horizon=min(3, max_cycles),
                observation_dim=10 if scenario == "simple_adversary_v3" else 18,
            )
        )

    def evaluate(self, policy: Policy, *, seed: int, mode: WorldMode = "observer") -> RolloutResult:
        from pivot.environments.external_adaptive.mpe2_world import MPE2Policy

        rng = np.random.default_rng(int(seed) + 811)
        base = MPE2Policy.random(seed=int(seed) + 17, observation_dim=self._world.config.observation_dim, action_dim=5)
        bias = base.bias.copy()
        intensity = float(policy.parameters.get("intensity", 0.0))
        bias[0] += intensity
        if "bias" in policy.parameters:
            bias[1] += float(policy.parameters["bias"])
        mapped = MPE2Policy(base.weights, bias)
        if mode == "observer":
            result = self._world.evaluate_observer(mapped, seed=seed)
        else:
            result = self._world.evaluate_actor(mapped, seed=seed, opponent_bias=0.0)
        metadata = dict(result.metadata)
        metadata.update({"environment_id": self.environment_id, "environment_family": self.environment_family})
        # Consume no hidden random outcome; this call only makes the adapter's
        # seed contract explicit for future implementations.
        _ = rng
        return RolloutResult(result.value, result.environment_steps, result.simulator_calls, result.compute_cost, metadata)


def environment_for(environment_id: str, response_strength: float) -> object:
    if environment_id == "performative_control":
        return PerformativeControlWorld(V9WorldConfig(response_strength=response_strength))
    if environment_id == "congestion_resource":
        return CongestionResourceWorld(V9WorldConfig(response_strength=response_strength))
    if environment_id == "mpe2_frozen":
        return FrozenMPE2World()
    raise ValueError(f"unknown V9 environment: {environment_id}")
