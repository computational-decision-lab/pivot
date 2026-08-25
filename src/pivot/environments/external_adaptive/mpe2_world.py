from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pivot.core.result import RolloutResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MPE2Config:
    """Version-pinned settings for the public Farama MPE2 adapter."""

    max_cycles: int = 25
    focal_agent: str = "agent_0"
    scenario: str = "simple_adversary_v3"
    environment_version: str = "mpe2-1.1.0/simple_adversary_v3"
    pettingzoo_version: str = "1.27.0"
    action_dim: int = 5
    observation_dim: int = 10
    opponent_strength: float = 1.0
    observer_horizon: int = 3

    def __post_init__(self) -> None:
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        expected_observation_dim = {"simple_adversary_v3": 10, "simple_spread_v3": 18}.get(self.scenario)
        if expected_observation_dim is None:
            raise ValueError("scenario must be simple_adversary_v3 or simple_spread_v3")
        if self.action_dim != 5 or self.observation_dim != expected_observation_dim:
            raise ValueError(f"{self.scenario} requires 5 actions and {expected_observation_dim} observations")
        if not math.isfinite(self.opponent_strength) or self.opponent_strength < 0.0:
            raise ValueError("opponent_strength must be finite and non-negative")
        if self.observer_horizon <= 0 or self.observer_horizon > self.max_cycles:
            raise ValueError("observer_horizon must lie in [1, max_cycles]")


@dataclass(frozen=True)
class MPE2Policy:
    """Small deterministic linear policy over the public MPE2 observation."""

    weights: FloatArray
    bias: FloatArray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if weights.ndim != 2 or bias.ndim != 1 or weights.shape[0] != bias.shape[0]:
            raise ValueError("policy weights and bias have incompatible shapes")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("policy parameters must be finite")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "bias", bias.copy())

    @classmethod
    def random(cls, *, seed: int, observation_dim: int, action_dim: int) -> MPE2Policy:
        if observation_dim <= 0 or action_dim <= 1:
            raise ValueError("policy dimensions are invalid")
        rng = np.random.default_rng(seed)
        return cls(
            weights=np.asarray(rng.normal(0.0, 0.2, size=(action_dim, observation_dim)), dtype=np.float64),
            bias=np.asarray(rng.normal(0.0, 0.05, size=action_dim), dtype=np.float64),
        )

    @property
    def policy_id(self) -> str:
        payload = json.dumps(
            {"weights": self.weights.round(12).tolist(), "bias": self.bias.round(12).tolist()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def logits(self, observation: FloatArray) -> FloatArray:
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        if vector.shape != (self.weights.shape[1],):
            raise ValueError("observation has the wrong dimension")
        return np.asarray(self.weights @ vector + self.bias, dtype=np.float64)

    def action(self, observation: FloatArray) -> int:
        return int(np.argmax(self.logits(observation)))

    def distance(self, other: MPE2Policy) -> float:
        if self.weights.shape != other.weights.shape:
            raise ValueError("policies must have equal shapes")
        return float(np.linalg.norm(self.weights - other.weights) + np.linalg.norm(self.bias - other.bias))


class MPE2World:
    """Thin, independently versioned adapter around Farama's MPE2 world.

    The actor path executes the public environment's transition dynamics. The
    observer path performs only a short-horizon direct replay with frozen
    replay opponents; it is intentionally a cheap proxy and never treated as
    deployment ground truth.
    """

    def __init__(self, config: MPE2Config | None = None) -> None:
        self.config = config or MPE2Config()

    def evaluate_actor(
        self,
        policy: MPE2Policy,
        *,
        seed: int,
        opponent_bias: float = 0.0,
        opponent_family: str = "reactive-observation",
    ) -> RolloutResult:
        env = self._make_env()
        observations, _ = env.reset(seed=int(seed))
        total = 0.0
        steps = 0
        try:
            while env.agents and steps < self.config.max_cycles:
                actions = {
                    agent: self._opponent_action(
                        agent, observations[agent], seed, steps, opponent_bias, opponent_family
                    )
                    for agent in env.agents
                }
                if self.config.focal_agent in env.agents:
                    actions[self.config.focal_agent] = policy.action(observations[self.config.focal_agent])
                observations, rewards, terminations, truncations, _ = env.step(actions)
                total += float(rewards.get(self.config.focal_agent, 0.0))
                steps += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
        finally:
            env.close()
        return RolloutResult(
            value=total,
            environment_steps=steps,
            simulator_calls=1,
            metadata=self._metadata("actor", seed, steps, opponent_bias, opponent_family),
        )

    def evaluate_observer(self, policy: MPE2Policy, *, seed: int) -> RolloutResult:
        env = self._make_env()
        observations, _ = env.reset(seed=int(seed))
        total = 0.0
        steps = 0
        try:
            while env.agents and steps < self.config.observer_horizon:
                observation = observations[self.config.focal_agent]
                action = policy.action(observation)
                actions = {
                    agent: self._replay_action(agent, observations[agent], seed, steps)
                    for agent in env.agents
                }
                actions[self.config.focal_agent] = action
                observations, rewards, terminations, truncations, _ = env.step(actions)
                # The short-horizon direct replay uses the public reward
                # transition but freezes opponents and stops before long-run
                # deployment feedback. It is a cheap, deliberately imperfect
                # proxy for the full actor rollout.
                total += float(rewards.get(self.config.focal_agent, 0.0))
                steps += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
        finally:
            env.close()
        return RolloutResult(
            value=total,
            environment_steps=steps,
            simulator_calls=1,
            metadata=self._metadata("observer", seed, steps, 0.0),
        )

    def _make_env(self) -> Any:
        try:
            from mpe2 import simple_adversary_v3, simple_spread_v3
        except ImportError as error:
            raise RuntimeError(
                "V7 external adapter requires the optional 'external' dependencies: "
                "pettingzoo==1.27.0 and mpe2==1.1.0"
            ) from error
        module = simple_adversary_v3 if self.config.scenario == "simple_adversary_v3" else simple_spread_v3
        return module.parallel_env(max_cycles=self.config.max_cycles, continuous_actions=False)

    def _opponent_action(
        self,
        agent: str,
        observation: FloatArray,
        seed: int,
        step: int,
        opponent_bias: float,
        opponent_family: str,
    ) -> int:
        if agent == self.config.focal_agent:
            return 0
        base = self._replay_action(agent, observation, seed, step)
        if opponent_bias == 0.0:
            return base
        # Independent reactive adaptation: opponents shift toward the action
        # most likely to reduce focal movement as the focal policy changes.
        values = np.asarray(observation, dtype=float)
        if opponent_family == "reactive-observation":
            direction = int(abs(float(values.mean())) * 10) % self.config.action_dim
            trigger = (step + int(seed)) % 2 == 0
        elif opponent_family == "threshold-reactive":
            direction = int(abs(float(values[: min(5, len(values))].sum())) * 7) % self.config.action_dim
            trigger = float(values.mean()) > 0.0
        elif opponent_family == "explicit-best-response":
            focal_signal = int(np.argmax(values[: self.config.action_dim]))
            action_values = np.asarray(
                [
                    -abs(action - focal_signal) * self.config.opponent_strength
                    + 0.01 * action
                    for action in range(self.config.action_dim)
                ],
                dtype=np.float64,
            )
            direction = int(np.argmax(action_values))
            trigger = True
        else:
            raise ValueError(
                "opponent_family must be reactive-observation, threshold-reactive, "
                "or explicit-best-response"
            )
        return direction if (trigger and opponent_bias > 0.0) else base

    def _replay_action(self, agent: str, observation: FloatArray, seed: int, step: int) -> int:
        values = np.asarray(observation, dtype=np.float64)
        score = float(values.sum()) + 0.013 * (seed + step + len(agent))
        return int(abs(math.floor(score * 1000.0))) % self.config.action_dim

    def _metadata(
        self, mode: str, seed: int, steps: int, opponent_bias: float, opponent_family: str = "none"
    ) -> dict[str, object]:
        return {
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "opponent_bias": opponent_bias,
            "opponent_family": opponent_family,
            "environment_source": f"Farama-MPE2-{self.config.scenario}",
            "environment_version": self.config.environment_version,
            "pettingzoo_version": self.config.pettingzoo_version,
            "focal_agent": self.config.focal_agent,
            "observer_proxy": "short_horizon_direct_replay_reward",
            "observer_horizon": self.config.observer_horizon,
        }


def generate_mpe2_candidates(
    incumbent: MPE2Policy, *, count: int, seed: int, scale: float
) -> tuple[MPE2Policy, ...]:
    if count <= 0 or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("candidate count and scale must be positive")
    rng = np.random.default_rng(seed)
    candidates: list[MPE2Policy] = []
    for _ in range(count):
        weights = incumbent.weights + rng.normal(0.0, scale, size=incumbent.weights.shape)
        bias = incumbent.bias + rng.normal(0.0, scale, size=incumbent.bias.shape)
        candidates.append(MPE2Policy(weights, bias))
    return tuple(candidates)
