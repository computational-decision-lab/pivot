"""Deterministic MPE2 adapter for bounded revision experiments.

The observer and actor use the same seed and horizon. Both first execute a
fixed reference rollout. The observer replays its response-agent actions in
the evaluated rollout, while the actor recomputes actions from current public
observations. The response rule is a deterministic scenario heuristic; it is
not trained, optimized, or presented as a best response.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult

MPE2Scenario = Literal["simple_adversary", "simple_spread"]
MPE2Mode = Literal["observer", "actor"]
FloatArray = NDArray[np.float64]
ActionArray = NDArray[np.float32]

_MPE2_VERSION = "1.1.0"
_PETTINGZOO_VERSION = "1.27.0"
_FOCAL_AGENT = "agent_0"
_OBSERVATION_DIMS = {"simple_adversary": 10, "simple_spread": 18}
_RESPONSE_AGENTS = {
    "simple_adversary": ("adversary_0", "agent_1"),
    "simple_spread": ("agent_1", "agent_2"),
}


@dataclass(frozen=True)
class RevisionMPE2Config:
    """Pinned configuration for the two revision MPE2 technical smokes."""

    scenario: MPE2Scenario = "simple_adversary"
    horizon: int = 25
    focal_agent: str = _FOCAL_AGENT
    continuous_actions: Literal[True] = True

    def __post_init__(self) -> None:
        if self.scenario not in _OBSERVATION_DIMS:
            raise ValueError("scenario must be simple_adversary or simple_spread")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.focal_agent != _FOCAL_AGENT:
            raise ValueError("revision MPE2 requires focal_agent='agent_0'")
        if self.continuous_actions is not True:
            raise ValueError("continuous_actions must remain true for revision MPE2")


@dataclass(frozen=True)
class _RolloutTrace:
    value: float
    steps: int
    focal_actions: tuple[tuple[float, ...], ...]
    response_actions: tuple[tuple[tuple[str, tuple[float, ...]], ...], ...]
    response_observations: tuple[tuple[tuple[str, tuple[float, ...]], ...], ...]
    reward_trace: tuple[float, ...]


class RevisionMPE2World:
    """Run paired reference/replay MPE2 evaluations with auditable response."""

    environment_family = "mpe2_revision_external"
    query_cost_unit = "environment_steps"
    paired_query_cost_is_upper_bound = True

    def __init__(self, config: RevisionMPE2Config | None = None) -> None:
        self.config = config or RevisionMPE2Config()
        self.environment_id = f"mpe2_revision_{self.config.scenario}"
        self.response_agents = _RESPONSE_AGENTS[self.config.scenario]
        try:
            self._module = importlib.import_module(f"mpe2.{self.config.scenario}_v3")
            self._mpe2_version = version("mpe2")
            self._pettingzoo_version = version("pettingzoo")
        except (ModuleNotFoundError, PackageNotFoundError) as error:
            raise ModuleNotFoundError(
                "RevisionMPE2World requires requirements/revision-mpe2.txt"
            ) from error
        if self._mpe2_version != _MPE2_VERSION or self._pettingzoo_version != _PETTINGZOO_VERSION:
            raise RuntimeError(
                "RevisionMPE2World requires mpe2==1.1.0 and pettingzoo==1.27.0; "
                "install requirements/revision-mpe2.txt in an isolated environment"
            )

    @property
    def paired_query_cost(self) -> float:
        """Return the predeclared upper bound for incumbent/candidate HF work."""

        return float(4 * self.config.horizon)

    def actual_paired_query_cost(self, incumbent: RolloutResult, candidate: RolloutResult) -> float:
        """Return measured work for a completed incumbent/candidate HF pair."""

        for result in (incumbent, candidate):
            if result.metadata.get("environment_id") != self.environment_id:
                raise ValueError("paired query result belongs to a different environment")
        return float(incumbent.environment_steps + candidate.environment_steps)

    def policy_action(self, policy: Policy, observation: NDArray[Any]) -> ActionArray:
        """Map policy content and a focal observation to a stable continuous action."""

        vector = self._observation(observation, focal=True)
        target = self._task_vector(self.config.focal_agent, vector)
        intensity = float(policy.parameters.get("intensity", 0.0))
        bias = float(policy.parameters.get("bias", 0.0))
        extra = sum(
            self._parameter_coefficient(key) * value
            for key, value in sorted(policy.parameters.items())
            if key not in {"intensity", "bias"}
        )
        angle = math.pi * math.tanh(intensity + 0.25 * extra)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        shifted = np.asarray(
            [
                cosine * target[0] - sine * target[1] + 0.35 * math.tanh(bias),
                sine * target[0] + cosine * target[1] - 0.20 * math.tanh(bias),
            ],
            dtype=np.float64,
        )
        return self._continuous_action(shifted)

    def evaluate(
        self,
        policy: Policy,
        *,
        seed: int,
        mode: MPE2Mode = "observer",
    ) -> RolloutResult:
        """Evaluate a policy against replayed or observation-reactive responses."""

        if mode not in ("observer", "actor"):
            raise ValueError(f"unsupported MPE2 evaluation mode: {mode}")

        reference = self._rollout(seed=seed, policy=None, replay_actions=None)
        replay_actions = reference.response_actions if mode == "observer" else None
        evaluated = self._rollout(seed=seed, policy=policy, replay_actions=replay_actions)
        if reference.steps != evaluated.steps:
            raise RuntimeError(
                "reference and evaluated MPE2 rollouts did not share one time horizon"
            )

        mismatches, decisions = self._action_difference(reference, evaluated)
        observation_l2 = self._observation_difference(reference, evaluated)
        environment_steps = reference.steps + evaluated.steps
        metadata: dict[str, object] = {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "scenario": self.config.scenario,
            "scenario_module": f"{self.config.scenario}_v3",
            "mpe2_version": self._mpe2_version,
            "pettingzoo_version": self._pettingzoo_version,
            "parallel_api": True,
            "continuous_actions": self.config.continuous_actions,
            "action_space": "continuous_box_0_1_shape_5",
            "action_dtype": "float32",
            "mode": mode,
            "seed": int(seed),
            "horizon": self.config.horizon,
            "reference_steps": reference.steps,
            "evaluated_steps": evaluated.steps,
            "focal_agent": self.config.focal_agent,
            "focal_policy_id": policy.policy_id,
            "focal_controller": "deterministic_policy_parameter_rotation",
            "focal_action_trace_sha256": _trace_hash(evaluated.focal_actions),
            "response_agents": self.response_agents,
            "response_control": (
                "fixed_reference_action_replay"
                if mode == "observer"
                else "observation_reactive_heuristic"
            ),
            "response_rule": "scenario_task_direction_from_current_public_observation",
            "response_claim": "not_optimized_not_best_response",
            "response_intervention": "focal_policy_changes_shared_dynamics_and_observations",
            "reference_response_action_trace_sha256": _trace_hash(reference.response_actions),
            "evaluated_response_action_trace_sha256": _trace_hash(evaluated.response_actions),
            "reference_response_observation_trace_sha256": _trace_hash(
                reference.response_observations
            ),
            "evaluated_response_observation_trace_sha256": _trace_hash(
                evaluated.response_observations
            ),
            "response_action_mismatch_count": mismatches,
            "response_action_decisions": decisions,
            "response_action_mismatch_rate": mismatches / decisions if decisions else 0.0,
            "response_observation_l2_from_reference": observation_l2,
            "reference_reward_trace_sha256": _trace_hash(reference.reward_trace),
            "evaluated_reward_trace_sha256": _trace_hash(evaluated.reward_trace),
            "simulator_call_definition": "one_reference_rollout_plus_one_evaluated_rollout",
            "evaluation_query_cost": float(environment_steps),
            "query_cost_unit": self.query_cost_unit,
            "paired_query_cost_upper_bound": self.paired_query_cost,
            "paired_query_cost_is_upper_bound": self.paired_query_cost_is_upper_bound,
            "claim_scope": "DEV_technical_smoke_only_no_best_response_claim",
        }
        return RolloutResult(
            value=evaluated.value,
            environment_steps=environment_steps,
            simulator_calls=2,
            compute_cost=float(environment_steps),
            metadata=metadata,
        )

    def _rollout(
        self,
        *,
        seed: int,
        policy: Policy | None,
        replay_actions: tuple[tuple[tuple[str, tuple[float, ...]], ...], ...] | None,
    ) -> _RolloutTrace:
        env = self._make_env()
        observations, _ = env.reset(seed=int(seed))
        focal_actions: list[tuple[float, ...]] = []
        response_actions: list[tuple[tuple[str, tuple[float, ...]], ...]] = []
        response_observations: list[tuple[tuple[str, tuple[float, ...]], ...]] = []
        rewards: list[float] = []
        try:
            while env.agents and len(rewards) < self.config.horizon:
                if self.config.focal_agent not in env.agents:
                    raise RuntimeError("focal agent left the MPE2 environment early")
                step = len(rewards)
                response_observations.append(
                    tuple(
                        (
                            agent,
                            tuple(float(value) for value in observations[agent]),
                        )
                        for agent in self.response_agents
                    )
                )
                if replay_actions is None:
                    response = {
                        agent: self._response_action(agent, observations[agent])
                        for agent in self.response_agents
                    }
                else:
                    if step >= len(replay_actions):
                        raise RuntimeError("reference replay ended before evaluated rollout")
                    response = {
                        agent: np.asarray(action, dtype=np.float32)
                        for agent, action in replay_actions[step]
                    }
                focal_action = (
                    self._reference_focal_action(observations[self.config.focal_agent])
                    if policy is None
                    else self.policy_action(policy, observations[self.config.focal_agent])
                )
                actions = {
                    agent: np.asarray(response[agent], dtype=np.float32)
                    for agent in self.response_agents
                }
                actions[self.config.focal_agent] = focal_action
                response_actions.append(
                    tuple(
                        (agent, tuple(float(value) for value in response[agent]))
                        for agent in sorted(response)
                    )
                )
                focal_actions.append(tuple(float(value) for value in focal_action))
                observations, step_rewards, terminations, truncations, _ = env.step(actions)
                rewards.append(float(step_rewards.get(self.config.focal_agent, 0.0)))
                if all(terminations.values()) or all(truncations.values()):
                    break
        finally:
            env.close()
        return _RolloutTrace(
            value=float(sum(rewards)),
            steps=len(rewards),
            focal_actions=tuple(focal_actions),
            response_actions=tuple(response_actions),
            response_observations=tuple(response_observations),
            reward_trace=tuple(rewards),
        )

    def _make_env(self) -> Any:
        return self._module.parallel_env(
            max_cycles=self.config.horizon,
            continuous_actions=self.config.continuous_actions,
            dynamic_rescaling=False,
        )

    def _reference_focal_action(self, observation: NDArray[Any]) -> ActionArray:
        vector = self._observation(observation, focal=True)
        return self._continuous_action(self._task_vector(self.config.focal_agent, vector))

    def _response_action(self, agent: str, observation: NDArray[Any]) -> ActionArray:
        vector = self._observation(observation, focal=False)
        return self._continuous_action(self._task_vector(agent, vector))

    def _task_vector(self, agent: str, observation: FloatArray) -> FloatArray:
        if self.config.scenario == "simple_spread":
            target = _nearest_vector(observation[4:10])
            if agent == self.config.focal_agent:
                return target
            nearest_peer = _nearest_vector(observation[10:14])
            return np.asarray(target - 0.75 * nearest_peer, dtype=np.float64)
        if agent.startswith("adversary_"):
            landmarks = np.asarray(observation[0:4], dtype=np.float64).reshape(-1, 2)
            good_agents = np.asarray(observation[4:8], dtype=np.float64).reshape(-1, 2)
            distances = np.linalg.norm(
                landmarks[:, np.newaxis, :] - good_agents[np.newaxis, :, :],
                axis=2,
            )
            inferred_target = landmarks[int(np.argmin(np.min(distances, axis=1)))]
            nearest_good_agent = _nearest_vector(observation[4:8])
            return np.asarray(
                inferred_target + 0.75 * nearest_good_agent,
                dtype=np.float64,
            )
        goal = np.asarray(observation[0:2], dtype=np.float64)
        if agent == self.config.focal_agent:
            return goal
        nearest_peer = _nearest_vector(observation[6:10])
        return np.asarray(goal - 0.75 * nearest_peer, dtype=np.float64)

    def _observation(self, observation: NDArray[Any], *, focal: bool) -> FloatArray:
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        if focal and vector.shape != (_OBSERVATION_DIMS[self.config.scenario],):
            raise ValueError("focal observation has the wrong dimension")
        if vector.size < 2 or not np.all(np.isfinite(vector)):
            raise ValueError("MPE2 observation must contain finite coordinates")
        return vector

    @staticmethod
    def _continuous_action(vector: FloatArray) -> ActionArray:
        x, y = (float(value) for value in vector[:2])
        scale = max(1.0, abs(x), abs(y))
        normalized_x = x / scale
        normalized_y = y / scale
        return np.asarray(
            [
                0.0,
                max(0.0, -normalized_x),
                max(0.0, normalized_x),
                max(0.0, -normalized_y),
                max(0.0, normalized_y),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _parameter_coefficient(key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return 2.0 * (integer / (2**64 - 1)) - 1.0

    @staticmethod
    def _action_difference(reference: _RolloutTrace, evaluated: _RolloutTrace) -> tuple[int, int]:
        reference_actions = [action for row in reference.response_actions for _, action in row]
        evaluated_actions = [action for row in evaluated.response_actions for _, action in row]
        if len(reference_actions) != len(evaluated_actions):
            raise RuntimeError("response action traces have different lengths")
        mismatches = sum(
            not np.allclose(left, right, rtol=0.0, atol=1e-12)
            for left, right in zip(reference_actions, evaluated_actions, strict=True)
        )
        return mismatches, len(reference_actions)

    @staticmethod
    def _observation_difference(reference: _RolloutTrace, evaluated: _RolloutTrace) -> float:
        reference_values = np.asarray(
            [
                value
                for row in reference.response_observations
                for _, observation in row
                for value in observation
            ],
            dtype=np.float64,
        )
        evaluated_values = np.asarray(
            [
                value
                for row in evaluated.response_observations
                for _, observation in row
                for value in observation
            ],
            dtype=np.float64,
        )
        if reference_values.shape != evaluated_values.shape:
            raise RuntimeError("response observation traces have different shapes")
        return float(np.linalg.norm(reference_values - evaluated_values))


def _nearest_vector(values: NDArray[Any]) -> FloatArray:
    vectors = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return np.asarray(vectors[np.argmin(np.linalg.norm(vectors, axis=1))], dtype=np.float64)


def _trace_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
