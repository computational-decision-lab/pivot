from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult

PhysicalMode = Literal["observer", "actor", "strategic"]


@dataclass(frozen=True)
class MetaDrivePhysicalConfig:
    horizon: int = 20
    decision_repeat: int = 5
    traffic_density: float = 0.30
    map_id: str = "S"
    ego_spawn_longitude: float = 35.0
    reference_throttle: float = 0.25

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.decision_repeat <= 0:
            raise ValueError("decision_repeat must be positive")
        if not 0.0 < self.traffic_density <= 1.0:
            raise ValueError("traffic_density must be in (0, 1]")
        if not self.map_id:
            raise ValueError("map_id must not be empty")
        if not math.isfinite(self.ego_spawn_longitude):
            raise ValueError("ego_spawn_longitude must be finite")
        if not -1.0 <= self.reference_throttle <= 1.0:
            raise ValueError("reference_throttle must be in [-1, 1]")


@dataclass(frozen=True)
class _TrafficTrace:
    peer_states: tuple[tuple[dict[str, Any], ...], ...]
    peer_snapshots: tuple[tuple[tuple[float, ...], ...], ...]
    peer_actions: tuple[tuple[tuple[float, float], ...], ...]
    focal_snapshots: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    render_mode: str
    terminated_steps: tuple[int, ...]
    replay_max_abs_error: float = 0.0


class _OpenLoopTrafficPolicy:
    """Minimal MetaDrive policy protocol for a fixed sequence of peer controls."""

    name = "OpenLoopTrafficPolicy"

    def __init__(self, actions: tuple[tuple[float, float], ...]) -> None:
        self._actions = actions
        self._cursor = 0
        self.action_info: dict[str, object] = {}

    def act(self) -> list[float]:
        if self._cursor >= len(self._actions):
            raise RuntimeError("open-loop traffic action sequence exhausted")
        action = self._actions[self._cursor]
        self._cursor += 1
        self.action_info["action"] = action
        return [action[0], action[1]]

    def get_action_info(self) -> dict[str, object]:
        return copy.deepcopy(self.action_info)

    def reset(self) -> None:
        self._cursor = 0
        self.action_info.clear()

    def destroy(self) -> None:
        self.action_info.clear()


class MetaDrivePhysicalWorld:
    """Headless physical-response gate using MetaDrive's reactive IDM traffic.

    Every result includes two equal-horizon simulator calls. The first creates a
    seed-matched neutral reference trajectory. Actor and strategic evaluation
    then leave MetaDrive's IDM peer policies active. Observer evaluation instead
    replays the reference peer controls and physical states while applying the
    candidate ego policy. This isolates peer response without changing horizon.
    """

    environment_id = "metadrive_physical_response"
    environment_family = "external_physical_driving"
    paired_query_cost_unit = "metadrive_decision_steps"

    def __init__(self, config: MetaDrivePhysicalConfig | None = None) -> None:
        self.config = config or MetaDrivePhysicalConfig()
        try:
            self._metadrive = importlib.import_module("metadrive")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "MetaDrivePhysicalWorld requires the pinned optional dependency; "
                "install requirements/revision-physical.txt"
            ) from error
        self._version = importlib.metadata.version("metadrive-simulator")
        self._closed = False

    @property
    def paired_query_cost(self) -> float:
        """Decision-step cost of evaluating an incumbent/candidate pair."""
        rollouts_per_evaluation = 2
        policies_per_paired_query = 2
        return float(
            policies_per_paired_query * rollouts_per_evaluation * self.config.horizon
        )

    def policy_action(self, policy: Policy) -> tuple[float, float]:
        """Map a core Policy to one seed-independent MetaDrive control action."""
        steering_source = policy.parameters.get("steering", policy.parameters.get("bias", 0.0))
        throttle_source = policy.parameters.get(
            "throttle_brake", policy.parameters.get("intensity", 0.0)
        )
        steering = self._clip(0.25 * float(steering_source))
        throttle = self._clip(float(throttle_source))
        return steering, throttle

    def evaluate(
        self,
        policy: Policy,
        *,
        seed: int,
        mode: PhysicalMode = "observer",
    ) -> RolloutResult:
        if self._closed:
            raise RuntimeError("MetaDrivePhysicalWorld is closed")
        if mode not in ("observer", "actor", "strategic"):
            raise ValueError(f"unsupported physical evaluation mode: {mode}")

        reference = self._rollout(
            seed=int(seed),
            focal_action=(0.0, self.config.reference_throttle),
            reference=None,
        )
        traffic_control = "open_loop_reference_replay" if mode == "observer" else "reactive_idm"
        evaluated = self._rollout(
            seed=int(seed),
            focal_action=self.policy_action(policy),
            reference=reference if mode == "observer" else None,
        )

        peer_action_l1 = self._action_l1(evaluated.peer_actions, reference.peer_actions)
        peer_trajectory_l2 = self._trajectory_l2(
            evaluated.peer_snapshots, reference.peer_snapshots
        )
        decision_steps = len(reference.rewards) + len(evaluated.rewards)
        physics_steps = decision_steps * self.config.decision_repeat
        metadata: dict[str, object] = {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "environment_version": self._version,
            "mode": mode,
            "seed": int(seed),
            "horizon": self.config.horizon,
            "reference_steps": len(reference.rewards),
            "evaluated_steps": len(evaluated.rewards),
            "decision_repeat": self.config.decision_repeat,
            "physics_steps": physics_steps,
            "paired_query_cost": self.paired_query_cost,
            "paired_query_cost_unit": self.paired_query_cost_unit,
            "simulator_call_definition": "one_reference_rollout_plus_one_evaluated_rollout",
            "traffic_policy": "metadrive.policy.idm_policy.IDMPolicy",
            "traffic_control": traffic_control,
            "counterfactual": "seed_matched_equal_horizon_open_loop_peer_state_and_control_replay",
            "focal_action": self.policy_action(policy),
            "focal_trajectory_sha256": self._trace_hash(evaluated.focal_snapshots),
            "peer_trajectory_sha256": self._trace_hash(evaluated.peer_snapshots),
            "reference_peer_trajectory_sha256": self._trace_hash(reference.peer_snapshots),
            "peer_action_l1_from_reference": peer_action_l1,
            "peer_trajectory_l2_from_reference": peer_trajectory_l2,
            "open_loop_state_replay_max_abs_error": evaluated.replay_max_abs_error,
            "peer_count": len(reference.peer_snapshots[0]),
            "reference_terminated_steps": reference.terminated_steps,
            "evaluated_terminated_steps": evaluated.terminated_steps,
            "render_mode": evaluated.render_mode,
            "headless": evaluated.render_mode == "none",
            "claim_scope": "DEV_technical_gate_only_no_confirmatory_claim",
            "response_proof": (
                "peer_action_and_physical_trajectory_divergence_at_fixed_seed_and_horizon"
            ),
        }
        return RolloutResult(
            value=float(sum(evaluated.rewards) / len(evaluated.rewards)),
            environment_steps=decision_steps,
            simulator_calls=2,
            compute_cost=float(physics_steps),
            metadata=metadata,
        )

    def close(self) -> None:
        self._closed = True

    def _rollout(
        self,
        *,
        seed: int,
        focal_action: tuple[float, float],
        reference: _TrafficTrace | None,
    ) -> _TrafficTrace:
        env_class = self._metadrive.MetaDriveEnv
        env = env_class(self._environment_config(seed))
        peer_states: list[tuple[dict[str, Any], ...]] = []
        peer_snapshots: list[tuple[tuple[float, ...], ...]] = []
        peer_actions: list[tuple[tuple[float, float], ...]] = []
        focal_snapshots: list[tuple[float, ...]] = []
        rewards: list[float] = []
        terminated_steps: list[int] = []
        render_mode = "unknown"
        replay_max_abs_error = 0.0
        try:
            env.reset(seed=seed)
            render_mode = str(env.engine.mode)
            peers = self._ordered_peers(env)
            if not peers:
                raise RuntimeError("MetaDrive physical gate requires at least one traffic vehicle")
            if reference is not None:
                expected_peer_count = len(reference.peer_snapshots[0])
                if len(peers) != expected_peer_count:
                    raise RuntimeError(
                        f"peer count changed across paired rollouts: {len(peers)} != "
                        f"{expected_peer_count}"
                    )
                self._install_open_loop_policies(env, peers, reference.peer_actions)

            for step in range(self.config.horizon):
                _, reward, terminated, truncated, _ = env.step(list(focal_action))
                current_peers = tuple(peers)
                if any(vehicle.name not in env.engine._object_policies for vehicle in current_peers):
                    raise RuntimeError("traffic vehicle changed during bounded physical rollout")
                actions = tuple(
                    self._as_action(env.engine.get_policy(vehicle.name).action_info["action"])
                    for vehicle in current_peers
                )
                if reference is not None:
                    for vehicle, state in zip(
                        current_peers, reference.peer_states[step], strict=True
                    ):
                        vehicle.set_state(copy.deepcopy(state))
                    replay_max_abs_error = max(
                        replay_max_abs_error,
                        self._max_snapshot_error(
                            self._peer_snapshot(current_peers),
                            reference.peer_snapshots[step],
                        ),
                    )
                states = tuple(copy.deepcopy(vehicle.get_state()) for vehicle in current_peers)
                peer_states.append(states)
                peer_snapshots.append(self._peer_snapshot(current_peers))
                peer_actions.append(actions)
                focal_snapshots.append(self._vehicle_snapshot(env.agent))
                rewards.append(float(reward))
                if terminated and step + 1 < self.config.horizon:
                    terminated_steps.append(step + 1)
                if truncated and step + 1 < self.config.horizon:
                    raise RuntimeError("MetaDrive truncated before the configured horizon")
        finally:
            env.close()

        if len(rewards) != self.config.horizon:
            raise RuntimeError(
                f"MetaDrive rollout length {len(rewards)} did not match horizon "
                f"{self.config.horizon}"
            )
        return _TrafficTrace(
            peer_states=tuple(peer_states),
            peer_snapshots=tuple(peer_snapshots),
            peer_actions=tuple(peer_actions),
            focal_snapshots=tuple(focal_snapshots),
            rewards=tuple(rewards),
            render_mode=render_mode,
            terminated_steps=tuple(terminated_steps),
            replay_max_abs_error=replay_max_abs_error,
        )

    def _environment_config(self, seed: int) -> dict[str, object]:
        return {
            "use_render": False,
            "image_observation": False,
            "num_scenarios": 1,
            "start_seed": int(seed),
            "map": self.config.map_id,
            "traffic_density": self.config.traffic_density,
            "random_traffic": False,
            "traffic_mode": "respawn",
            "horizon": self.config.horizon,
            "truncate_as_terminate": False,
            "decision_repeat": self.config.decision_repeat,
            "out_of_route_done": False,
            "out_of_road_done": False,
            "on_continuous_line_done": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "crash_human_done": False,
            "show_sidewalk": False,
            "show_terrain": False,
            "show_skybox": False,
            "log_level": 50,
            "vehicle_config": {"lidar": {"num_lasers": 16, "distance": 30}},
            "agent_configs": {
                "default_agent": {
                    "use_special_color": True,
                    "spawn_lane_index": (">", ">>", 0),
                    "spawn_longitude": self.config.ego_spawn_longitude,
                }
            },
        }

    @staticmethod
    def _install_open_loop_policies(
        env: Any,
        peers: tuple[Any, ...],
        actions_by_step: tuple[tuple[tuple[float, float], ...], ...],
    ) -> None:
        for peer_index, vehicle in enumerate(peers):
            sequence = tuple(step_actions[peer_index] for step_actions in actions_by_step)
            old_policy = env.engine.get_policy(vehicle.name)
            old_policy.destroy()
            env.engine._object_policies[vehicle.name] = _OpenLoopTrafficPolicy(sequence)

    @staticmethod
    def _ordered_peers(env: Any) -> tuple[Any, ...]:
        vehicles = env.engine.traffic_manager.traffic_vehicles
        return tuple(
            sorted(
                vehicles,
                key=lambda vehicle: (
                    int(vehicle.lane_index[-1]),
                    round(float(vehicle.position[0]), 6),
                    round(float(vehicle.position[1]), 6),
                ),
            )
        )

    @classmethod
    def _peer_snapshot(cls, peers: tuple[Any, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(cls._vehicle_snapshot(vehicle) for vehicle in peers)

    @staticmethod
    def _vehicle_snapshot(vehicle: Any) -> tuple[float, ...]:
        position = vehicle.position
        velocity = vehicle.velocity
        return (
            float(position[0]),
            float(position[1]),
            float(vehicle.get_z()),
            float(vehicle.heading_theta),
            float(velocity[0]),
            float(velocity[1]),
        )

    @staticmethod
    def _as_action(action: Any) -> tuple[float, float]:
        if len(action) != 2:
            raise RuntimeError(f"unexpected MetaDrive traffic action: {action!r}")
        return float(action[0]), float(action[1])

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _trace_hash(trace: object) -> str:
        payload = json.dumps(trace, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _action_l1(
        actual: tuple[tuple[tuple[float, float], ...], ...],
        reference: tuple[tuple[tuple[float, float], ...], ...],
    ) -> float:
        return float(
            sum(
                abs(actual_value - reference_value)
                for actual_step, reference_step in zip(actual, reference, strict=True)
                for actual_peer, reference_peer in zip(
                    actual_step, reference_step, strict=True
                )
                for actual_value, reference_value in zip(
                    actual_peer, reference_peer, strict=True
                )
            )
        )

    @staticmethod
    def _trajectory_l2(
        actual: tuple[tuple[tuple[float, ...], ...], ...],
        reference: tuple[tuple[tuple[float, ...], ...], ...],
    ) -> float:
        squared_error = sum(
            (actual_value - reference_value) ** 2
            for actual_step, reference_step in zip(actual, reference, strict=True)
            for actual_peer, reference_peer in zip(actual_step, reference_step, strict=True)
            for actual_value, reference_value in zip(actual_peer, reference_peer, strict=True)
        )
        return float(math.sqrt(squared_error))

    @staticmethod
    def _max_snapshot_error(
        actual: tuple[tuple[float, ...], ...],
        reference: tuple[tuple[float, ...], ...],
    ) -> float:
        return max(
            abs(actual_value - reference_value)
            for actual_peer, reference_peer in zip(actual, reference, strict=True)
            for actual_value, reference_value in zip(actual_peer, reference_peer, strict=True)
        )


__all__ = ["MetaDrivePhysicalConfig", "MetaDrivePhysicalWorld", "PhysicalMode"]
