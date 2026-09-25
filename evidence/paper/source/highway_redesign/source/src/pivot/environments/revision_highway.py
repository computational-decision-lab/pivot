"""Headless HighwayEnv physical-response benchmark.

The benchmark keeps a seed-matched reference trajectory for IDM/MOBIL traffic.
Actor evaluations leave the native reactive traffic policies enabled; observer
evaluations replay the recorded peer controls and restore peer state after each
physics frame.  This makes the proxy/deployment distinction explicit while
keeping paired rollouts at the same seed and horizon.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult

HighwayMode = Literal["observer", "actor", "strategic"]
InterventionPhase = Literal["early", "late"]


@dataclass(frozen=True)
class HighwayPhysicalConfig:
    horizon: int = 20
    lanes_count: int = 4
    vehicles_count: int = 20
    vehicles_density: float = 1.0
    policy_frequency: int = 1
    simulation_frequency: int = 5
    duration: float | None = None
    initial_lane_id: int | None = None
    action_hold_steps: int | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.lanes_count < 2:
            raise ValueError("lanes_count must be at least 2")
        if self.vehicles_count <= 0:
            raise ValueError("vehicles_count must be positive")
        if not 0.0 < self.vehicles_density <= 1.0:
            raise ValueError("vehicles_density must be in (0, 1]")
        if self.policy_frequency <= 0 or self.simulation_frequency <= 0:
            raise ValueError("frequencies must be positive")
        if self.simulation_frequency % self.policy_frequency:
            raise ValueError("simulation_frequency must be divisible by policy_frequency")
        if self.duration is not None and (
            not math.isfinite(self.duration) or self.duration <= 0
        ):
            raise ValueError("duration must be a positive finite number")
        if self.initial_lane_id is not None and not (
            isinstance(self.initial_lane_id, int)
            and not isinstance(self.initial_lane_id, bool)
            and 0 <= self.initial_lane_id < self.lanes_count
        ):
            raise ValueError("initial_lane_id must be an integer within the lane range")
        if self.action_hold_steps is not None and not (
            isinstance(self.action_hold_steps, int)
            and not isinstance(self.action_hold_steps, bool)
            and 1 <= self.action_hold_steps <= self.horizon
        ):
            raise ValueError("action_hold_steps must be an integer in [1, horizon]")

    @property
    def episode_duration(self) -> float:
        return (
            float(self.duration)
            if self.duration is not None
            else float(self.horizon / self.policy_frequency)
        )

    @property
    def effective_action_hold_steps(self) -> int:
        """Number of decision steps for which the candidate action is applied."""

        return self.horizon if self.action_hold_steps is None else self.action_hold_steps


@dataclass(frozen=True)
class _HighwayTrace:
    peer_states: tuple[tuple[dict[str, Any], ...], ...]
    peer_snapshots: tuple[tuple[tuple[float, ...], ...], ...]
    peer_actions: tuple[tuple[tuple[float, float], ...], ...]
    focal_snapshots: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    physical_frames: int
    render_mode: str
    terminated_steps: tuple[int, ...]
    initial_available_actions: tuple[int, ...]
    replay_max_abs_error: float = 0.0


class HighwayPhysicalWorld:
    """CPU-only physical reactive gate backed by HighwayEnv IDM/MOBIL traffic."""

    environment_id = "highwayenv_physical_response"
    environment_family = "external_physical_driving"
    paired_query_cost_unit = "highway_decision_steps"

    def __init__(self, config: HighwayPhysicalConfig | None = None) -> None:
        self.config = config or HighwayPhysicalConfig()
        try:
            importlib.import_module("highway_env")
            importlib.import_module("gymnasium")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "HighwayPhysicalWorld requires highway-env==1.12.1; "
                "install requirements/revision-highway.txt"
            ) from error
        self._version = importlib.metadata.version("highway-env")
        self._closed = False

    @property
    def paired_query_cost(self) -> float:
        return float(2 * 2 * self.config.horizon)

    def policy_action(self, policy: Policy) -> int:
        """Map a core policy to HighwayEnv's five discrete meta-actions."""

        intensity = float(policy.parameters.get("intensity", 0.0))
        bias = float(policy.parameters.get("bias", 0.0))
        if bias < -0.5:
            return 0  # LANE_LEFT
        if bias > 0.5:
            return 2  # LANE_RIGHT
        if intensity > 0.5:
            return 3  # FASTER
        if intensity < -0.5:
            return 4  # SLOWER
        return 1  # IDLE

    def evaluate(
        self,
        policy: Policy,
        *,
        seed: int,
        mode: HighwayMode = "observer",
    ) -> RolloutResult:
        if self._closed:
            raise RuntimeError("HighwayPhysicalWorld is closed")
        if mode not in ("observer", "actor", "strategic"):
            raise ValueError(f"unsupported physical evaluation mode: {mode}")

        reference = self._rollout(seed=int(seed), focal_action=1, reference=None)
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
        traffic_control = (
            "fixed_reference_replay" if mode == "observer" else "reactive_idm_mobil"
        )
        metadata: dict[str, object] = {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "environment_version": self._version,
            "mode": mode,
            "seed": int(seed),
            "horizon": self.config.horizon,
            "reference_steps": len(reference.rewards),
            "evaluated_steps": len(evaluated.rewards),
            "policy_frequency": self.config.policy_frequency,
            "simulation_frequency": self.config.simulation_frequency,
            "physical_frames": reference.physical_frames + evaluated.physical_frames,
            "paired_query_cost": self.paired_query_cost,
            "paired_query_cost_unit": self.paired_query_cost_unit,
            "simulator_call_definition": "one_reference_rollout_plus_one_evaluated_rollout",
            "traffic_policy": "highway_env.vehicle.behavior.IDMVehicle",
            "traffic_response_model": "IDM_longitudinal_plus_MOBIL_lateral",
            "traffic_control": traffic_control,
            "counterfactual": "seed_matched_equal_horizon_peer_action_and_state_replay",
            "focal_action": self.policy_action(policy),
            "initial_lane_id": self.config.initial_lane_id,
            "initial_available_actions": list(evaluated.initial_available_actions),
            "reference_initial_available_actions": list(reference.initial_available_actions),
            "action_hold_steps": self.config.effective_action_hold_steps,
            "action_schedule": (
                "repeated"
                if self.config.effective_action_hold_steps == self.config.horizon
                else "initial_intervention_then_idle"
            ),
            "focal_trajectory_sha256": self._trace_hash(evaluated.focal_snapshots),
            "peer_trajectory_sha256": self._trace_hash(evaluated.peer_snapshots),
            "reference_peer_trajectory_sha256": self._trace_hash(reference.peer_snapshots),
            "peer_action_l1_from_reference": peer_action_l1,
            "peer_trajectory_l2_from_reference": peer_trajectory_l2,
            "fixed_world_replay_max_abs_error": evaluated.replay_max_abs_error,
            "peer_count": len(reference.peer_snapshots[0]),
            "render_mode": evaluated.render_mode,
            "headless": evaluated.render_mode == "none",
            "reference_terminated_steps": reference.terminated_steps,
            "evaluated_terminated_steps": evaluated.terminated_steps,
            "claim_scope": "DEV_technical_gate_only_no_confirmatory_claim",
            "response_proof": "peer_action_and_physical_trajectory_divergence_at_fixed_seed_and_horizon",
        }
        return RolloutResult(
            value=float(sum(evaluated.rewards) / len(evaluated.rewards)),
            environment_steps=decision_steps,
            simulator_calls=2,
            compute_cost=float(reference.physical_frames + evaluated.physical_frames),
            metadata=metadata,
        )

    def evaluate_scheduled(
        self,
        policy: Policy,
        *,
        seed: int,
        mode: HighwayMode = "observer",
        action_phase: InterventionPhase,
    ) -> RolloutResult:
        """Evaluate one fixed half-horizon intervention schedule.

        This additive interface is used by the B=4 redesign.  The incumbent
        reference remains IDLE for the whole horizon; a candidate action is
        applied in either the first or second half and IDLE is applied in the
        other half.
        """

        if self._closed:
            raise RuntimeError("HighwayPhysicalWorld is closed")
        if mode not in ("observer", "actor", "strategic"):
            raise ValueError(f"unsupported physical evaluation mode: {mode}")
        if action_phase not in ("early", "late"):
            raise ValueError("action_phase must be 'early' or 'late'")
        if self.config.horizon < 2 or self.config.horizon % 2:
            raise ValueError("scheduled evaluation requires an even horizon")

        reference = self._rollout(seed=int(seed), focal_action=1, reference=None)
        evaluated = self._rollout(
            seed=int(seed),
            focal_action=self.policy_action(policy),
            reference=reference if mode == "observer" else None,
            intervention_phase=action_phase,
        )
        peer_action_l1 = self._action_l1(evaluated.peer_actions, reference.peer_actions)
        peer_trajectory_l2 = self._trajectory_l2(
            evaluated.peer_snapshots, reference.peer_snapshots
        )
        half = self.config.horizon // 2
        metadata: dict[str, object] = {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "environment_version": self._version,
            "mode": mode,
            "seed": int(seed),
            "horizon": self.config.horizon,
            "reference_steps": len(reference.rewards),
            "evaluated_steps": len(evaluated.rewards),
            "policy_frequency": self.config.policy_frequency,
            "simulation_frequency": self.config.simulation_frequency,
            "physical_frames": reference.physical_frames + evaluated.physical_frames,
            "paired_query_cost": self.paired_query_cost,
            "paired_query_cost_unit": self.paired_query_cost_unit,
            "simulator_call_definition": "one_reference_rollout_plus_one_evaluated_rollout",
            "traffic_policy": "highway_env.vehicle.behavior.IDMVehicle",
            "traffic_response_model": "IDM_longitudinal_plus_MOBIL_lateral",
            "traffic_control": (
                "fixed_reference_replay" if mode == "observer" else "reactive_idm_mobil"
            ),
            "counterfactual": "seed_matched_equal_horizon_peer_action_and_state_replay",
            "focal_action": self.policy_action(policy),
            "initial_lane_id": self.config.initial_lane_id,
            "initial_available_actions": list(evaluated.initial_available_actions),
            "reference_initial_available_actions": list(reference.initial_available_actions),
            "action_hold_steps": self.config.effective_action_hold_steps,
            "action_schedule": f"{action_phase}_intervention_then_idle",
            "intervention_phase": action_phase,
            "intervention_start": 0 if action_phase == "early" else half,
            "intervention_end": half if action_phase == "early" else self.config.horizon,
            "focal_trajectory_sha256": self._trace_hash(evaluated.focal_snapshots),
            "peer_trajectory_sha256": self._trace_hash(evaluated.peer_snapshots),
            "reference_peer_trajectory_sha256": self._trace_hash(reference.peer_snapshots),
            "peer_action_l1_from_reference": peer_action_l1,
            "peer_trajectory_l2_from_reference": peer_trajectory_l2,
            "fixed_world_replay_max_abs_error": evaluated.replay_max_abs_error,
            "peer_count": len(reference.peer_snapshots[0]),
            "render_mode": evaluated.render_mode,
            "headless": evaluated.render_mode == "none",
            "reference_terminated_steps": reference.terminated_steps,
            "evaluated_terminated_steps": evaluated.terminated_steps,
            "claim_scope": "DEV_technical_gate_only_no_confirmatory_claim",
            "response_proof": "peer_action_and_physical_trajectory_divergence_at_fixed_seed_and_horizon",
        }
        return RolloutResult(
            value=float(sum(evaluated.rewards) / len(evaluated.rewards)),
            environment_steps=len(reference.rewards) + len(evaluated.rewards),
            simulator_calls=2,
            compute_cost=float(reference.physical_frames + evaluated.physical_frames),
            metadata=metadata,
        )

    def close(self) -> None:
        self._closed = True

    def _make_env(self, seed: int) -> Any:
        gymnasium = importlib.import_module("gymnasium")
        config = {
            "lanes_count": self.config.lanes_count,
            "vehicles_count": self.config.vehicles_count,
            "vehicles_density": self.config.vehicles_density,
            "controlled_vehicles": 1,
            "duration": self.config.episode_duration,
            "policy_frequency": self.config.policy_frequency,
            "simulation_frequency": self.config.simulation_frequency,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "offroad_terminal": False,
            "normalize_reward": True,
            "show_trajectories": False,
            "real_time_rendering": False,
            "manual_control": False,
        }
        if self.config.initial_lane_id is not None:
            config["initial_lane_id"] = self.config.initial_lane_id
        env = gymnasium.make("highway-v0", render_mode=None, config=config)
        env.reset(seed=seed)
        return env

    def _rollout(
        self,
        *,
        seed: int,
        focal_action: int,
        reference: _HighwayTrace | None,
        intervention_phase: str = "repeated",
    ) -> _HighwayTrace:
        env = self._make_env(seed)
        unwrapped = env.unwrapped
        initial_available_actions = tuple(
            int(action) for action in unwrapped.action_type.get_available_actions()
        )
        peers = self._ordered_peers(unwrapped)
        if not peers:
            env.close()
            raise RuntimeError("Highway physical gate requires at least one traffic vehicle")
        if reference is not None and len(peers) != len(reference.peer_snapshots[0]):
            env.close()
            raise RuntimeError("peer count changed across paired HighwayEnv rollouts")

        frames_per_decision = self.config.simulation_frequency // self.config.policy_frequency
        expected_frames = self.config.horizon * frames_per_decision
        frame_index = 0
        peer_states: list[tuple[dict[str, Any], ...]] = []
        peer_snapshots: list[tuple[tuple[float, ...], ...]] = []
        peer_actions: list[tuple[tuple[float, float], ...]] = []
        focal_snapshots: list[tuple[float, ...]] = []
        rewards: list[float] = []
        terminated_steps: list[int] = []
        replay_error = 0.0
        original_act = unwrapped.road.act
        original_step = unwrapped.road.step

        def road_act() -> None:
            nonlocal frame_index
            if reference is None:
                original_act()
            else:
                if frame_index >= len(reference.peer_actions):
                    raise RuntimeError("reference traffic action sequence exhausted")
                for vehicle, action in zip(
                    peers, reference.peer_actions[frame_index], strict=True
                ):
                    vehicle.action = {
                        "steering": float(action[0]),
                        "acceleration": float(action[1]),
                    }
                unwrapped.vehicle.act()

        def road_step(dt: float) -> None:
            nonlocal frame_index, replay_error
            original_step(dt)
            if frame_index >= expected_frames:
                raise RuntimeError("HighwayEnv produced more physics frames than configured")
            actions = tuple(
                (float(vehicle.action["steering"]), float(vehicle.action["acceleration"]))
                for vehicle in peers
            )
            if reference is not None:
                expected_states = reference.peer_states[frame_index]
                raw_snapshot = self._peer_snapshot(peers)
                replay_error = max(
                    replay_error,
                    self._max_snapshot_error(raw_snapshot, reference.peer_snapshots[frame_index]),
                )
                for vehicle, state in zip(peers, expected_states, strict=True):
                    self._restore_peer(vehicle, state)
                snapshot = self._peer_snapshot(peers)
                states = tuple(copy.deepcopy(state) for state in expected_states)
            else:
                snapshot = self._peer_snapshot(peers)
                states = tuple(self._capture_peer(vehicle) for vehicle in peers)
            peer_actions.append(actions)
            peer_snapshots.append(snapshot)
            peer_states.append(states)
            frame_index += 1

        unwrapped.road.act = road_act
        unwrapped.road.step = road_step
        try:
            for decision_step in range(self.config.horizon):
                if intervention_phase == "repeated":
                    action = (
                        focal_action
                        if decision_step < self.config.effective_action_hold_steps
                        else 1
                    )
                elif intervention_phase == "early":
                    action = focal_action if decision_step < self.config.horizon // 2 else 1
                elif intervention_phase == "late":
                    action = 1 if decision_step < self.config.horizon // 2 else focal_action
                else:
                    raise ValueError(f"unsupported intervention phase: {intervention_phase}")
                _, reward, terminated, truncated, _ = env.step(action)
                if terminated:
                    terminated_steps.append(len(rewards) + 1)
                if truncated and len(rewards) + 1 < self.config.horizon:
                    raise RuntimeError("HighwayEnv truncated before the configured horizon")
                rewards.append(float(reward))
                focal_snapshots.append(self._vehicle_snapshot(unwrapped.vehicle))
        finally:
            env.close()

        if len(rewards) != self.config.horizon or frame_index != expected_frames:
            raise RuntimeError(
                f"HighwayEnv rollout length mismatch: decisions={len(rewards)}, "
                f"frames={frame_index}, expected_frames={expected_frames}"
            )
        return _HighwayTrace(
            peer_states=tuple(peer_states),
            peer_snapshots=tuple(peer_snapshots),
            peer_actions=tuple(peer_actions),
            focal_snapshots=tuple(focal_snapshots),
            rewards=tuple(rewards),
            physical_frames=frame_index,
            render_mode="none",
            terminated_steps=tuple(terminated_steps),
            initial_available_actions=initial_available_actions,
            replay_max_abs_error=replay_error,
        )

    @staticmethod
    def _ordered_peers(env: Any) -> tuple[Any, ...]:
        ego = env.vehicle
        return tuple(
            sorted(
                (vehicle for vehicle in env.road.vehicles if vehicle is not ego),
                key=lambda vehicle: (
                    str(vehicle.lane_index[0]),
                    str(vehicle.lane_index[1]),
                    int(vehicle.lane_index[2]),
                    round(float(vehicle.position[0]), 6),
                    round(float(vehicle.position[1]), 6),
                ),
            )
        )

    @staticmethod
    def _capture_peer(vehicle: Any) -> dict[str, Any]:
        return {
            "position": np.asarray(vehicle.position, dtype=float).copy(),
            "heading": float(vehicle.heading),
            "speed": float(vehicle.speed),
            "lane_index": copy.deepcopy(vehicle.lane_index),
            "target_lane_index": copy.deepcopy(getattr(vehicle, "target_lane_index", vehicle.lane_index)),
            "target_speed": float(getattr(vehicle, "target_speed", vehicle.speed)),
            "crashed": bool(vehicle.crashed),
            "action": copy.deepcopy(vehicle.action),
            "timer": float(getattr(vehicle, "timer", 0.0)),
        }

    @staticmethod
    def _restore_peer(vehicle: Any, state: dict[str, Any]) -> None:
        vehicle.position = np.asarray(state["position"], dtype=float).copy()
        vehicle.heading = float(state["heading"])
        vehicle.speed = float(state["speed"])
        vehicle.lane_index = copy.deepcopy(state["lane_index"])
        if hasattr(vehicle, "target_lane_index"):
            vehicle.target_lane_index = copy.deepcopy(state["target_lane_index"])
        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = float(state["target_speed"])
        vehicle.crashed = bool(state["crashed"])
        vehicle.action = copy.deepcopy(state["action"])
        if hasattr(vehicle, "timer"):
            vehicle.timer = float(state["timer"])

    @staticmethod
    def _vehicle_snapshot(vehicle: Any) -> tuple[float, ...]:
        return (
            float(vehicle.position[0]),
            float(vehicle.position[1]),
            float(vehicle.heading),
            float(vehicle.speed),
            float(vehicle.lane_index[2]),
        )

    @classmethod
    def _peer_snapshot(cls, peers: tuple[Any, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(cls._vehicle_snapshot(vehicle) for vehicle in peers)

    @staticmethod
    def _trace_hash(trace: object) -> str:
        return hashlib.sha256(
            json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    @staticmethod
    def _action_l1(
        actual: tuple[tuple[tuple[float, float], ...], ...],
        reference: tuple[tuple[tuple[float, float], ...], ...],
    ) -> float:
        return float(
            sum(
                abs(actual_value - reference_value)
                for actual_step, reference_step in zip(actual, reference, strict=True)
                for actual_peer, reference_peer in zip(actual_step, reference_step, strict=True)
                for actual_value, reference_value in zip(actual_peer, reference_peer, strict=True)
            )
        )

    @staticmethod
    def _trajectory_l2(
        actual: tuple[tuple[tuple[float, ...], ...], ...],
        reference: tuple[tuple[tuple[float, ...], ...], ...],
    ) -> float:
        return float(
            math.sqrt(
                sum(
                    (actual_value - reference_value) ** 2
                    for actual_step, reference_step in zip(actual, reference, strict=True)
                    for actual_peer, reference_peer in zip(actual_step, reference_step, strict=True)
                    for actual_value, reference_value in zip(actual_peer, reference_peer, strict=True)
                )
            )
        )

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


__all__ = ["HighwayMode", "HighwayPhysicalConfig", "HighwayPhysicalWorld"]
