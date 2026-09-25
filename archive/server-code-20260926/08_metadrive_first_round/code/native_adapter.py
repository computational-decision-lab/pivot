"""Native MetaDrive bottleneck evaluation; no learning or selection logic.

Native environment overrides (all listed here):
  num_agents=8, horizon=300, use_render=False, image_observation=False,
  allow_respawn=False, force_seed_spawn_manager=True, random_agent_model=False,
  store_map=False (destroy retired native maps instead of retaining them),
  force_destroy=True (recreate native vehicle bodies; pooled reuse was nondeterministic),
  agent_policy=IDMPolicy, start_seed=0, num_scenarios=100_000,
  log_level=logging.WARNING, and spawn_roads=[Road(FirstPGBlock.NODE_2,
  FirstPGBlock.NODE_3)].  Only the positive bottleneck approach is populated;
  the native map, physics, reward, routing and termination are unchanged.

Caller may override num_agents, horizon, start_seed, num_scenarios, log_level,
  and map_config. Other config keys are rejected. Native defaults supply any
  map or reward keys not explicitly overridden and are recorded in metadata.

Every run requires all four controller parameters: speed (native km/h),
  headway (native TIME_WANTED), distance (native DISTANCE_WANTED), and
  lane_change_distance (native SAFE_LANE_CHANGE_DISTANCE). No defaults are
  silently filled for these scientific parameters. Parameters are assigned
  to policy INSTANCES after each reset, never to the IDMPolicy class.

The whole initial roster is followed to native termination or the fixed
  horizon, including after agent0 finishes. Background return is divided by
  the initial background roster size, not by the number of surviving cars.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any, Dict

import numpy as np


PARAMETER_KEYS = ("speed", "headway", "distance", "lane_change_distance")


def _agent_sort_key(name: str):
    return (int(name[5:]) if name.startswith("agent") and name[5:].isdigit() else math.inf, name)


def _validated_params(values: dict, label: str) -> Dict[str, float]:
    if set(values) != set(PARAMETER_KEYS):
        raise ValueError(f"{label} requires exactly {PARAMETER_KEYS}; got {sorted(values)}")
    ret = {key: float(values[key]) for key in PARAMETER_KEYS}
    if not all(math.isfinite(value) and value > 0 for value in ret.values()):
        raise ValueError(f"{label} parameters must be finite and strictly positive: {ret}")
    return ret


class NativeEvaluator:
    """One reusable native MetaDrive environment; one evaluator per process."""

    def __init__(self, config: dict):
        from metadrive.component.pgblock.first_block import FirstPGBlock
        from metadrive.component.road_network import Road
        from metadrive.envs.marl_envs.marl_bottleneck import MultiAgentBottleneckEnv
        from metadrive.policy.idm_policy import IDMPolicy
        from metadrive.version import VERSION

        permitted = {"num_agents", "horizon", "start_seed", "num_scenarios", "log_level", "map_config"}
        unknown = set(config) - permitted
        if unknown:
            raise ValueError(f"Unknown evaluator config keys: {sorted(unknown)}")
        native_config = {
            "num_agents": 8,
            "horizon": 300,
            "use_render": False,
            "image_observation": False,
            "allow_respawn": False,
            "store_map": False,
            "force_destroy": True,
            "force_seed_spawn_manager": True,
            "random_agent_model": False,
            "agent_policy": IDMPolicy,
            "start_seed": 0,
            "num_scenarios": 100_000,
            "log_level": logging.WARNING,
            "spawn_roads": [Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)],
        }
        native_config.update(config)
        if int(native_config["num_agents"]) != native_config["num_agents"] or native_config["num_agents"] < 2:
            raise ValueError("num_agents must be an integer >= 2")
        if int(native_config["horizon"]) != native_config["horizon"] or native_config["horizon"] <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = int(native_config["horizon"])
        self.num_agents = int(native_config["num_agents"])
        self.start_seed = int(native_config["start_seed"])
        self.num_scenarios = int(native_config["num_scenarios"])
        self.env = MultiAgentBottleneckEnv(native_config)
        self.closed = False
        self.version = VERSION
        self.config_metadata = {
            **{key: value for key, value in native_config.items() if key not in ("agent_policy", "spawn_roads")},
            "agent_policy": "metadrive.policy.idm_policy.IDMPolicy",
            "spawn_roads": [[FirstPGBlock.NODE_2, FirstPGBlock.NODE_3]],
            "native_map_config": dict(self.env.config["map_config"]),
            "native_reward_and_done_config": {
                key: self.env.config.get(key) for key in (
                    "driving_reward", "speed_reward", "success_reward", "use_lateral_reward",
                    "out_of_road_penalty", "crash_vehicle_penalty", "crash_object_penalty",
                    "crash_done", "crash_vehicle_done", "out_of_road_done", "delay_done",
                    "cross_yellow_line_done", "truncate_as_terminate", "decision_repeat",
                    "physics_world_step_size",
                )
            },
        }
        # Fail early if metadata contains a non-serializable unexpected upstream value.
        json.dumps(self.config_metadata, allow_nan=False)

    @staticmethod
    def _set_policy_params(policy: Any, params: Dict[str, float]) -> None:
        policy.NORMAL_SPEED = params["speed"]
        policy.target_speed = params["speed"]
        policy.TIME_WANTED = params["headway"]
        policy.DISTANCE_WANTED = params["distance"]
        policy.SAFE_LANE_CHANGE_DISTANCE = params["lane_change_distance"]

    def run_episode(self, seed: int, ego_params: dict, bg_params: dict) -> dict:
        if self.closed:
            raise RuntimeError("NativeEvaluator has been closed")
        if int(seed) != seed or not self.start_seed <= seed < self.start_seed + self.num_scenarios:
            raise ValueError("seed must be an integer inside [start_seed, start_seed + num_scenarios)")
        ego_params = _validated_params(ego_params, "ego_params")
        bg_params = _validated_params(bg_params, "bg_params")
        started = time.monotonic()
        self.env.reset(seed=int(seed))
        initial_ids = sorted(self.env.agents, key=_agent_sort_key)
        if len(initial_ids) != self.num_agents or "agent0" not in initial_ids:
            raise RuntimeError(f"Unexpected initial native agent roster: {initial_ids}")
        initial_positions = {
            agent_id: [float(x) for x in self.env.agents[agent_id].position]
            for agent_id in initial_ids
        }
        initial_lanes = {
            agent_id: list(self.env.agents[agent_id].lane.index) for agent_id in initial_ids
        }
        initial_longitudinals = {
            agent_id: float(self.env.agents[agent_id].lane.local_coordinates(
                self.env.agents[agent_id].position)[0]) for agent_id in initial_ids
        }
        # All agents share the same native approach, so the lane-local
        # longitudinal coordinate gives their initial order toward the merge.
        ego_longitudinal = initial_longitudinals["agent0"]
        initial_geometry = {
            "ego_longitudinal": ego_longitudinal,
            "ego_front_rank_1_is_frontmost": 1 + sum(
                longitudinal > ego_longitudinal for agent_id, longitudinal in initial_longitudinals.items()
                if agent_id != "agent0"
            ),
            "background_agents_ahead": sum(
                longitudinal > ego_longitudinal for agent_id, longitudinal in initial_longitudinals.items()
                if agent_id != "agent0"
            ),
            "ego_distance_to_approach_end": float(
                self.env.agents["agent0"].lane.length - ego_longitudinal
            ),
            "background_agents_within_10m_longitudinal": sum(
                abs(longitudinal - ego_longitudinal) <= 10.0
                for agent_id, longitudinal in initial_longitudinals.items() if agent_id != "agent0"
            ),
        }
        returns = {agent_id: 0.0 for agent_id in initial_ids}
        flags = {
            agent_id: {"crash": False, "arrive_dest": False, "out_of_road": False,
                       "terminated": False, "truncated": False}
            for agent_id in initial_ids
        }
        for agent_id in initial_ids:
            policy = self.env.engine.get_policy(self.env.agents[agent_id].name)
            if policy is None:
                raise RuntimeError(f"Missing native policy for {agent_id}")
            self._set_policy_params(policy, ego_params if agent_id == "agent0" else bg_params)
        trajectory = hashlib.sha256()
        trajectory.update(json.dumps(initial_positions, sort_keys=True).encode())
        actual_agent_steps = 0
        steps = 0
        for step in range(self.horizon):
            current_ids = sorted(self.env.agents, key=_agent_sort_key)
            if not current_ids:
                break
            # IDMPolicy computes native actions; these placeholders satisfy the
            # environment's agent-key action interface and are not driving commands.
            _, rewards, terminated, truncated, infos = self.env.step({
                agent_id: [0.0, 0.0] for agent_id in current_ids
            })
            steps = step + 1
            actual_agent_steps += len(rewards)
            if not set(rewards).issubset(returns):
                raise RuntimeError(f"Unexpected respawn/new agent: {set(rewards) - set(returns)}")
            for agent_id, reward in rewards.items():
                returns[agent_id] += float(reward)
                info = infos.get(agent_id, {})
                for flag in ("crash", "arrive_dest", "out_of_road"):
                    flags[agent_id][flag] |= bool(info.get(flag, False))
                flags[agent_id]["terminated"] |= bool(terminated.get(agent_id, False))
                flags[agent_id]["truncated"] |= bool(truncated.get(agent_id, False))
            state = {
                "step": steps,
                "rewards": {agent_id: float(rewards[agent_id]) for agent_id in sorted(rewards, key=_agent_sort_key)},
                "positions": {
                    agent_id: [float(x) for x in self.env.agents[agent_id].position]
                    for agent_id in sorted(self.env.agents, key=_agent_sort_key)
                },
            }
            trajectory.update(json.dumps(state, sort_keys=True, allow_nan=False).encode())
            if bool(terminated.get("__all__", False)) or bool(truncated.get("__all__", False)):
                break
        result = {
            "seed": int(seed),
            "ego_params": ego_params,
            "background_params": bg_params,
            "ego_return": returns["agent0"],
            "background_mean_return": sum(value for agent_id, value in returns.items() if agent_id != "agent0") / (len(initial_ids) - 1),
            "per_agent_returns": returns,
            "per_agent_flags": flags,
            "ego_crash": flags["agent0"]["crash"],
            "ego_arrival": flags["agent0"]["arrive_dest"],
            "ego_out_of_road": flags["agent0"]["out_of_road"],
            "steps": steps,
            "actual_agent_steps": actual_agent_steps,
            "initial_agent_ids": initial_ids,
            "initial_positions": initial_positions,
            "initial_lanes": initial_lanes,
            "initial_longitudinals": initial_longitudinals,
            "initial_geometry": initial_geometry,
            "trajectory_sha256": trajectory.hexdigest(),
            "seconds": time.monotonic() - started,
            "metadrive_version": self.version,
        }
        if not np.isfinite(list(returns.values())).all():
            raise RuntimeError(f"Non-finite native return: {returns}")
        json.dumps(result, allow_nan=False)
        return result

    def close(self) -> None:
        if not self.closed:
            self.env.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
