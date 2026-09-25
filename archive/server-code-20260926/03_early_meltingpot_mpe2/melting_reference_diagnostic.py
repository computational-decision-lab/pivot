#!/usr/bin/env python3
"""Official fixed stag/hare Specialist behavior controls; evaluation only.

Runs stag-stag, hare-hare and stag-hare with separate recurrent states. Only
RGB, own INVENTORY and READY_TO_SHOOT enter the official puppeteer; it adds
GOAL. Model input signatures are checked before evaluating. No downloads,
training, scenario selection, or changes to the existing environment wrapper.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import statistics
import sys
import tarfile
import time
import csv

from melting_behavior_diagnostic import (
    diagnostic_parallel_class, event_metrics, histogram_stats, plain, sha, write_json,
)


SUBSTRATE = "stag_hunt_in_the_matrix__repeated"
MODEL_PREFIX = f"assets/saved_models/{SUBSTRATE}/puppet_1/"
MODEL_FILES = ("saved_model.pb", "variables/variables.index", "variables/variables.data-00000-of-00001")
ALLOWED_OBSERVATIONS = frozenset(("RGB", "INVENTORY", "READY_TO_SHOOT"))
MATCHUPS = (("stag", "stag"), ("hare", "hare"), ("stag", "hare"))


def seed_for(seed, *parts):
    raw = json.dumps([seed, "melting_reference_diagnostic_v1", *parts]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little") % (2**31 - 1)


def model_member_name(name):
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe archive member path")
    if name.startswith(MODEL_PREFIX) and name[len(MODEL_PREFIX):] in MODEL_FILES:
        return name
    return None


def extract_model(archive, output):
    """Extract only the three named regular model files, never archive links."""
    found = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = model_member_name(member.name)
            if name is None:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Model asset must be a regular file")
            if name in found or member.size > 128 * 1024 * 1024:
                raise ValueError("Duplicate or unexpectedly large model asset")
            found.add(name)
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, destination.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(chunk)
    expected = {MODEL_PREFIX + name for name in MODEL_FILES}
    if found != expected:
        raise ValueError(f"Archive missing model files: {sorted(expected - found)}")
    return output / MODEL_PREFIX


def reference_parallel_class(base):
    observed_base = diagnostic_parallel_class(base)

    class ReferenceParallel(observed_base):
        def observations(self, timestep):
            import numpy as np
            # This whitelist is the sole path from native observations to bots.
            self.local_policy_timesteps = [
                timestep._replace(
                    # The pinned official model declares both inputs LiteralNone.
                    # Native rewards remain available separately from env.step().
                    reward=None,
                    discount=None,
                    observation={key: np.array(observation[key], copy=True) for key in ALLOWED_OBSERVATIONS},
                ) for i, observation in enumerate(timestep.observation)
            ]
            return super().observations(timestep)

    return ReferenceParallel


def model_observation_specs(policy):
    """Inspect the existing official permissive-model signature, not weights."""
    function = policy._puppet._model.step
    arguments = function.canonical_arguments.arguments
    timesteps = []

    def visit(value):
        if hasattr(value, "observation"):
            timesteps.append(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(arguments)
    if len(timesteps) != 1:
        raise ValueError("Cannot identify the official model timestep input for observation audit")
    specs = timesteps[0].observation
    if timesteps[0].reward is not None or timesteps[0].discount is not None:
        raise ValueError("Expected the pinned reference model to ignore reward and discount inputs")
    required = {key for key, spec in specs.items() if spec is not None}
    forbidden = required - (ALLOWED_OBSERVATIONS | {"GOAL"})
    if forbidden:
        raise ValueError(f"Official model requires observations outside the allowed set: {sorted(forbidden)}")
    return {"required_observation_keys": sorted(required), "reward_and_discount_inputs": "LiteralNone",
            "observation_specs": {str(key): repr(value) for key, value in specs.items()}}


class OfficialSpecialistRollout:
    """One frozen official network, reusable across fresh per-player states.

    episode returns (row, raw_events). The caller owns high-level choices,
    training accounting, evaluation separation, and any policy learning.
    """
    def __init__(self, model_path, task="melting_stag", horizon=2500, crossbench_root=None):
        if task != "melting_stag":
            raise ValueError("Only the requested repeated stag-hunt reference is supported")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            os.environ[key] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if crossbench_root:
            sys.path.insert(0, str(Path(crossbench_root).resolve()))
        import numpy as np
        import tensorflow as tf
        from meltingpot import bot
        from meltingpot.utils.policies import puppet_policy, saved_model_policy
        from meltingpot.utils.puppeteers import in_the_matrix
        from crossbench.melting import Parallel
        self.np, self.tf = np, tf
        self.horizon, self.task = int(horizon), task
        self.model_path = Path(model_path).resolve()
        self._closed = False
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            threading_note = "TensorFlow inter/intra-op threads set to one"
        except RuntimeError:
            threading_note = "TensorFlow already initialized; retained existing thread configuration"
        self._env_class = reference_parallel_class(Parallel)
        probe = self._env_class(task, self.horizon)
        try:
            settings = probe.cfg.lab2d_settings_builder(roles=probe.cfg.default_player_roles, config=probe.cfg)
            native_cap = int(settings["maxEpisodeLengthFrames"])
            if self.horizon > native_cap:
                raise ValueError("horizon exceeds the native maximum")
            action_table = plain(probe.cfg.action_set)
        finally:
            probe.close()
        self.metadata = {"task": task, "horizon": self.horizon, "native_maximum_frames": native_cap,
                         "model_path": str(self.model_path), "model_file_sha256": self.checkpoint_hashes(),
                         "tensorflow_version": tf.__version__, "threading_note": threading_note,
                         "action_table": action_table, "network_instances": 1,
                         "allowed_policy_observations": sorted(ALLOWED_OBSERVATIONS),
                         "puppeteer_added_observation": "GOAL", "bots": {}}
        self._lowlevel = saved_model_policy.SavedModelPolicy(str(self.model_path))
        self.policies = {}
        try:
            for name in ("stag", "hare"):
                bot_name = f"{SUBSTRATE}__puppet_{name}_margin_0"
                config = dataclasses.replace(bot.get_config(bot_name), model_path=str(self.model_path))
                puppeteer = config.puppeteer_builder()
                if type(puppeteer) is not in_the_matrix.Specialist:
                    raise ValueError("Only exact fixed Specialist puppeteers are allowed")
                policy = puppet_policy.PuppetPolicy(puppeteer=puppeteer, puppet=self._lowlevel)
                self.policies[name] = policy
                self.metadata["bots"][name] = {"bot_name": bot_name, "model_path": str(self.model_path),
                                               "puppeteer_class": type(puppeteer).__name__,
                                               **model_observation_specs(policy)}
        except Exception:
            self._lowlevel.close()
            self._closed = True
            raise

    def checkpoint_hashes(self):
        return {name: sha(self.model_path / name) for name in MODEL_FILES}

    def episode(self, matchup, env_seed, policy_seeds, deadline, allowance):
        if self._closed:
            raise RuntimeError("Rollout engine is closed")
        if len(matchup) != 2 or any(name not in self.policies for name in matchup):
            raise ValueError("matchup must contain two stag/hare identities")
        if len(policy_seeds) != 2 or allowance <= 0:
            raise ValueError("Provide two policy seeds and a positive frame allowance")
        return evaluate_matchup(self._env_class(self.task, self.horizon), self.policies,
                                tuple(matchup), int(env_seed), list(policy_seeds), self.horizon,
                                deadline, int(allowance), self.tf, self.np)

    def close(self):
        if not self._closed:
            self._lowlevel.close()
            self._closed = True


def initial_state(policy, seed, tf, np):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    state = policy.initial_state()
    # Official PuppetPolicy state = (puppeteer_state, (random_key, recurrent_state)).
    key = state[1][0]
    key_value = key.numpy().tolist() if hasattr(key, "numpy") else plain(key)
    return state, key_value


def evaluate_matchup(env, policies, matchup, seed, policy_seeds, horizon, deadline, allowance, tf, np):
    env.reset(seed=seed)
    states, keys = [], []
    for i, name in enumerate(matchup):
        state, key = initial_state(policies[name], policy_seeds[i], tf, np)
        states.append(state); keys.append(key)
    n_actions = env.action_space("agent_0").n
    counts = [[0] * n_actions for _ in range(2)]
    returns = [0.0, 0.0]
    nonzero_frames = [0, 0]
    events = []
    trajectory = hashlib.sha256()
    steps, complete, end_reason = 0, False, "frame_cap"
    begin = time.monotonic()
    try:
        while steps < horizon:
            if time.monotonic() >= deadline:
                end_reason = "wall_time_budget"; break
            if steps >= allowance:
                end_reason = "environment_step_budget"; break
            actions = {}
            for i, name in enumerate(matchup):
                timestep = env.local_policy_timesteps[i]
                if set(timestep.observation) != ALLOWED_OBSERVATIONS:
                    raise AssertionError("Policy observation whitelist changed")
                value, states[i] = policies[name].step(timestep, states[i])
                value = int(value)
                if value < 0 or value >= n_actions:
                    raise ValueError("Invalid bot action")
                counts[i][value] += 1
                actions[f"agent_{i}"] = value
                for key in sorted(timestep.observation):
                    trajectory.update(timestep.observation[key].tobytes())
            _, rewards, term, _, _ = env.step(actions)
            steps += 1
            trajectory.update(json.dumps([actions, rewards], sort_keys=True).encode())
            for i in range(2):
                reward = float(rewards[f"agent_{i}"])
                if not math.isfinite(reward):
                    raise ValueError("Non-finite native reward")
                returns[i] += reward
                nonzero_frames[i] += int(reward != 0)
            events.extend({"frame": steps, "raw": event} for event in env.frame_events)
            if not env.agents:
                complete = True
                end_reason = "native_termination" if any(term.values()) else "frame_cap"
                break
        if steps == horizon:
            complete = True
        row = {"task": "melting_stag", "train_seed": None, "eval_mode": "official_specialist_reference",
               "policy": "reference_" + "_".join(matchup), "focal_policy": matchup[0], "response_policy": matchup[1],
               "matchup": "-".join(matchup), "horizon_mode": str(horizon), "frame_cap": horizon,
               "evaluation_seed": seed, "policy_initialization_seeds": policy_seeds,
               "policy_initial_random_keys": keys, "env_steps": steps,
               "complete": complete, "end_reason": end_reason,
               "focal_return": returns[0], "response_return": returns[1],
               "focal_native_return_per_1000_frames": returns[0] * 1000 / steps if steps else None,
               "nonzero_reward_frames": {f"agent_{i}": nonzero_frames[i] for i in range(2)},
               "actions": {f"agent_{i}": histogram_stats(counts[i]) for i in range(2)},
               "trajectory_sha256": trajectory.hexdigest(), "seconds": round(time.monotonic() - begin, 4),
               **event_metrics(events)}
        return row, events
    finally:
        env.close()


def save(output, rows, protocol, status, reason, begin, error):
    write_json(output / "episodes.json", rows)
    fields = ["task", "eval_mode", "policy", "matchup", "horizon_mode", "evaluation_seed", "env_steps",
              "complete", "end_reason", "focal_return", "response_return", "interaction_count",
              "first_interaction_frame", "focal_resources", "response_resources", "parse_errors"]
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields if key in row}
            flat.update(focal_resources=row["resource_collection_totals"]["agent_0"],
                        response_resources=row["resource_collection_totals"]["agent_1"],
                        parse_errors=len(row["event_parse_errors"]))
            writer.writerow(flat)
    groups = []
    for matchup in MATCHUPS:
        label = "reference_" + "_".join(matchup)
        group = [row for row in rows if row["policy"] == label and row["complete"] and not row["event_parse_errors"]]
        if group:
            groups.append({"policy": label, "matchup": "-".join(matchup), "horizon_mode": str(protocol["horizon"]),
                           "complete_episodes": len(group), "mean_native_return": statistics.mean(r["focal_return"] for r in group),
                           "mean_response_return": statistics.mean(r["response_return"] for r in group),
                           "mean_interaction_count": statistics.mean(r["interaction_count"] for r in group),
                           "episode_fraction_with_interaction": sum(r["interaction_count"] > 0 for r in group) / len(group)})
    summary = {"status": status, "stop_reason": reason, "error": error, "task": "melting_stag",
               "eval_mode": "official_specialist_reference", "new_training_steps": 0,
               "evaluation_steps": sum(r["env_steps"] for r in rows), "seconds": round(time.monotonic() - begin, 4),
               "policy_summaries": groups, "event_parse_error_count": sum(len(r["event_parse_errors"]) for r in rows),
               "model_file_sha256": protocol.get("model_file_sha256"), "limits": protocol["limits"],
               "interpretation": "Fixed official policy behavior controls only. No trained-versus-untrained comparison, response learning, PIVOT result, or statistical success claim."}
    write_json(output / "summary.json", summary)
    return summary


def run(args):
    begin = time.monotonic(); deadline = begin + args.max_seconds
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.crossbench_root:
        sys.path.insert(0, str(args.crossbench_root.resolve()))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    plan = [{"episode": index, "evaluation_seed": seed_for(args.seed, "environment", index),
             "policy_initialization_seeds": [seed_for(args.seed, "policy", index, role) for role in (0, 1)]}
            for index in range(args.episodes)]
    protocol = {"purpose": "official_specialist_behavior_control_no_training", "task": "melting_stag",
                "eval_mode": "official_specialist_reference", "horizons": [str(args.horizon)], "horizon": args.horizon,
                "matchups": ["-".join(pair) for pair in MATCHUPS], "episodes_per_matchup": args.episodes,
                "seed_plan": plan, "source_sha256": sha(__file__),
                "behavior_utility_sha256": sha(Path(__file__).with_name("melting_behavior_diagnostic.py")),
                "allowed_policy_observations": sorted(ALLOWED_OBSERVATIONS), "puppeteer_added_observation": "GOAL",
                "limits": {"max_seconds": args.max_seconds, "max_env_steps": args.max_env_steps},
                "seed_note": "Environment seeds paired across matchups. Python, NumPy and TensorFlow seeds reset before each player's initial_state; resulting explicit model random keys recorded. Full reproducibility requires trajectory replay on this pinned runtime.",
                "new_training_steps": 0, "checkpoint_manifest": None,
                "native_duration_note": "Fixed official Specialist behavior control at the stated frame cap; raw local RGB and recurrent official model differ from the existing pooled-RGB PPO baseline."}
    write_json(output / "protocol.json", protocol)
    rows, engine, error = [], None, None
    status, reason = "complete", "planned_episodes_complete"
    try:
        if args.assets_archive:
            protocol["assets_archive"] = str(args.assets_archive.resolve())
            protocol["assets_archive_sha256"] = sha(args.assets_archive)
            model_path = extract_model(args.assets_archive, output / "reference_assets")
        else:
            model_path = args.model_path.resolve()
        model_hashes = {name: sha(model_path / name) for name in MODEL_FILES}
        protocol.update(model_path=str(model_path), model_file_sha256=model_hashes)
        engine = OfficialSpecialistRollout(model_path, horizon=args.horizon, crossbench_root=args.crossbench_root)
        protocol.update(engine.metadata)
        write_json(output / "protocol.json", protocol)
        with (output / "events.jsonl").open("w") as detail:
            for item in plan:
                for matchup in MATCHUPS:
                    remaining = args.max_env_steps - sum(row["env_steps"] for row in rows)
                    if time.monotonic() >= deadline:
                        status, reason = "budget_stopped", "wall_time_budget"; break
                    if remaining < args.horizon:
                        status, reason = "budget_stopped", "insufficient_steps_for_complete_episode"; break
                    row, events = engine.episode(matchup, item["evaluation_seed"],
                                                 item["policy_initialization_seeds"], deadline, remaining)
                    rows.append(row)
                    for event in events:
                        detail.write(json.dumps({"policy": row["policy"], "evaluation_seed": row["evaluation_seed"], **event}, allow_nan=False) + "\n")
                    detail.flush(); write_json(output / "episodes.json", rows)
                    print(json.dumps({key: row[key] for key in ("policy", "evaluation_seed", "env_steps", "complete", "focal_return", "response_return", "interaction_count", "first_interaction_frame")}), flush=True)
                    if not row["complete"]:
                        status, reason = "budget_stopped", row["end_reason"]; break
                if status != "complete":
                    break
        if any(row["event_parse_errors"] for row in rows):
            status, reason = "invalid_event_metrics", "raw_events_retained_for_parser_review"
        if model_hashes != {name: sha(model_path / name) for name in MODEL_FILES}:
            status, reason = "failed", "model_file_hash_changed"
    except Exception as exc:
        status, reason, error = "failed", "reference_evaluation_error", f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            engine.close()
        write_json(output / "protocol.json", protocol)
        summary = save(output, rows, protocol, status, reason, begin, error)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 1 if status in ("failed", "invalid_event_metrics") else 0


def self_test():
    from types import SimpleNamespace
    from collections import namedtuple
    assert model_member_name(MODEL_PREFIX + "saved_model.pb") == MODEL_PREFIX + "saved_model.pb"
    assert model_member_name("assets/irrelevant") is None
    for name in ("../saved_model.pb", "/tmp/saved_model.pb", MODEL_PREFIX + "../saved_model.pb"):
        try:
            model_member_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe archive path accepted")
    assert seed_for(4, "policy", 0, 0) != seed_for(4, "policy", 0, 1)
    assert seed_for(4, "environment", 0) == seed_for(4, "environment", 0)
    assert "INTERACTION_INVENTORIES" not in ALLOWED_OBSERVATIONS and "WORLD.RGB" not in ALLOWED_OBSERVATIONS
    timestep_type = namedtuple("TimeStep", "step_type reward discount observation")
    observations = {"RGB": "tensor", "GOAL": "tensor", "INVENTORY": "tensor", "READY_TO_SHOOT": "tensor",
                    "WORLD.RGB": None, "INTERACTION_INVENTORIES": None}
    timestep = timestep_type("tensor", None, None, observations)
    function = SimpleNamespace(canonical_arguments=SimpleNamespace(arguments={"args": ("random_key", timestep, "state")}))
    policy = SimpleNamespace(_puppet=SimpleNamespace(_model=SimpleNamespace(step=function)))
    assert set(model_observation_specs(policy)["required_observation_keys"]) == ALLOWED_OBSERVATIONS | {"GOAL"}
    observations["WORLD.RGB"] = "tensor"
    try:
        model_observation_specs(policy)
    except ValueError:
        pass
    else:
        raise AssertionError("Privileged model input accepted")
    print("PASS: asset paths, matchup plan, separate seeds, nested model signature audit, literal-None exclusion, forbidden input rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--assets-archive", type=Path)
    sources.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crossbench-root", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-env-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.output is None or not (args.assets_archive or args.model_path):
        parser.error("--output and either --assets-archive or --model-path are required")
    if min(args.episodes, args.horizon, args.max_seconds, args.max_env_steps) <= 0:
        parser.error("Episode, frame and runtime budgets must be positive")
    if args.max_seconds > 7200:
        parser.error("This diagnostic is limited to at most 7200 seconds")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
