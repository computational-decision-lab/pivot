#!/usr/bin/env python3
"""Bounded, evaluation-only Melting Pot behavior and checkpoint diagnostic.

Example (in the existing Melting runtime, with crossbench on PYTHONPATH)::

  python melting_behavior_diagnostic.py --run-dir runs/melting_cross_main/\
melting_stag/test/seed_1300 --output diagnostics/seed_1300 \
    --horizons 1000,native --episodes 8 --max-seconds 900 --max-env-steps 96000

No downloads, training, changes to old wrappers, or privileged policy inputs.
Native-duration evaluation is an extrapolation diagnostic: the original
remaining-time feature reaches zero at its training horizon and stays zero.
Use --self-test for dependency-free checks without opening an environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


TASK_ALIASES = {
    "repeated_stag_hunt": "melting_stag", "melting_stag": "melting_stag",
    "stag_hunt_in_the_matrix__repeated": "melting_stag",
    "repeated_prisoners_dilemma": "melting_pd", "melting_pd": "melting_pd",
    "prisoners_dilemma_in_the_matrix__repeated": "melting_pd",
}


def plain(value):
    """Copy event buffers immediately; the environment may reuse them."""
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(plain(k)): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def decode_event(event):
    """Accept DMLab2D mapping payloads and flattened dict-style payloads."""
    event = plain(event)
    if not isinstance(event, list) or len(event) < 2:
        raise ValueError("Expected (event_name, event_payload)")
    name = str(event[0])
    payload = event[1] if len(event) == 2 else event[1:]
    while isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if isinstance(payload, dict):
        return name, payload
    if isinstance(payload, list):
        if payload and payload[0] == "dict":
            payload = payload[1:]
        if all(isinstance(x, list) and len(x) == 2 for x in payload):
            return name, dict(payload)
        if len(payload) % 2 == 0 and all(isinstance(k, str) for k in payload[::2]):
            return name, dict(zip(payload[::2], payload[1::2]))
    raise ValueError("Unsupported event payload; raw event retained")


def scalar_int(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    result = int(value)
    if float(value) != result:
        raise ValueError("Non-integral event index")
    return result


def event_metrics(events):
    counts = Counter()
    collections = {"agent_0": Counter(), "agent_1": Counter()}
    participants = Counter()
    interactions = []
    errors = []
    for record in events:
        try:
            raw = plain(record["raw"])
            if isinstance(raw, list) and raw and raw[0] not in ("collected_resource", "interaction"):
                counts[str(raw[0])] += 1
                continue
            name, payload = decode_event(record["raw"])
            counts[name] += 1
            if name == "collected_resource":
                player, resource = scalar_int(payload["player_index"]), scalar_int(payload["class"])
                if player not in (1, 2) or resource not in (1, 2):
                    raise ValueError("Unexpected player/resource index")
                collections[f"agent_{player - 1}"][str(resource)] += 1
            elif name == "interaction":
                row = scalar_int(payload["row_player_idx"])
                col = scalar_int(payload["col_player_idx"])
                if row not in (1, 2) or col not in (1, 2) or row == col:
                    raise ValueError("Unexpected interaction player indices")
                participants[f"agent_{row - 1}"] += 1
                participants[f"agent_{col - 1}"] += 1
                interactions.append({"frame": record["frame"], **payload})
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"frame": record["frame"], "raw": record["raw"], "error": str(exc)})
    return {
        "event_counts": dict(counts),
        "resource_collection_counts": {a: dict(c) for a, c in collections.items()},
        "resource_collection_totals": {a: sum(c.values()) for a, c in collections.items()},
        "interaction_count": len(interactions),
        "interaction_counts_by_player": {a: participants[a] for a in collections},
        "first_interaction_frame": interactions[0]["frame"] if interactions else None,
        "interactions": interactions, "event_parse_errors": errors,
    }


def histogram_stats(counts):
    total = sum(counts)
    probabilities = [x / total if total else 0.0 for x in counts]
    return {"counts": counts, "frequencies": probabilities,
            "empirical_entropy_nats": -sum(p * math.log(p) for p in probabilities if p)}


def paired_quality(rows):
    groups = {}
    for row in rows:
        if row["complete"]:
            groups.setdefault((row["horizon_mode"], row["evaluation_seed"]), {})[row["policy"]] = row
    pairs = []
    for (mode, seed), group in sorted(groups.items()):
        if {"trained", "untrained"}.issubset(group):
            a, b = group["untrained"], group["trained"]
            pairs.append({"horizon_mode": mode, "evaluation_seed": seed,
                          "native_return_gain": b["focal_return"] - a["focal_return"],
                          "interaction_count_gain": b["interaction_count"] - a["interaction_count"],
                          "focal_collection_gain": b["resource_collection_totals"]["agent_0"] - a["resource_collection_totals"]["agent_0"]})
    summaries = {}
    for mode in sorted({p["horizon_mode"] for p in pairs}):
        gains = [p["native_return_gain"] for p in pairs if p["horizon_mode"] == mode]
        summaries[mode] = {"paired_episodes": len(gains), "mean_native_return_gain": statistics.mean(gains),
                           "paired_episode_gain_std": statistics.stdev(gains) if len(gains) > 1 else None,
                           "positive_gain_fraction": sum(g > 0 for g in gains) / len(gains)}
    return {"pairs": pairs, "by_horizon": summaries,
            "inference_note": "One trained checkpoint pair per invocation. Episode variation is not independent training-seed uncertainty; no superiority claim."}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(plain(obj), indent=2, allow_nan=False) + "\n")


def evaluation_seed(base, mode, index, forbidden):
    attempt = 0
    while True:
        data = f"melting_behavior_diagnostic_v1/{base}/{mode}/{index}/{attempt}".encode()
        value = int.from_bytes(hashlib.sha256(data).digest()[:4], "big") % (2**31 - 2)
        if value not in forbidden:
            forbidden.add(value)
            return value
        attempt += 1


def diagnostic_parallel_class(base):
    class DiagnosticParallel(base):
        """Only observes the existing public event stream; policy obs unchanged."""
        def __init__(self, *args, **kwargs):
            self._event_subscription = None
            self.frame_events = []
            super().__init__(*args, **kwargs)

        def reset(self, seed=None):
            result = super().reset(seed=seed)
            self.frame_events = []
            self._event_subscription = self.env.observables().events.subscribe(
                on_next=lambda event: self.frame_events.append(plain(event)))
            return result

        def step(self, actions):
            self.frame_events = []
            return super().step(actions)

        def close(self):
            if self._event_subscription is not None:
                self._event_subscription.dispose()
                self._event_subscription = None
            super().close()
    return DiagnosticParallel


def evaluate_episode(env, model, opponent, label, mode, seed, cap, time_horizon,
                     deadline, allowance, rng_factory, sample_action):
    import numpy as np
    obs, _ = env.reset(seed=seed)
    rngs = [rng_factory(seed, role) for role in ("focal", "response")]
    action_space = env.action_space("agent_0")
    hist = [[0] * action_space.n for _ in range(2)]
    rewards = [0.0, 0.0]
    reward_frames = [0, 0]
    events = []
    steps = 0
    end_reason = "frame_cap"
    complete = False
    started = time.monotonic()
    try:
        while steps < cap:
            if time.monotonic() >= deadline:
                end_reason = "wall_time_budget"; break
            if steps >= allowance:
                end_reason = "environment_step_budget"; break
            remaining = max(0.0, 1.0 - steps / time_horizon)
            acts = {}
            for i, policy in enumerate((model, opponent)):
                x = np.r_[obs[f"agent_{i}"], remaining].astype(np.float32)
                value = int(sample_action(policy, x, action_space, rngs[i]))
                hist[i][value] += 1
                acts[f"agent_{i}"] = value
            obs, rew, term, trunc, _ = env.step(acts)
            steps += 1
            for i in range(2):
                value = float(rew[f"agent_{i}"])
                if not math.isfinite(value):
                    raise ValueError("Non-finite native reward")
                rewards[i] += value
                reward_frames[i] += int(value != 0)
            events.extend({"frame": steps, "raw": event} for event in env.frame_events)
            if not env.agents:
                end_reason = "native_termination" if any(term.values()) else "frame_cap"
                complete = True
                break
        if steps == cap:
            complete = True
        metrics = event_metrics(events)
        return {"policy": label, "horizon_mode": mode, "evaluation_seed": seed,
                "frame_cap": cap, "time_feature_horizon": time_horizon,
                "env_steps": steps, "complete": complete, "end_reason": end_reason,
                "focal_return": rewards[0], "response_return": rewards[1],
                "focal_native_return_per_1000_frames": rewards[0] * 1000 / steps if steps else None,
                "nonzero_reward_frames": {f"agent_{i}": reward_frames[i] for i in range(2)},
                "actions": {f"agent_{i}": histogram_stats(hist[i]) for i in range(2)},
                "seconds": round(time.monotonic() - started, 4), **metrics}, events
    finally:
        env.close()


def save_results(output, rows, protocol, status, reason, begin, error=None):
    write_json(output / "episodes.json", rows)
    fields = ["horizon_mode", "policy", "evaluation_seed", "env_steps", "complete", "end_reason",
              "focal_return", "response_return", "focal_native_return_per_1000_frames",
              "interaction_count", "first_interaction_frame", "focal_resources", "response_resources",
              "focal_action_counts", "response_action_counts", "parse_errors"]
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields if k in row}
            flat.update(focal_resources=row["resource_collection_totals"]["agent_0"],
                        response_resources=row["resource_collection_totals"]["agent_1"],
                        focal_action_counts=json.dumps(row["actions"]["agent_0"]["counts"]),
                        response_action_counts=json.dumps(row["actions"]["agent_1"]["counts"]),
                        parse_errors=len(row["event_parse_errors"]))
            writer.writerow(flat)
    stats = []
    for mode in protocol["horizons"]:
        for policy in ("untrained", "trained"):
            group = [r for r in rows if r["horizon_mode"] == mode and r["policy"] == policy and r["complete"]]
            if group:
                first = [r["first_interaction_frame"] for r in group if r["first_interaction_frame"] is not None]
                stats.append({"horizon_mode": mode, "policy": policy, "complete_episodes": len(group),
                              "mean_native_return": statistics.mean(r["focal_return"] for r in group),
                              "mean_interaction_count": statistics.mean(r["interaction_count"] for r in group),
                              "episode_fraction_with_interaction": sum(r["interaction_count"] > 0 for r in group) / len(group),
                              "median_first_interaction_frame_conditional": statistics.median(first) if first else None})
    summary = {"status": status, "stop_reason": reason, "error": error,
               "seconds": round(time.monotonic() - begin, 4), "task": protocol["task"],
               "new_training_steps": 0, "evaluation_steps": sum(r["env_steps"] for r in rows),
               "event_parse_error_count": sum(len(r["event_parse_errors"]) for r in rows),
               "policy_summaries": stats, "quality": paired_quality(rows),
               "checkpoints_sha256": protocol["checkpoints_sha256"], "limits": protocol["limits"],
               "native_duration_note": protocol["native_duration_note"]}
    write_json(output / "summary.json", summary)
    return summary


def run(args):
    begin = time.monotonic()
    deadline = begin + args.max_seconds
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.crossbench_root:
        sys.path.insert(0, str(args.crossbench_root.resolve()))
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from benchmark import seed_for
    from crossbench.envs import action
    from crossbench.melting import Parallel
    torch.set_num_threads(1)
    task = TASK_ALIASES[args.task]
    base = args.run_dir.resolve()
    paths = {"trained": args.trained or base / "initial/focal.zip",
             "untrained": args.untrained or base / "initial/focal0_untrained.zip",
             "opponent": args.opponent or base / "initial/response.zip"}
    paths = {k: Path(p).resolve() for k, p in paths.items()}
    hashes = {k: sha(p) for k, p in paths.items()}
    manifest = json.loads((base / "manifest.json").read_text()) if (base / "manifest.json").exists() else None
    if manifest and manifest.get("task") != task:
        raise ValueError("Requested task differs from checkpoint manifest")
    time_horizon = args.time_feature_horizon or (manifest or {}).get("config", {}).get("horizon", 1000)
    cls = diagnostic_parallel_class(Parallel)
    probe = cls(task, 1000)
    try:
        settings = probe.cfg.lab2d_settings_builder(roles=probe.cfg.default_player_roles, config=probe.cfg)
        native_cap = int(settings["maxEpisodeLengthFrames"])
        action_table = plain(probe.cfg.action_set)
        expected_shape = (probe.observation_space("agent_0").shape[0] + 1,)
    finally:
        probe.close()
    caps = {mode: native_cap if mode == "native" else int(mode) for mode in args.horizons}
    if any(cap > native_cap for cap in caps.values()):
        raise ValueError("Frame cap exceeds the native maximum")
    models = {k: PPO.load(str(p), device="cpu") for k, p in paths.items()}
    for label, model in models.items():
        if model.observation_space.shape != expected_shape:
            raise ValueError(f"{label} observation shape {model.observation_space.shape} != {expected_shape}")
        if model.action_space.n != len(action_table):
            raise ValueError(f"{label} action space differs from substrate")
    forbidden = set()
    old_seed_path = base / "seeds.json"
    if old_seed_path.exists():
        forbidden.update(v for v in json.loads(old_seed_path.read_text()).values() if isinstance(v, int))
    plan = [{"horizon_mode": mode, "episode": i, "evaluation_seed": evaluation_seed(args.seed, mode, i, forbidden)}
            for i in range(args.episodes) for mode in args.horizons]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"purpose": "development_behavior_diagnostic_no_training", "task": task,
                "run_dir": str(base), "horizons": args.horizons, "frame_caps": caps,
                "native_maximum_frames": native_cap, "time_feature_horizon": time_horizon,
                "episodes_per_policy_per_horizon": args.episodes, "seed_plan": plan,
                "checkpoints": {k: str(p) for k, p in paths.items()}, "checkpoints_sha256": hashes,
                "source_sha256": sha(__file__), "action_table": action_table,
                "checkpoint_manifest": manifest, "event_capture": "Subscribe to existing public event observable after reset; do not read event queue twice",
                "policy_observations": "Unchanged local pooled RGB, own inventory, readiness, clamped remaining time; event data never enters policies",
                "native_duration_note": "Native maximum and stochastic termination preserved. Time feature uses checkpoint training horizon and clamps at zero. Beyond training horizon is extrapolation, not comparable to the 1000-frame benchmark or a newly trained native-horizon agent.",
                "limits": {"max_seconds": args.max_seconds, "max_env_steps": args.max_env_steps,
                           "deadline_includes_import_and_loading": True}, "new_training_steps": 0}
    write_json(output / "protocol.json", protocol)
    rows, status, reason, error = [], "complete", "planned_episodes_complete", None
    rng_factory = lambda seed, role: np.random.default_rng(seed_for(seed, role))
    try:
        with (output / "events.jsonl").open("w") as detail:
            for item in plan:
                mode, seed = item["horizon_mode"], item["evaluation_seed"]
                remaining = args.max_env_steps - sum(r["env_steps"] for r in rows)
                if time.monotonic() >= deadline:
                    status, reason = "budget_stopped", "wall_time_budget"; break
                if remaining < 2 * caps[mode]:
                    status, reason = "budget_stopped", "insufficient_steps_for_complete_pair"; break
                for label in ("untrained", "trained"):
                    env = cls(task, caps[mode])
                    row, events = evaluate_episode(env, models[label], models["opponent"], label, mode, seed,
                                                  caps[mode], time_horizon, deadline, remaining,
                                                  rng_factory, action)
                    rows.append(row)
                    remaining -= row["env_steps"]
                    for event in events:
                        detail.write(json.dumps({"horizon_mode": mode, "evaluation_seed": seed,
                                                 "policy": label, **event}, allow_nan=False) + "\n")
                    detail.flush()
                    write_json(output / "episodes.json", rows)
                    print(json.dumps({k: row[k] for k in ("policy", "horizon_mode", "evaluation_seed", "env_steps", "complete", "focal_return", "interaction_count", "first_interaction_frame")}), flush=True)
                    if not row["complete"]:
                        status, reason = "budget_stopped", row["end_reason"]; break
                if status != "complete":
                    break
        if any(r["event_parse_errors"] for r in rows):
            status, reason = "invalid_event_metrics", "raw_events_retained_for_parser_review"
    except Exception as exc:
        status, reason, error = "failed", "evaluation_error", f"{type(exc).__name__}: {exc}"
    finally:
        if hashes != {k: sha(p) for k, p in paths.items()}:
            status, reason = "failed", "checkpoint_hash_changed"
        summary = save_results(output, rows, protocol, status, reason, begin, error)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 1 if status in ("failed", "invalid_event_metrics") else 0


def self_test():
    events = [
        {"frame": 2, "raw": ("collected_resource", {"player_index": [1], "class": [2]})},
        {"frame": 4, "raw": ("collected_resource", ["dict", "player_index", 2, "class", 1])},
        {"frame": 7, "raw": ("interaction", {"row_player_idx": 2, "col_player_idx": 1, "row_reward": 3, "col_reward": 2})},
    ]
    metrics = event_metrics(events)
    assert metrics["interaction_count"] == 1 and metrics["first_interaction_frame"] == 7
    assert metrics["resource_collection_totals"] == {"agent_0": 1, "agent_1": 1}
    assert not metrics["event_parse_errors"]
    bad = event_metrics([{"frame": 1, "raw": ("interaction", {"row_player_idx": 1})}])
    assert bad["interaction_count"] == 0 and len(bad["event_parse_errors"]) == 1
    other = event_metrics([{"frame": 1, "raw": ("AvatarStarted", "success")}])
    assert other["event_counts"] == {"AvatarStarted": 1} and not other["event_parse_errors"]
    assert histogram_stats([0, 3])["empirical_entropy_nats"] == 0
    assert abs(histogram_stats([2, 2])["empirical_entropy_nats"] - math.log(2)) < 1e-12
    forbidden = set(); seeds = [evaluation_seed(42, "native", i, forbidden) for i in range(100)]
    assert len(set(seeds)) == 100
    common = {"horizon_mode": "1000", "evaluation_seed": 3, "complete": True,
              "interaction_count": 1, "resource_collection_totals": {"agent_0": 4}}
    rows = [{**common, "policy": "trained", "focal_return": 4},
            {**common, "policy": "untrained", "focal_return": 2}]
    assert paired_quality(rows)["pairs"][0]["native_return_gain"] == 2
    rows[0]["complete"] = False
    assert not paired_quality(rows)["pairs"]
    print("PASS: event decoding, true event metrics, parse-error detection, action histograms, fresh seed allocation, paired complete-episode quality")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), default="repeated_stag_hunt")
    parser.add_argument("--horizons", default="1000,native", help="Comma-separated frame caps and/or native")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-env-steps", type=int, default=96000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--time-feature-horizon", type=int)
    parser.add_argument("--crossbench-root", type=Path)
    for label in ("trained", "untrained", "opponent"):
        parser.add_argument("--" + label, type=Path, help="Override checkpoint path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.run_dir is None or args.output is None:
        parser.error("--run-dir and --output are required")
    if args.episodes <= 0 or args.max_seconds <= 0 or args.max_env_steps <= 0:
        parser.error("Episode and runtime budgets must be positive")
    if args.max_seconds > 7200:
        parser.error("This diagnostic is limited to at most 7200 seconds")
    if args.time_feature_horizon is not None and args.time_feature_horizon <= 0:
        parser.error("--time-feature-horizon must be positive")
    args.horizons = args.horizons.split(",")
    if len(set(args.horizons)) != len(args.horizons) or any(mode != "native" and (not mode.isdigit() or int(mode) <= 0) for mode in args.horizons):
        parser.error("--horizons must contain unique positive integers and/or native")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
