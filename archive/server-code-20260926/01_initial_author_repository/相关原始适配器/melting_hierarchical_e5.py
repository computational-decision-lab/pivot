#!/usr/bin/env python3
"""Real-rollout E5C pilot over fixed official Specialist policies.

Only the episode-level Bernoulli logit learns. Native rewards enter a fixed
scaled/clipped REINFORCE update; no payoff matrix or PPO checkpoint is used.
Each candidate and response branch starts from its declared common snapshot.
Every episode is persisted, and all method choices are sealed before audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from melting_e5_adapter import (
    METHODS, fit_external_posterior, make_candidates, run_e5_decision, source_provenance,
)

VERSION = "official_specialist_bernoulli_e5_v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def probability(logit):
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


class BudgetStop(RuntimeError):
    pass


class SeedBook:
    """Persist collision-checked random templates; paired branches reuse keys."""
    def __init__(self, path, seed, task):
        self.path, self.seed, self.task = Path(path), int(seed), task
        self.values = read(path) if self.path.exists() else {}
        self.used = set(self.values.values())
        if len(self.used) != len(self.values):
            raise ValueError("seed registry contains collisions")

    def get(self, *parts):
        key = json.dumps(parts)
        if key not in self.values:
            attempt = 0
            while True:
                raw = digest([VERSION, self.seed, self.task, parts, attempt])
                value = int(raw[:12], 16) % (2**31 - 1)
                if value not in self.used:
                    break
                attempt += 1
            self.values[key] = value
            self.used.add(value)
            write(self.path, self.values)
        return self.values[key]


class Episodes:
    def __init__(self, output, backend, config, max_seconds, max_env_steps):
        self.output, self.backend, self.config = Path(output), backend, config
        self.deadline = time.monotonic() + max_seconds
        self.max_env_steps = int(max_env_steps)
        self.seeds = SeedBook(self.output / "seeds.json", config["seed"], config["task"])
        self.cost_path = self.output / "episode_costs.json"
        self.costs = read(self.cost_path) if self.cost_path.exists() else {}

    @property
    def frames(self):
        return sum(row["env_steps"] for row in self.costs.values())

    def cost_record(self, path, row):
        self.costs[str(path.relative_to(self.output))] = {
            "env_steps": int(row["env_steps"]), "complete": bool(row["complete"]),
            "stage": row["stage"], "key": row["signature"]["key"],
            "seconds": float(row.get("seconds", 0.0)),
        }
        write(self.cost_path, self.costs)

    def run(self, key, template, focal_logit, response_logit):
        # File identity includes branch; random-template identity intentionally
        # excludes old/new within a pair. Audit templates use a separate stream.
        path = self.output / "episodes" / (digest(key)[:24] + ".json")
        focal_p, response_p = probability(focal_logit), probability(response_logit)
        uniform_seeds = [self.seeds.get(*template, "mixture", role) for role in (0, 1)]
        uniforms = [float(np.random.default_rng(s).random()) for s in uniform_seeds]
        chosen = [int(uniforms[0] < focal_p), int(uniforms[1] < response_p)]
        matchup = tuple("stag" if z else "hare" for z in chosen)
        environment_seed = self.seeds.get(*template, "environment")
        policy_seeds = [self.seeds.get(*template, "policy", role) for role in (0, 1)]
        signature = {
            "key": list(key), "template": list(template), "focal_logit": float(focal_logit),
            "response_logit": float(response_logit), "mixture_uniform_seeds": uniform_seeds,
            "mixture_uniforms": uniforms, "matchup": list(matchup),
            "environment_seed": environment_seed, "policy_seeds": policy_seeds,
            "horizon": self.config["horizon"],
        }
        if path.exists():
            row = read(path)
            if row["signature"] != signature or not row["complete"]:
                raise ValueError("cached episode does not match its requested inputs")
            self.cost_record(path, row)
            if row.get("event_parse_errors"):
                raise ValueError("cached episode has invalid event metrics")
            return row
        remaining = self.max_env_steps - self.frames
        if time.monotonic() >= self.deadline:
            raise BudgetStop("wall_time_budget")
        if remaining < self.config["horizon"]:
            raise BudgetStop("insufficient_frames_for_complete_episode")
        row, events = self.backend.episode(
            matchup=matchup, env_seed=environment_seed, policy_seeds=policy_seeds,
            deadline=self.deadline, allowance=remaining,
        )
        for role in ("focal", "response"):
            if not math.isfinite(float(row[f"{role}_return"])):
                raise ValueError("nonfinite native episode reward")
        row.update(signature=signature, stage=key[0], task=self.config["task"],
                   train_seed=self.config["seed"], sampled_specialists=chosen,
                   focal_probability=focal_p, response_probability=response_p,
                   episode_id=digest(key)[:24])
        if not row["complete"]:
            index = len(list(path.parent.glob(path.stem + ".partial_*.json")))
            partial = path.with_name(path.stem + f".partial_{index}.json")
            write(partial, row)
            self.cost_record(partial, row)
            write(partial.with_suffix(".events.json"), events)
            raise BudgetStop(row.get("end_reason", "incomplete_episode"))
        write(path, row)
        self.cost_record(path, row)
        write(path.with_suffix(".events.json"), events)
        if row.get("event_parse_errors"):
            raise ValueError("episode event metrics failed validation; row retained but not used for training")
        print(json.dumps({"stage": key[0], "seed": self.config["seed"],
                          "episode_id": row["episode_id"], "env_steps": row["env_steps"],
                          "physical_env_steps": self.frames,
                          "focal_return": row["focal_return"], "response_return": row["response_return"],
                          "interaction_count": row.get("interaction_count")}), flush=True)
        return row


def train_logit(episodes, *, role, initial_logit, other_logit, key, template, count, learning_rate):
    config = episodes.config
    theta = float(initial_logit)
    past_reward_sum = 0.0
    history = []
    for index in range(count):
        focal, response = (theta, other_logit) if role == "focal" else (other_logit, theta)
        row = episodes.run((*key, index), (*template, index), focal, response)
        p = probability(theta)
        z = row["sampled_specialists"][0 if role == "focal" else 1]
        reward = float(row[f"{role}_return"])
        scaled = float(np.clip(reward / config["reward_scale"], -config["reward_clip"], config["reward_clip"]))
        baseline = past_reward_sum / index if index and config["baseline"] == "running_mean" else 0.0
        advantage = scaled - baseline
        gradient = advantage * (z - p)
        updated = float(np.clip(theta + learning_rate * gradient, -config["logit_clip"], config["logit_clip"]))
        history.append({"episode_id": row["episode_id"], "logit_before": theta,
                        "probability": p, "sampled_stag": z, "native_return": reward,
                        "scaled_clipped_return": scaled, "baseline_before": baseline,
                        "advantage": advantage, "gradient": gradient,
                        "learning_rate": learning_rate, "logit_after": updated})
        theta = updated
        past_reward_sum += scaled
        write(episodes.output / "training" / (digest(key)[:24] + ".json"),
              {"role": role, "initial_logit": initial_logit, "other_logit": other_logit,
               "optimizer": "plain_SGD_no_momentum", "baseline": config["baseline"],
               "baseline_uses_only_previous_training_episodes": True,
               "past_scaled_reward_sum": past_reward_sum, "training_episode_count": index + 1,
               "history": history,
               "final_logit": theta, "complete": index + 1 == count})
    return theta


def average(rows, field):
    return float(np.mean([row[field] for row in rows]))


def difference(old, new, field="focal_return"):
    return float(np.mean([n[field] - o[field] for o, n in zip(old, new)]))


def response_pair(episodes, candidate, candidate_logit, stream):
    config = episodes.config
    path = episodes.output / stream / f"candidate_{candidate}" / "paired.json"
    signature = {"candidate": candidate, "candidate_logit": candidate_logit,
                 "stream": stream, "config_sha256": digest(config)}
    if path.exists():
        cached = read(path)
        if cached["signature"] != signature:
            raise ValueError("cached paired response has changed inputs")
        return cached
    if stream == "audit":
        verify_seal(episodes.output)
    old_logit, opponent = config["old_logit"], config["opponent_logit"]
    starting_frames = episodes.frames
    learned = []
    for branch, focal in (("old", old_logit), ("new", candidate_logit)):
        learned.append(train_logit(
            episodes, role="response", initial_logit=opponent, other_logit=focal,
            key=(stream, candidate, "response_training", branch),
            template=(stream, candidate, "response_training"),
            count=config["response_train_episodes"], learning_rate=config["response_learning_rate"],
        ))
    groups = []
    for branch, focal, op in (("old", old_logit, learned[0]), ("new", candidate_logit, learned[1])):
        groups.append([episodes.run((stream, candidate, "evaluation", branch, index),
                                   (stream, candidate, "evaluation", index), focal, op)
                       for index in range(config["eval_episodes"])])
    old, new = groups
    # Count complete episode costs from their keys, including cached episodes.
    prefix = (stream, candidate)
    involved = []
    for ledger_path, cost in episodes.costs.items():
        if ".partial_" in ledger_path:
            continue
        episode_key = cost["key"]
        if tuple(episode_key[:2]) != prefix or episode_key[2] not in ("response_training", "evaluation"):
            continue
        episode_path = episodes.output / ledger_path
        # These are this stream's response records only, and audit is sealed.
        row = read(episode_path)
        involved.append(row)
    pair = {
        "signature": signature, "candidate": candidate, "stream": stream,
        "delta": difference(old, new), "old": old, "new": new,
        "old_response_logit": learned[0], "new_response_logit": learned[1],
        "initial_opponent_logit": opponent,
        "logical_query_episodes": 2 * (config["response_train_episodes"] + config["eval_episodes"]),
        "total_env_steps": sum(row["env_steps"] for row in involved),
        "new_physical_frames_in_this_call": episodes.frames - starting_frames,
        "evaluation_seeds": [row["signature"]["environment_seed"] for row in old],
        "training_seeds": [episodes.seeds.get(stream, candidate, "response_training", i, "environment")
                           for i in range(config["response_train_episodes"])],
    }
    if stream == "audit" and config["audit_mechanism"]:
        frozen = []
        for branch, focal in (("old", old_logit), ("new", candidate_logit)):
            frozen.append([episodes.run((stream, candidate, "frozen_evaluation", branch, i),
                                       (stream, candidate, "evaluation", i), focal, opponent)
                           for i in range(config["eval_episodes"])])
        pair.update(frozen_delta=difference(*frozen),
                    response_effect=pair["delta"] - difference(*frozen),
                    old_response_own_gain=difference(frozen[0], old, "response_return"),
                    new_response_own_gain=difference(frozen[1], new, "response_return"),
                    frozen_old=frozen[0], frozen_new=frozen[1])
    write(path, pair)
    return pair


def verify_seal(output):
    output = Path(output)
    seal = read(output / "selection_seal.json")
    if sha(output / "decisions_frozen.json") != seal["decisions_sha256"]:
        raise ValueError("frozen decisions no longer match their selection seal")
    for field, name in (("candidate_sha256", "candidates.json"), ("features_sha256", "features.json"),
                        ("posterior_sha256", "posterior.json")):
        if field in seal and sha(output / name) != seal[field]:
            raise ValueError(f"frozen selection input changed: {name}")
    return seal


def calibration_inputs(directory, task, expected_contract):
    paths = sorted(Path(directory).rglob("calibration_summary.json"))
    if not paths:
        raise ValueError("no completed hierarchical calibration panels found")
    rows, hashes, seeds = [], {}, []
    for path in paths:
        payload = read(path)
        if payload["task"] != task:
            continue
        if payload.get("status") != "complete" or payload.get("version") != VERSION:
            raise ValueError("calibration panel is incomplete or from another implementation")
        if payload["calibration_contract"] != expected_contract:
            raise ValueError("calibration and test use different task/model/learning contracts")
        if payload["seed"] in seeds:
            raise ValueError("duplicate calibration seed")
        seeds.append(payload["seed"])
        hashes[str(path.resolve())] = sha(path)
        rows.extend(payload["rows"])
    if not rows:
        raise ValueError("no calibration rows for this task")
    return rows, hashes, seeds


def run_panel(args, backend):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.candidate_learning_rate is None:
        args.candidate_learning_rate = args.learning_rate
    if args.response_learning_rate is None:
        args.response_learning_rate = args.learning_rate
    config = {key: getattr(args, key) for key in (
        "task", "seed", "phase", "k", "horizon", "candidate_train_episodes",
        "response_train_episodes", "eval_episodes", "proxy_episodes", "candidate_learning_rate", "response_learning_rate", "baseline",
        "reward_scale", "reward_clip", "logit_clip", "old_logit", "opponent_logit", "audit_mechanism",
    )}
    config.update(version=VERSION, budgets=list(args.budgets), methods=list(METHODS),
                  candidate_lr_multipliers=[0.5, 1.0, 2.0, 1.0],
                  diagnostic_only=args.k == 2,
                  feature_rule="[proxy_delta, p_new-p_old, abs(p_new-p_old)]")
    identity = backend.metadata
    contract = {key: value for key, value in config.items() if key not in ("phase", "seed", "budgets", "methods", "audit_mechanism")}
    contract["official_model"] = identity
    contract["runner_source_sha256"] = sha(__file__)
    contract["reference_source_sha256"] = sha(Path(__file__).with_name("melting_reference_diagnostic.py"))
    contract["behavior_utility_sha256"] = sha(Path(__file__).with_name("melting_behavior_diagnostic.py"))
    cal_rows, cal_hashes, cal_seeds = [], {}, []
    if args.phase == "test":
        if args.calibration_dir is None:
            raise ValueError("test requires --calibration-dir containing hierarchical calibration summaries")
        cal_rows, cal_hashes, cal_seeds = calibration_inputs(args.calibration_dir, args.task, contract)
        if args.seed in cal_seeds:
            raise ValueError("test seed overlaps hierarchical calibration seeds")
    protocol = {
        "config": config, "calibration_contract": contract, "official_reference": identity,
        "source_sha256": sha(__file__), "e5_source": source_provenance(),
        "adapter_sha256": sha(Path(__file__).with_name("melting_e5_adapter.py")),
        "reference_source_sha256": sha(Path(__file__).with_name("melting_reference_diagnostic.py")),
        "behavior_utility_sha256": sha(Path(__file__).with_name("melting_behavior_diagnostic.py")),
        "calibration_input_hashes": cal_hashes, "calibration_seeds": cal_seeds,
        "cost_unit": "complete real environment episodes; actual frames recorded separately",
        "training_rule": "plain SGD REINFORCE on fixed scaled/clipped native reward minus past-only running mean (or zero); fixed logit clipping",
        "mixture_rule": "draw Specialist once per episode; frozen official network, fresh recurrent state",
        "evidence_scope": "restricted hierarchical-policy single-round E5C pilot, not PPO recovery or full official suite",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("immutable panel protocol changed on resume")
    write(protocol_path, protocol)
    episodes = Episodes(output, backend, config, args.max_seconds, args.max_env_steps)
    begin = time.monotonic()
    try:
        candidates = []
        for j in range(args.k):
            rate = args.candidate_learning_rate * config["candidate_lr_multipliers"][j % 4]
            candidates.append(train_logit(
                episodes, role="focal", initial_logit=args.old_logit, other_logit=args.opponent_logit,
                key=("candidate_training", j), template=("candidate_training", j),
                count=args.candidate_train_episodes, learning_rate=rate,
            ))
        write(output / "candidates.json", {"old_logit": args.old_logit,
              "candidate_logits": candidates, "candidate_probabilities": [probability(v) for v in candidates],
              "independent_candidate_seed_templates": True})
        old_proxy = [episodes.run(("proxy", "old", i), ("proxy", i), args.old_logit, args.opponent_logit)
                     for i in range(args.proxy_episodes)]
        features, proxies = [], []
        for j, theta in enumerate(candidates):
            new = [episodes.run(("proxy", j, i), ("proxy", i), theta, args.opponent_logit)
                   for i in range(args.proxy_episodes)]
            proxy = difference(old_proxy, new)
            change = probability(theta) - probability(args.old_logit)
            features.append([proxy, change, abs(change)])
            proxies.append({"candidate": j, "delta": proxy, "old": old_proxy, "new": new})
        write(output / "features.json", features)
        write(output / "proxy.json", proxies)
        if args.phase == "calibration":
            rows = []
            for j, theta in enumerate(candidates):
                pair = response_pair(episodes, j, theta, "calibration")
                rows.append({"seed": args.seed, "features": features[j],
                             "target_correction": pair["delta"] - features[j][0],
                             "candidate": j, "response_delta": pair["delta"]})
            summary = {"status": "complete", "version": VERSION, "phase": "calibration",
                       "task": args.task, "seed": args.seed, "rows": rows,
                       "calibration_contract": contract, "physical_env_steps": episodes.frames}
            write(output / "calibration_summary.json", summary)
        else:
            posterior, transform = fit_external_posterior(cal_rows)
            write(output / "posterior.json", transform)
            logical_cost = 2 * (args.response_train_episodes + args.eval_episodes)
            panel = make_candidates(features, transform, logical_cost)
            if (output / "selection_seal.json").exists():
                verify_seal(output)
                decisions = read(output / "decisions_frozen.json")
            else:
                if (output / "audit").exists():
                    raise ValueError("audit exists before selection was sealed")
                def query(candidate):
                    j = int(candidate["transition_id"])
                    paired = response_pair(episodes, j, candidates[j], "selection")
                    path = output / "selection" / f"candidate_{j}" / "paired.json"
                    # E5 cost is episodes, NOT the variable actual frame count.
                    return {"delta": paired["delta"], "hf_query_cost": logical_cost,
                            "stream": "selection", "training_seeds": paired["training_seeds"],
                            "evaluation_seeds": paired["evaluation_seeds"],
                            "input_files": [{"path": str(path), "sha256": sha(path)}]}
                decisions = []
                for method in METHODS:
                    for budget in ((0,) if method == "proxy_only" else args.budgets):
                        decision = run_e5_decision(panel, posterior, query, method=method, budget=budget, seed=args.seed)
                        decision.update(task=args.task, seed=args.seed, train_seed=args.seed,
                                        eval_mode="official_specialist_hierarchical", cost_unit="episodes",
                                        logical_query_episodes=decision["query_cost"],
                                        logical_query_env_frames=sum(read(output / "selection" / f"candidate_{j}" / "paired.json")["total_env_steps"] for j in decision["queried_ids"]))
                        decisions.append(decision)
                write(output / "decisions_frozen.json", decisions)
                write(output / "selection_seal.json", {
                    "decisions_sha256": sha(output / "decisions_frozen.json"),
                    "calibration_input_hashes": cal_hashes, "audit_opened": False,
                    "audit_generated": False, "n_decisions": len(decisions),
                    "candidate_sha256": sha(output / "candidates.json"),
                    "features_sha256": sha(output / "features.json"),
                    "posterior_sha256": sha(output / "posterior.json"),
                    "physical_env_steps_at_freeze": episodes.frames,
                })
            audits = [response_pair(episodes, j, theta, "audit") for j, theta in enumerate(candidates)]
            verify_seal(output)
            selection_seeds = {value for key, value in episodes.seeds.values.items() if json.loads(key)[0] == "selection"}
            audit_seeds = {value for key, value in episodes.seeds.values.items() if json.loads(key)[0] == "audit"}
            if selection_seeds.intersection(audit_seeds):
                raise ValueError("selection and audit random streams overlap")
            write(output / "isolation.json", {"selection_seed_count": len(selection_seeds),
                  "audit_seed_count": len(audit_seeds), "overlap_count": 0,
                  "covers": ["environment", "mixture_uniforms", "low_level_random_initialization"],
                  "seed_registry_sha256": sha(output / "seeds.json")})
            scored = []
            for decision in decisions:
                pair = audits[int(decision["selected_candidate_id"])]
                path = output / "audit" / f"candidate_{pair['candidate']}" / "paired.json"
                scored.append({**decision, "audit_gain": pair["delta"],
                               "audit_provenance": {"path": str(path), "sha256": sha(path),
                                                    "response_training_independent": True,
                                                    "evaluation_independent_of_selection": True},
                               **{key: pair[key] for key in ("frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain") if key in pair}})
            write(output / "scored_decisions.json", scored)
            audit_values = [pair["delta"] for pair in audits]
            predicted = [row[0] for row in features]
            positive = [j for j, value in enumerate(predicted) if value > 0]
            non_tied = [j for j, value in enumerate(predicted) if value != 0 and audit_values[j] != 0]
            summary = {"status": "complete", "version": VERSION, "phase": "test", "task": args.task,
                       "seed": args.seed, "candidate_count": args.k, "n_decisions": len(decisions),
                       "physical_env_steps": episodes.frames, "audit_isolated": True,
                       "selection_seal": verify_seal(output),
                       "proxy_ide": float(np.mean(np.abs(np.array(predicted) - audit_values))),
                       "proxy_isc": float(np.mean([np.sign(predicted[j]) == np.sign(audit_values[j]) for j in non_tied])) if non_tied else None,
                       "proxy_positive_count": len(positive),
                       "noisy_negative_audit_count": sum(audit_values[j] < 0 for j in positive),
                       "metrics_note": "Audit means are noisy; no ground-truth regret or confirmed reversal claim.",
                       "decision_records": scored}
        summary["seconds_this_invocation"] = time.monotonic() - begin
        if hasattr(backend, "checkpoint_hashes") and backend.checkpoint_hashes() != identity["model_file_sha256"]:
            raise ValueError("frozen official checkpoint files changed during the panel")
        summary["completed_physical_episodes"] = sum(row["complete"] for row in episodes.costs.values())
        summary["incomplete_physical_attempts"] = sum(not row["complete"] for row in episodes.costs.values())
        write(output / "summary.json", summary)
        write(output / "status.json", {"status": "complete", "physical_env_steps": episodes.frames})
        return summary
    except BudgetStop as exc:
        status = {"status": "budget_stopped", "reason": str(exc), "physical_env_steps": episodes.frames,
                  "selection_frozen": (output / "selection_seal.json").exists()}
        write(output / "status.json", status)
        return status
    except Exception as exc:
        write(output / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                       "physical_env_steps": episodes.frames})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("calibration", "test"))
    p.add_argument("--task", default="melting_stag")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--output", type=Path)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--crossbench-root", type=Path)
    p.add_argument("--calibration-dir", type=Path)
    p.add_argument("--k", type=int, choices=(2, 4, 8), default=4,
                   help="K=2 is marked diagnostic_only; K=4/8 are benchmark slices")
    p.add_argument("--budgets", type=lambda v: tuple(int(x) for x in v.split(",")), default=(0, 1, 2, 4))
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--candidate-train-episodes", type=int, default=2)
    p.add_argument("--response-train-episodes", type=int, default=2)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--proxy-episodes", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--candidate-learning-rate", type=float)
    p.add_argument("--response-learning-rate", type=float)
    p.add_argument("--baseline", choices=("running_mean", "zero"), default="running_mean")
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--reward-clip", type=float, default=2.0)
    p.add_argument("--logit-clip", type=float, default=8.0)
    p.add_argument("--old-logit", type=float, default=0.0)
    p.add_argument("--opponent-logit", type=float, default=0.0)
    p.add_argument("--audit-mechanism", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--max-env-steps", type=int, default=2000000)
    p.add_argument("--self-test", action="store_true")
    return p


def self_test():
    """Exercise scheduling/isolation/resume in temporary files, never user data."""
    class LocalCheckRollout:
        metadata = {"implementation": "internal_unit_test_only"}
        def __init__(self):
            self.calls = []
        def episode(self, *, matchup, env_seed, policy_seeds, deadline, allowance):
            self.calls.append((matchup, env_seed, policy_seeds))
            # This fixture checks control flow only. It never enters real runs.
            value = float(2 + (env_seed % 7))
            return {"complete": True, "env_steps": 3, "focal_return": value,
                    "response_return": value + 1.0, "seconds": 0.001,
                    "interaction_count": 1, "event_parse_errors": []}, []
    import contextlib
    import io
    with tempfile.TemporaryDirectory(prefix="hierarchical-e5-unit-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        root = Path(tmp)
        def arguments(phase, seed, target):
            return parser().parse_args(["--phase", phase, "--seed", str(seed), "--output", str(target),
                                       "--horizon", "3", "--candidate-train-episodes", "1",
                                       "--response-train-episodes", "1", "--eval-episodes", "1",
                                       "--proxy-episodes", "1", "--calibration-dir", str(root / "calibration")])
        backend = LocalCheckRollout()
        cal = arguments("calibration", 1, root / "calibration" / "seed_1")
        run_panel(cal, backend)
        count = len(backend.calls)
        run_panel(cal, backend)
        assert len(backend.calls) == count, "resume re-executed completed episodes"
        test = arguments("test", 2, root / "test" / "seed_2")
        test.max_env_steps = 3  # Force an interruption before the next full episode.
        stopped = run_panel(test, backend)
        assert stopped["status"] == "budget_stopped"
        test.max_env_steps = 100000
        result = run_panel(test, backend)
        assert result["status"] == "complete" and result["audit_isolated"]
        count = len(backend.calls)
        run_panel(test, backend)
        assert len(backend.calls) == count
        output = test.output
        selection, audit = read(output / "selection" / "candidate_0" / "paired.json"), read(output / "audit" / "candidate_0" / "paired.json")
        assert set(selection["training_seeds"]).isdisjoint(audit["training_seeds"])
        assert set(selection["evaluation_seeds"]).isdisjoint(audit["evaluation_seeds"])
        assert selection["old"][0]["signature"]["template"] == selection["new"][0]["signature"]["template"]
        assert selection["logical_query_episodes"] == 4
        assert result["completed_physical_episodes"] == 49
        assert result["physical_env_steps"] == 49 * 3
        decisions = read(output / "decisions_frozen.json")
        assert all("audit_gain" not in row for row in decisions)
        assert all(row["query_cost"] == 4 * row["query_packages_used"] for row in decisions)
        assert len(decisions) == 17
        verify_seal(output)
        # Separately verify the past-only baseline and update equation for
        # multiple training episodes (the scheduling fixture above uses T=1).
        grad_config = dict(read(output / "protocol.json")["config"])
        trainer = Episodes(root / "gradient_check", backend, grad_config, 100, 100000)
        final = train_logit(trainer, role="response", initial_logit=0.0, other_logit=0.0,
                            key=("gradient_check",), template=("gradient_check",),
                            count=3, learning_rate=0.5)
        history = read(next((root / "gradient_check" / "training").glob("*.json")))["history"]
        for index, step in enumerate(history):
            expected_baseline = sum(s["scaled_clipped_return"] for s in history[:index]) / index if index else 0.0
            assert math.isclose(step["baseline_before"], expected_baseline)
            expected = step["logit_before"] + 0.5 * (step["scaled_clipped_return"] - expected_baseline) * (step["sampled_stag"] - step["probability"])
            assert math.isclose(step["logit_after"], expected)
        assert final == history[-1]["logit_after"]
    print("PASS: temporary control-flow, REINFORCE history, independent seed streams, fixed logical costs, freeze-before-audit and interrupted/completed resume checks")


def main():
    p = parser()
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.phase is None or args.output is None or args.model_path is None:
        p.error("--phase, --output and --model-path are required")
    counts = (args.candidate_train_episodes, args.response_train_episodes, args.eval_episodes,
              args.proxy_episodes, args.horizon, args.max_env_steps)
    numbers = (args.learning_rate, args.reward_scale, args.reward_clip, args.logit_clip, args.max_seconds)
    if min(counts) <= 0 or any(not math.isfinite(v) or v <= 0 for v in numbers):
        p.error("episode counts, learning settings and resource limits must be finite and positive")
    if any(value is not None and (not math.isfinite(value) or value <= 0)
           for value in (args.candidate_learning_rate, args.response_learning_rate)):
        p.error("candidate and response learning rates must be finite and positive")
    if not args.budgets or len(set(args.budgets)) != len(args.budgets) or any(b < 0 or b > args.k for b in args.budgets):
        p.error("budgets must be unique integers between 0 and K")
    if not math.isfinite(args.old_logit) or not math.isfinite(args.opponent_logit):
        p.error("initial logits must be finite")
    from melting_reference_diagnostic import OfficialSpecialistRollout
    backend = OfficialSpecialistRollout(model_path=args.model_path, task=args.task,
                                       horizon=args.horizon, crossbench_root=args.crossbench_root)
    try:
        result = run_panel(args, backend)
    finally:
        backend.close()
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "physical_env_steps": result["physical_env_steps"]}), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
