#!/usr/bin/env python3
"""Bounded real-native response-learning diagnostic, not an E5 selection run.

Three fixed focal Bernoulli policies each face an independently initialized
learned responder. Real episode rewards drive unchanged REINFORCE updates.
Two fresh four-stratum evaluation blocks are collected only after every
training trajectory is sealed. Exact mixture reweighting shares those blocks
across frozen checkpoints, without treating derived estimates as new episodes.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time

# Permit the standard sibling checkout layout; explicit PYTHONPATH also works.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "colin_pivot", _HERE.parent / "colin_pivot" / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

import melting_hierarchical_e5_stratified as v2

VERSION = "melting_response_sprint_v1"
STREAM = "response_sprint_v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def logit(p):
    return math.log(p / (1.0 - p))


def source_hashes():
    names = ("melting_response_sprint.py", "melting_hierarchical_e5_stratified.py",
             "melting_hierarchical_e5.py", "melting_reference_diagnostic.py",
             "melting_behavior_diagnostic.py", "melting_e5_adapter.py")
    return {name: v2.sha(_HERE / name) for name in names}


def counts(episodes):
    completed = [c for c in episodes.costs.values() if c["complete"]]
    return {"actual_complete_episodes": len(completed),
            "actual_partial_episodes": sum(not c["complete"] for c in episodes.costs.values()),
            "actual_training_episodes": sum(c["stage"] == "training" for c in completed),
            "actual_evaluation_episodes": sum(c["stage"] == "evaluation" for c in completed),
            "actual_env_frames": episodes.frames}


def expected_counts(config):
    training = len(config["target_probabilities"]) * config["train_episodes"]
    evaluation = 2 * 4 * config["stratum_replicates"]
    return {"expected_training_episodes": training,
            "expected_evaluation_episodes": evaluation,
            "expected_total_episodes": training + evaluation}


def verify_frozen(output, seal):
    if v2.sha(output / "protocol.json") != seal["protocol_sha256"]:
        raise ValueError("protocol changed after training freeze")
    for target in seal["targets"]:
        history = target["training_history"]
        if v2.sha(output / history["path"]) != history["sha256"]:
            raise ValueError("training history changed after freeze")


def execute(output, backend, config, *, max_seconds, max_env_steps, deadline_utc=None):
    """Run one root seed; backend must execute actual native games in production."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = v2.Episodes(output, backend, config, max_seconds, max_env_steps)
    start = time.monotonic()
    expected = expected_counts(config)
    protocol = {"version": VERSION, "config": config, "sources_sha256": source_hashes(),
                "backend": backend.metadata,
                "interpretation": "response-learning mechanism diagnostic; no candidate selection or PIVOT decision",
                "training": {
                    "algorithm": "unchanged v1.train_logit native episodic REINFORCE",
                    "learned_parameters": "one responder Bernoulli logit per fixed focal target",
                    "reward": "responder native return scaled then clipped; previous-episode running baseline",
                    "across_target_training_crn": True,
                    "shared_template": [STREAM, "training", "episode_index"],
                    "target_trajectories_independently_initialized": True,
                    "checkpoint_training_prefixes": True,
                    "evaluation_is_never_read_by_training": True},
                "evaluation": {
                    "blocks": ["A", "B"], "independent_blocks": True,
                    "independent_of_training": True,
                    "forced_skill_combinations": [[0, 0], [0, 1], [1, 0], [1, 1]],
                    "skill_encoding": {"0": "hare", "1": "stag"},
                    "replicates_per_stratum": config["stratum_replicates"],
                    "shared_across_targets_and_checkpoints": True,
                    "within_replicate_strata_crn": True,
                    "estimator": "exact Bernoulli product weights times fresh native conditional means",
                    "statistical_unit": "independent root training seed; blocks/targets/checkpoints are paired"},
                "expected_counts": expected}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if v2.read(protocol_path) != protocol:
            raise ValueError("existing run protocol does not match requested inputs")
    else:
        v2.write(protocol_path, protocol)
    v2.write(output / "attempt_budget.json", {
        "started_utc": utc_now(), "max_seconds_including_remaining_run": max_seconds,
        "max_env_steps": max_env_steps, "deadline_utc": deadline_utc,
        "deadline_enforced_in_each_native_episode": True})
    summary = {"version": VERSION, "seed": config["seed"], "task": config["task"],
               "config": config, **expected, "complete": False, "status": "running",
               "protocol_sha256": v2.sha(protocol_path)}
    v2.write(output / "summary.json", {**summary, **counts(episodes)})
    try:
        targets = []
        for index, p in enumerate(config["target_probabilities"]):
            v2.write(output / "status.json", {"status": "training", "target_id": index,
                                             "target_probability": p, "updated_utc": utc_now()})
            key = ("training", index)
            v2.train_logit(episodes, role="response", initial_logit=config["initial_response_logit"],
                           other_logit=logit(p), key=key,
                           template=(STREAM, "training"), count=config["train_episodes"],
                           learning_rate=config["response_learning_rate"])
            history_path = output / "training" / (v2.digest(key)[:24] + ".json")
            history = v2.read(history_path)
            if not history["complete"] or len(history["history"]) != config["train_episodes"]:
                raise ValueError("training did not reach every declared checkpoint")
            checkpoint_logits = {
                str(n): (config["initial_response_logit"] if n == 0 else history["history"][n-1]["logit_after"])
                for n in config["checkpoints"]}
            targets.append({"target_id": index, "target_probability": p,
                            "target_logit": logit(p), "training_episode_count": config["train_episodes"],
                            "training_key": list(key), "training_template": [STREAM, "training"],
                            "training_history": {"path": str(history_path.relative_to(output)),
                                                 "sha256": v2.sha(history_path)},
                            "checkpoint_logits": checkpoint_logits,
                            "checkpoint_probabilities": {n: v2.probability(t) for n, t in checkpoint_logits.items()}})
        seal = {"version": VERSION, "seed": config["seed"],
                "protocol_sha256": v2.sha(protocol_path), "config_sha256": v2.digest(config),
                "targets": targets, "all_training_frozen_before_evaluation": True,
                "training_reads_evaluation": False}
        seal_path = output / "freeze_training.json"
        if seal_path.exists() and v2.read(seal_path) != seal:
            raise ValueError("training freeze changed on resume")
        if not seal_path.exists():
            v2.write(seal_path, seal)
        verify_frozen(output, seal)
        seal_hash = v2.sha(seal_path)
        blocks = {}
        for name in ("A", "B"):
            verify_frozen(output, seal)
            v2.write(output / "status.json", {"status": "evaluation", "block": name,
                                             "training_seal_sha256": seal_hash, "updated_utc": utc_now()})
            path = output / "evaluation" / f"block_{name}.json"
            block = v2.evaluation_block(
                episodes, ("evaluation", name), (STREAM, "evaluation", name), path,
                {"training_seal_sha256": seal_hash, "all_probabilities_frozen_before_block": True})
            if not block.get("complete") or len(block["rows"]) != 4 * config["stratum_replicates"]:
                raise ValueError("incomplete evaluation block")
            blocks[name] = (block, v2.sha(path))
        verify_frozen(output, seal)
        if hasattr(backend, "checkpoint_hashes"):
            before = backend.metadata.get("model_file_sha256")
            if before is not None and backend.checkpoint_hashes() != before:
                raise ValueError("frozen official model files changed during run")
        results = []
        for target in targets:
            p = target["target_probability"]
            for n in config["checkpoints"]:
                q = target["checkpoint_probabilities"][str(n)]
                q0 = v2.probability(config["initial_response_logit"])
                for name, (block, block_hash) in blocks.items():
                    learned = v2.weighted_estimate(block, p, q)
                    initial = v2.weighted_estimate(block, p, q0)
                    stag = v2.weighted_estimate(block, p, 1.0)
                    hare = v2.weighted_estimate(block, p, 0.0)
                    results.append({"version": VERSION, "seed": config["seed"],
                        "target_id": target["target_id"], "target_probability": p,
                        "training_episodes": n, "response_logit": target["checkpoint_logits"][str(n)],
                        "response_probability": q, "q_initial": q0, "eval_block": name,
                        "focal_return": learned["focal_return"], "response_return": learned["response_return"],
                        "frozen_focal_return": initial["focal_return"],
                        "frozen_response_return": initial["response_return"],
                        "response_own_gain": learned["response_return"] - initial["response_return"],
                        "focal_response_effect": learned["focal_return"] - initial["focal_return"],
                        "stag_minus_hare_response_advantage": stag["response_return"] - hare["response_return"],
                        "stratum_weights": learned["stratum_weights"],
                        "evaluation_block_sha256": block_hash, "training_seal_sha256": seal_hash,
                        "derived_estimate_not_additional_episode": True})
        actual = counts(episodes)
        if (actual["actual_complete_episodes"] != expected["expected_total_episodes"] or
                actual["actual_training_episodes"] != expected["expected_training_episodes"] or
                actual["actual_evaluation_episodes"] != expected["expected_evaluation_episodes"]):
            raise ValueError("actual complete episode counts disagree with frozen protocol")
        v2.write(output / "results.json", results)
        summary.update(complete=True, status="complete", training_seal_sha256=seal_hash,
                       results_sha256=v2.sha(output / "results.json"), result_rows=len(results),
                       evaluation_blocks={n: {"path": f"evaluation/block_{n}.json", "sha256": b[1]}
                                          for n, b in blocks.items()})
    except v2.BudgetStop as exc:
        summary.update(status="budget_stopped", stop_reason=str(exc))
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(**counts(episodes), elapsed_seconds=time.monotonic() - start,
                       finished_utc=utc_now())
        v2.write(output / "summary.json", summary)
        v2.write(output / "status.json", {"status": summary["status"], "complete": summary["complete"],
                                         "updated_utc": utc_now()})
    return summary


def self_test():
    """Synthetic backend only validates orchestration; never creates evidence."""
    class FakeBackend:
        metadata = {"backend": "self_test_only_not_scientific_evidence"}
        def __init__(self):
            self.calls = 0
        def episode(self, matchup, env_seed, policy_seeds, deadline, allowance):
            self.calls += 1
            a, b = [int(x == "stag") for x in matchup]
            return {"complete": True, "env_steps": 1, "focal_return": 3.0 + a + 2*b,
                    "response_return": 10.0 + 4*a + 3*b, "event_parse_errors": 0}, []
    config = {"task": "melting_stag", "seed": 987654,
              "target_probabilities": [.25, .5, .75], "train_episodes": 2,
              "checkpoints": [0, 1, 2], "stratum_replicates": 1,
              "proxy_stratum_replicates": 1, "horizon": 1,
              "response_learning_rate": .5, "reward_scale": 100., "reward_clip": 2.,
              "baseline": "running_mean", "logit_clip": 8., "initial_response_logit": 0.}
    with tempfile.TemporaryDirectory(prefix="response_sprint_selftest_") as tmp:
        out = Path(tmp) / "complete"
        backend = FakeBackend()
        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute(out, backend, config, max_seconds=30, max_env_steps=14)
        assert summary["complete"] and backend.calls == 14
        rows = v2.read(out / "results.json")
        assert len(rows) == 18
        for row in rows:
            assert math.isclose(row["response_own_gain"], 3*(row["response_probability"]-.5), abs_tol=1e-12)
            assert math.isclose(row["focal_response_effect"], 2*(row["response_probability"]-.5), abs_tol=1e-12)
        all_rows = [v2.read(p) for p in (out / "episodes").glob("*.json") if not p.name.endswith(".events.json")]
        train = [r for r in all_rows if r["stage"] == "training"]
        evaluate = [r for r in all_rows if r["stage"] == "evaluation"]
        assert all(r["signature"]["sampling_mode"] == "sampled_training" for r in train)
        assert all(r["signature"]["sampling_mode"] == "forced_evaluation_stratum" for r in evaluate)
        train_seeds = {r["signature"]["environment_seed"] for r in train}
        eval_seeds = {r["signature"]["environment_seed"] for r in evaluate}
        assert len(train_seeds) == 2 and len(eval_seeds) == 2 and train_seeds.isdisjoint(eval_seeds)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = execute(out, backend, config, max_seconds=30, max_env_steps=14)
        assert resumed["complete"] and backend.calls == 14
        stopped = execute(Path(tmp)/"stopped", FakeBackend(), config, max_seconds=-1, max_env_steps=14)
        assert stopped["status"] == "budget_stopped" and not stopped["complete"]
        assert not (Path(tmp)/"stopped"/"results.json").exists()
    print("response_sprint self-test passed: 14 mock episodes, checkpoints, exact weights, paired streams, independent evaluation, resume, budget stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--crossbench-root", type=Path)
    parser.add_argument("--task", default="melting_stag")
    parser.add_argument("--target-probabilities", nargs="+", type=float, default=[.25, .5, .75])
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 4, 16, 32])
    parser.add_argument("--eval-episodes-per-stratum", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2500)
    parser.add_argument("--response-learning-rate", type=float, default=.5)
    parser.add_argument("--reward-scale", type=float, default=100.)
    parser.add_argument("--reward-clip", type=float, default=2.)
    parser.add_argument("--logit-clip", type=float, default=8.)
    parser.add_argument("--max-seconds", type=float, default=2400.)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--deadline-utc", help="optional absolute ISO UTC deadline in addition to --max-seconds")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output is None or args.seed is None or args.model_path is None:
        parser.error("--output, --seed, and --model-path are required")
    if (args.train_episodes < 1 or args.eval_episodes_per_stratum < 1 or args.horizon < 1 or
            args.max_seconds <= 0 or not math.isfinite(args.max_seconds) or
            any(not 0 < p < 1 for p in args.target_probabilities) or
            len(set(args.target_probabilities)) != len(args.target_probabilities) or
            args.checkpoints != sorted(set(args.checkpoints)) or 0 not in args.checkpoints or
            any(n < 0 or n > args.train_episodes for n in args.checkpoints) or
            any(not math.isfinite(x) or x <= 0 for x in
                (args.response_learning_rate, args.reward_scale, args.reward_clip, args.logit_clip))):
        parser.error("invalid probabilities/checkpoints or nonpositive budget/training parameter")
    config = {"task": args.task, "seed": args.seed,
              "target_probabilities": args.target_probabilities,
              "train_episodes": args.train_episodes, "checkpoints": args.checkpoints,
              "stratum_replicates": args.eval_episodes_per_stratum,
              "proxy_stratum_replicates": args.eval_episodes_per_stratum,
              "horizon": args.horizon, "response_learning_rate": args.response_learning_rate,
              "reward_scale": args.reward_scale, "reward_clip": args.reward_clip,
              "baseline": "running_mean", "logit_clip": args.logit_clip,
              "initial_response_logit": 0.0}
    deadline = time.monotonic() + args.max_seconds
    if args.deadline_utc:
        absolute = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if absolute.tzinfo is None:
            parser.error("--deadline-utc must include UTC timezone")
        deadline = min(deadline, time.monotonic() + absolute.timestamp() - time.time())
    maximum_frames = args.max_env_steps or expected_counts(config)["expected_total_episodes"] * args.horizon
    from melting_reference_diagnostic import OfficialSpecialistRollout
    backend = None
    try:
        backend = OfficialSpecialistRollout(args.model_path, task=args.task, horizon=args.horizon,
                                           crossbench_root=args.crossbench_root)
        summary = execute(args.output, backend, config, max_seconds=deadline-time.monotonic(),
                          max_env_steps=maximum_frames, deadline_utc=args.deadline_utc)
        print(json.dumps(summary), flush=True)
        return 0 if summary["complete"] else 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
