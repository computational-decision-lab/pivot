#!/usr/bin/env python3
"""V2: real-rollout training plus stratified evaluation of episodic mixtures.

The four conditional return means are measured afresh in the actual native
environment for each proxy/query/audit block. They are NOT a hand-written game
matrix and never enter candidate or responder training. A Bernoulli draw occurs
only at episode start; frozen skill networks and fresh recurrent states make
weighted evaluation exact in policy-space, subject to Monte Carlo return noise.
V1 and its sampled-evaluation artifacts remain separate.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

import melting_hierarchical_e5 as v1
from melting_e5_adapter import METHODS, fit_external_posterior, make_candidates, run_e5_decision, source_provenance

VERSION = "official_specialist_bernoulli_stratified_e5_v2"
COMBINATIONS = ((0, 0), (0, 1), (1, 0), (1, 1))  # 0=hare, 1=stag
read, write, sha, digest = v1.read, v1.write, v1.sha, v1.digest
probability, train_logit, BudgetStop, verify_seal = v1.probability, v1.train_logit, v1.BudgetStop, v1.verify_seal


class SeedBook(v1.SeedBook):
    def get(self, *parts):
        key = json.dumps(parts)
        if key not in self.values:
            attempt = 0
            while True:
                value = int(digest([VERSION, self.seed, self.task, parts, attempt])[:12], 16) % (2**31 - 1)
                if value not in self.used:
                    break
                attempt += 1
            self.values[key] = value
            self.used.add(value)
            write(self.path, self.values)
        return self.values[key]


class Episodes(v1.Episodes):
    def __init__(self, output, backend, config, max_seconds, max_env_steps):
        super().__init__(output, backend, config, max_seconds, max_env_steps)
        self.seeds = SeedBook(self.output / "seeds.json", config["seed"], config["task"])

    def run(self, key, template, focal_logit=None, response_logit=None, *, forced_specialists=None):
        path = self.output / "episodes" / (digest(key)[:24] + ".json")
        if forced_specialists is None:
            focal_p, response_p = probability(focal_logit), probability(response_logit)
            uniform_seeds = [self.seeds.get(*template, "mixture", role) for role in (0, 1)]
            uniforms = [float(np.random.default_rng(s).random()) for s in uniform_seeds]
            chosen = [int(uniforms[0] < focal_p), int(uniforms[1] < response_p)]
            mode = "sampled_training"
        else:
            if len(forced_specialists) != 2 or any(type(z) is not int or z not in (0, 1) for z in forced_specialists):
                raise ValueError("forced_specialists must contain two exact integer 0/1 values")
            chosen = list(forced_specialists)
            focal_p, response_p = map(float, chosen)
            focal_logit = response_logit = None
            uniform_seeds, uniforms, mode = [], [], "forced_evaluation_stratum"
        matchup = tuple("stag" if z else "hare" for z in chosen)
        environment_seed = self.seeds.get(*template, "environment")
        policy_seeds = [self.seeds.get(*template, "policy", role) for role in (0, 1)]
        signature = {
            "version": VERSION, "key": list(key), "template": list(template), "sampling_mode": mode,
            "focal_logit": focal_logit, "response_logit": response_logit,
            "forced_specialists": list(forced_specialists) if forced_specialists is not None else None,
            "mixture_uniform_seeds": uniform_seeds, "mixture_uniforms": uniforms,
            "matchup": list(matchup), "environment_seed": environment_seed,
            "policy_seeds": policy_seeds, "horizon": self.config["horizon"],
        }
        if path.exists():
            row = read(path)
            if row["signature"] != signature or not row["complete"] or row.get("event_parse_errors"):
                raise ValueError("cached episode is invalid or does not match its requested inputs")
            self.cost_record(path, row)
            return row
        remaining = self.max_env_steps - self.frames
        if time.monotonic() >= self.deadline:
            raise BudgetStop("wall_time_budget")
        if remaining < self.config["horizon"]:
            raise BudgetStop("insufficient_frames_for_complete_episode")
        row, events = self.backend.episode(matchup=matchup, env_seed=environment_seed,
                                           policy_seeds=policy_seeds, deadline=self.deadline, allowance=remaining)
        if any(not math.isfinite(float(row[f"{role}_return"])) for role in ("focal", "response")):
            raise ValueError("nonfinite native episode reward")
        row.update(signature=signature, stage=key[0], task=self.config["task"], train_seed=self.config["seed"],
                   eval_mode="official_specialist_hierarchical_stratified_v2", sampled_specialists=chosen,
                   focal_probability=focal_p, response_probability=response_p, episode_id=digest(key)[:24])
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
            raise ValueError("episode event metrics failed; saved row was not used for training")
        print(json.dumps({"stage": key[0], "seed": self.config["seed"], "episode_id": row["episode_id"],
                          "sampling_mode": mode, "env_steps": row["env_steps"], "physical_env_steps": self.frames,
                          "focal_return": row["focal_return"], "response_return": row["response_return"],
                          "interaction_count": row.get("interaction_count")}), flush=True)
        return row


def weighted_estimate(block, focal_probability, response_probability):
    p, q = float(focal_probability), float(response_probability)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, q)):
        raise ValueError("mixture probabilities must lie in [0, 1]")
    weights = {f"{a}{b}": (p if a else 1-p) * (q if b else 1-q) for a, b in COMBINATIONS}
    return {"focal_probability": p, "response_probability": q, "stratum_weights": weights,
            "focal_return": sum(weights[s] * block["conditional_return_means"][s]["focal_return"] for s in weights),
            "response_return": sum(weights[s] * block["conditional_return_means"][s]["response_return"] for s in weights),
            "estimator": "weighted_fresh_native_stratum_means"}


def evaluation_block(episodes, key, template, path, frozen_context):
    path = Path(path)
    replicates = episodes.config["proxy_stratum_replicates"] if key[0] == "proxy" else episodes.config["stratum_replicates"]
    signature = {"version": VERSION, "key": list(key), "template": list(template),
                 "replicates_per_stratum": replicates,
                 "config_sha256": digest(episodes.config), "frozen_context": frozen_context}
    if path.exists():
        block = read(path)
        if block["signature"] != signature:
            raise ValueError("cached stratified block has changed inputs")
        return block
    if key[0] == "audit":
        verify_seal(episodes.output)
    rows, means = [], {}
    for a, b in COMBINATIONS:
        stratum = []
        for r in range(replicates):
            # Within a replicate, share environment and low-level random keys
            # across strata. It is a paired block, not four independent seeds.
            row = episodes.run((*key, a, b, r), (*template, r), forced_specialists=(a, b))
            stratum.append(row)
            rows.append(row)
        means[f"{a}{b}"] = {field: float(np.mean([row[field] for row in stratum]))
                              for field in ("focal_return", "response_return")}
    block = {"version": VERSION, "signature": signature, "rows": rows,
             "conditional_return_means": means, "replicates_per_stratum": replicates,
             "logical_evaluation_episodes": len(rows), "actual_env_frames": sum(r["env_steps"] for r in rows),
             "evaluation_seeds": [episodes.seeds.get(*template, r, "environment") for r in range(replicates)],
             "complete": True, "training_reads_this_block": False,
             "sampling_note": "Four real native skill-combination strata; common random template within each replicate. Return means are noisy, not known payoffs."}
    write(path, block)
    return block


def response_pair(episodes, candidate, candidate_logit, stream):
    c = episodes.config
    path = episodes.output / stream / f"candidate_{candidate}" / "paired.json"
    signature = {"candidate": candidate, "candidate_logit": candidate_logit, "stream": stream,
                 "version": VERSION, "config_sha256": digest(c)}
    if path.exists():
        pair = read(path)
        if pair["signature"] != signature:
            raise ValueError("paired response input changed")
        return pair
    if stream == "audit":
        verify_seal(episodes.output)
    old, initial_op = c["old_logit"], c["opponent_logit"]
    learned = []
    for branch, focal in (("old", old), ("new", candidate_logit)):
        learned.append(train_logit(episodes, role="response", initial_logit=initial_op, other_logit=focal,
                      key=(stream, candidate, "response_training", branch),
                      template=(stream, candidate, "response_training"), count=c["response_train_episodes"],
                      learning_rate=c["response_learning_rate"]))
    p_old, p_new = probability(old), probability(candidate_logit)
    q_old, q_new, q_zero = probability(learned[0]), probability(learned[1]), probability(initial_op)
    context = {"p_old": p_old, "p_new": p_new, "q_old": q_old, "q_new": q_new,
               "q_initial": q_zero, "all_probabilities_frozen_before_block": True}
    block_path = path.with_name("evaluation_block.json")
    block = evaluation_block(episodes, (stream, candidate, "stratified_evaluation"),
                             (stream, candidate, "stratified_evaluation"), block_path, context)
    old_est, new_est = weighted_estimate(block, p_old, q_old), weighted_estimate(block, p_new, q_new)
    training_frames = sum(cost["env_steps"] for name, cost in episodes.costs.items()
                          if cost["complete"] and tuple(cost["key"][:3]) == (stream, candidate, "response_training"))
    pair = {"signature": signature, "candidate": candidate, "stream": stream, "version": VERSION,
            "evaluation_scheme": "four_strata_real_rollout_weighted_expectation",
            "old_estimate": old_est, "new_estimate": new_est,
            "delta": new_est["focal_return"] - old_est["focal_return"],
            "old_response_logit": learned[0], "new_response_logit": learned[1],
            "initial_opponent_logit": initial_op,
            "logical_query_episodes": 2*c["response_train_episodes"] + 4*c["stratum_replicates"],
            "total_env_steps": training_frames + block["actual_env_frames"],
            "actual_env_frames": training_frames + block["actual_env_frames"],
            "training_env_frames": training_frames, "evaluation_env_frames": block["actual_env_frames"],
            "evaluation_block": {"path": str(block_path), "sha256": sha(block_path)},
            "conditional_return_means": block["conditional_return_means"],
            "response_training_histories": {branch: {"path": str(episodes.output / "training" / (digest((stream, candidate, "response_training", branch))[:24]+".json")),
                 "sha256": sha(episodes.output / "training" / (digest((stream, candidate, "response_training", branch))[:24]+".json"))} for branch in ("old", "new")},
            "evaluation_seeds": block["evaluation_seeds"],
            "training_seeds": [episodes.seeds.get(stream, candidate, "response_training", i, "environment") for i in range(c["response_train_episodes"])],
            "independent_response_training_replicates_per_branch": 1,
            "evaluation_episodes_per_stratum": c["stratum_replicates"],
            "evaluation_episodes_total": 4*c["stratum_replicates"]}
    if stream == "audit" and c["audit_mechanism"]:
        frozen_old, frozen_new = weighted_estimate(block, p_old, q_zero), weighted_estimate(block, p_new, q_zero)
        crossed = weighted_estimate(block, p_new, q_old)
        frozen_delta = frozen_new["focal_return"] - frozen_old["focal_return"]
        pair.update(frozen_old_estimate=frozen_old, frozen_new_estimate=frozen_new,
                    frozen_delta=frozen_delta, response_effect=pair["delta"] - frozen_delta,
                    old_response_own_gain=old_est["response_return"] - frozen_old["response_return"],
                    new_response_own_gain=new_est["response_return"] - frozen_new["response_return"],
                    candidate_specific_effect=new_est["focal_return"] - crossed["focal_return"],
                    mechanism_note="All contrasts reweight the same independent audit block; these correlated estimates cost no additional episodes and are not independent experiments.")
    write(path, pair)
    return pair


def calibration_inputs(directory, task, contract):
    rows, hashes, seeds = [], {}, []
    for path in sorted(Path(directory).rglob("calibration_summary.json")):
        payload = read(path)
        if payload["task"] != task:
            continue
        if payload.get("status") != "complete" or payload.get("version") != VERSION:
            raise ValueError("calibration must contain only complete stratified-v2 panels for this task")
        if payload["calibration_contract"] != contract:
            raise ValueError("calibration/test source, model, or learning/evaluation contract differs")
        if payload["seed"] in seeds:
            raise ValueError("duplicate calibration seed")
        seeds.append(payload["seed"])
        hashes[str(path.resolve())] = sha(path)
        rows.extend(payload["rows"])
    if not rows:
        raise ValueError("no matching completed hierarchical-v2 calibration panels")
    return rows, hashes, seeds


def run_panel(args, backend):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("candidate_learning_rate", "response_learning_rate"):
        if getattr(args, name) is None:
            setattr(args, name, args.learning_rate)
    if args.proxy_stratum_replicates is None:
        args.proxy_stratum_replicates = args.stratum_replicates
    c = {key: getattr(args, key) for key in (
        "task", "seed", "phase", "k", "horizon", "candidate_train_episodes", "response_train_episodes",
        "stratum_replicates", "proxy_stratum_replicates", "candidate_learning_rate", "response_learning_rate", "baseline", "reward_scale",
        "reward_clip", "logit_clip", "old_logit", "opponent_logit", "audit_mechanism")}
    c.update(version=VERSION, budgets=list(args.budgets), methods=list(METHODS), diagnostic_only=args.k == 2,
             candidate_lr_multipliers=[0.5, 1.0, 2.0, 1.0], feature_rule="[proxy_delta, p_new-p_old, abs(p_new-p_old)]",
             evaluation_scheme="four_strata_real_rollout_weighted_expectation")
    source_files = (Path(__file__), Path(v1.__file__), Path(__file__).with_name("melting_reference_diagnostic.py"),
                    Path(__file__).with_name("melting_behavior_diagnostic.py"), Path(__file__).with_name("melting_e5_adapter.py"))
    sources = {path.name: sha(path) for path in source_files}
    contract = {key: value for key, value in c.items() if key not in ("phase", "seed", "budgets", "methods", "audit_mechanism")}
    contract.update(official_model=backend.metadata, sources_sha256=sources)
    cal_rows, cal_hashes, cal_seeds = [], {}, []
    if args.phase == "test":
        if args.calibration_dir is None:
            raise ValueError("test requires --calibration-dir")
        cal_rows, cal_hashes, cal_seeds = calibration_inputs(args.calibration_dir, args.task, contract)
        if args.seed in cal_seeds:
            raise ValueError("test seed overlaps calibration seeds")
    protocol = {"version": VERSION, "config": c, "calibration_contract": contract,
                "official_reference": backend.metadata, "sources_sha256": sources, "e5_source": source_provenance(),
                "calibration_input_hashes": cal_hashes, "calibration_seeds": cal_seeds,
                "training_rule": "real sampled episodes; REINFORCE with fixed scaled/clipped own native reward and past-only baseline",
                "evaluation_rule": "four exact Specialist combinations, fresh native rollouts per block, known product-mixture weights; never used for training",
                "query_unit": "two independently branched response trainings plus one stratified evaluation block",
                "proxy_scope": "one independent block shared across all already-frozen candidate probabilities",
                "cost_unit": "real complete episodes; native frames tracked separately",
                "freeze_scope": "all methods/budgets within each test panel before that panel's audit",
                "mechanism_scope": "correlated reweightings of the same audit block, no additional episodes",
                "evidence_scope": "one-dimensional episodic mixture over one frozen official skill network; one task, one update round"}
    if (output / "protocol.json").exists() and read(output / "protocol.json") != protocol:
        raise ValueError("immutable v2 panel protocol changed on resume")
    write(output / "protocol.json", protocol)
    episodes = Episodes(output, backend, c, args.max_seconds, args.max_env_steps)
    begin = time.monotonic()
    try:
        candidates = [train_logit(episodes, role="focal", initial_logit=args.old_logit, other_logit=args.opponent_logit,
                     key=("candidate_training", j), template=("candidate_training", j), count=args.candidate_train_episodes,
                     learning_rate=args.candidate_learning_rate*c["candidate_lr_multipliers"][j % 4]) for j in range(args.k)]
        candidate_payload = {"version": VERSION, "old_logit": args.old_logit, "candidate_logits": candidates,
                             "candidate_probabilities": [probability(v) for v in candidates],
                             "independent_candidate_seed_templates": True}
        write(output / "candidates.json", candidate_payload)
        p_old, q_initial = probability(args.old_logit), probability(args.opponent_logit)
        proxy_path = output / "proxy" / "evaluation_block.json"
        proxy_block = evaluation_block(episodes, ("proxy", "stratified_evaluation"), ("proxy", "stratified_evaluation"),
                                       proxy_path, {"candidate_sha256": sha(output / "candidates.json"), "q_initial": q_initial})
        old_proxy = weighted_estimate(proxy_block, p_old, q_initial)
        features, proxies = [], []
        for j, theta in enumerate(candidates):
            p_new = probability(theta)
            new_proxy = weighted_estimate(proxy_block, p_new, q_initial)
            delta = new_proxy["focal_return"] - old_proxy["focal_return"]
            features.append([delta, p_new-p_old, abs(p_new-p_old)])
            proxies.append({"candidate": j, "delta": delta, "old_estimate": old_proxy, "new_estimate": new_proxy,
                            "evaluation_block": {"path": str(proxy_path), "sha256": sha(proxy_path)}})
        write(output / "features.json", features)
        write(output / "proxy.json", proxies)
        if args.phase == "calibration":
            rows = []
            for j, theta in enumerate(candidates):
                pair = response_pair(episodes, j, theta, "calibration")
                rows.append({"seed": args.seed, "candidate": j, "features": features[j],
                             "target_correction": pair["delta"]-features[j][0], "response_delta": pair["delta"]})
            summary = {"status": "complete", "version": VERSION, "phase": "calibration", "task": args.task,
                       "seed": args.seed, "rows": rows, "calibration_contract": contract, "physical_env_steps": episodes.frames}
            write(output / "calibration_summary.json", summary)
        else:
            posterior, transform = fit_external_posterior(cal_rows)
            write(output / "posterior.json", transform)
            logical_cost = 2*args.response_train_episodes + 4*args.stratum_replicates
            panel = make_candidates(features, transform, logical_cost)
            if (output / "selection_seal.json").exists():
                verify_seal(output)
                decisions = read(output / "decisions_frozen.json")
            else:
                if (output / "audit").exists():
                    raise ValueError("audit exists before any selection seal")
                def query(candidate):
                    j = int(candidate["transition_id"])
                    pair = response_pair(episodes, j, candidates[j], "selection")
                    path = output / "selection" / f"candidate_{j}" / "paired.json"
                    return {"delta": pair["delta"], "hf_query_cost": logical_cost, "stream": "selection",
                            "training_seeds": pair["training_seeds"], "evaluation_seeds": pair["evaluation_seeds"],
                            "input_files": [{"path": str(path), "sha256": sha(path)}]}
                decisions = []
                for method in METHODS:
                    for budget in ((0,) if method == "proxy_only" else args.budgets):
                        decision = run_e5_decision(panel, posterior, query, method=method, budget=budget, seed=args.seed)
                        decision.update(task=args.task, seed=args.seed, train_seed=args.seed, version=VERSION,
                                        eval_mode="official_specialist_hierarchical_stratified_v2", cost_unit="episodes",
                                        selected_probability=probability(candidates[int(decision["selected_candidate_id"])]),
                                        logical_query_episodes=decision["query_cost"],
                                        logical_query_env_frames=sum(read(output / "selection" / f"candidate_{j}" / "paired.json")["total_env_steps"] for j in decision["queried_ids"]))
                        decisions.append(decision)
                write(output / "decisions_frozen.json", decisions)
                write(output / "selection_seal.json", {"version": VERSION, "decisions_sha256": sha(output / "decisions_frozen.json"),
                      "calibration_input_hashes": cal_hashes, "audit_opened": False, "audit_generated": False,
                      "n_decisions": len(decisions), "candidate_sha256": sha(output / "candidates.json"),
                      "features_sha256": sha(output / "features.json"), "posterior_sha256": sha(output / "posterior.json"),
                      "physical_env_steps_at_freeze": episodes.frames})
            audits = [response_pair(episodes, j, theta, "audit") for j, theta in enumerate(candidates)]
            verify_seal(output)
            selection_seeds = {v for key, v in episodes.seeds.values.items() if json.loads(key)[0] == "selection"}
            audit_seeds = {v for key, v in episodes.seeds.values.items() if json.loads(key)[0] == "audit"}
            if selection_seeds & audit_seeds:
                raise ValueError("selection/audit random streams overlap")
            write(output / "isolation.json", {"selection_seed_count": len(selection_seeds), "audit_seed_count": len(audit_seeds),
                  "overlap_count": 0, "covers": ["environment", "training_mixture_uniforms", "low_level_initialization"],
                  "seed_registry_sha256": sha(output / "seeds.json"),
                  "evaluation_note": "Within-block strata share replicate random templates intentionally; query/audit blocks are separate."})
            scored = []
            for decision in decisions:
                pair = audits[int(decision["selected_candidate_id"])]
                path = output / "audit" / f"candidate_{pair['candidate']}" / "paired.json"
                scored.append({**decision, "audit_gain": pair["delta"],
                               "audit_provenance": {"path": str(path), "sha256": sha(path),
                                  "response_training_independent": True, "evaluation_independent_of_selection": True},
                               **{key: pair[key] for key in ("frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect") if key in pair}})
            write(output / "scored_decisions.json", scored)
            truth = [pair["delta"] for pair in audits]
            predicted = [row[0] for row in features]
            positive = [j for j, value in enumerate(predicted) if value > 0]
            non_tied = [j for j, value in enumerate(predicted) if value != 0 and truth[j] != 0]
            summary = {"status": "complete", "version": VERSION, "phase": "test", "task": args.task,
                       "seed": args.seed, "candidate_count": args.k, "n_decisions": len(decisions),
                       "physical_env_steps": episodes.frames, "audit_isolated": True, "selection_seal": verify_seal(output),
                       "proxy_ide": float(np.mean(np.abs(np.array(predicted)-truth))),
                       "proxy_isc": float(np.mean([np.sign(predicted[j]) == np.sign(truth[j]) for j in non_tied])) if non_tied else None,
                       "proxy_positive_count": len(positive), "noisy_negative_audit_count": sum(truth[j] < 0 for j in positive),
                       "metrics_note": "Stratified weighted audit estimates are noisy. One independent responder-training branch per candidate/old-new target; stratum episodes are not additional responder training replicates.",
                       "decision_records": scored}
        if hasattr(backend, "checkpoint_hashes") and backend.checkpoint_hashes() != backend.metadata["model_file_sha256"]:
            raise ValueError("official checkpoint files changed")
        summary.update(seconds_this_invocation=time.monotonic()-begin,
                       completed_physical_episodes=sum(row["complete"] for row in episodes.costs.values()),
                       incomplete_physical_attempts=sum(not row["complete"] for row in episodes.costs.values()))
        write(output / "summary.json", summary)
        write(output / "status.json", {"status": "complete", "physical_env_steps": episodes.frames})
        return summary
    except BudgetStop as exc:
        status = {"status": "budget_stopped", "reason": str(exc), "physical_env_steps": episodes.frames,
                  "selection_frozen": (output / "selection_seal.json").exists()}
        write(output / "status.json", status)
        return status
    except Exception as exc:
        write(output / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "physical_env_steps": episodes.frames})
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
    p.add_argument("--k", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--budgets", type=lambda v: tuple(int(x) for x in v.split(",")), default=(0, 1, 2, 4))
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--candidate-train-episodes", type=int, default=4)
    p.add_argument("--response-train-episodes", type=int, default=4)
    p.add_argument("--stratum-replicates", "--stratified-replicates", dest="stratum_replicates", type=int, default=2)
    p.add_argument("--proxy-stratified-replicates", dest="proxy_stratum_replicates", type=int)
    p.add_argument("--learning-rate", type=float, default=0.5)
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
    # These deterministic fixtures are confined to local temporary directories
    # and are never exposed by the actual CLI's native rollout path.
    block = {"conditional_return_means": {f"{a}{b}": {"focal_return": 3+5*a+7*b+11*a*b,
             "response_return": 2+13*a-3*b+2*a*b} for a, b in COMBINATIONS}}
    for p, q in ((0, 0), (0, 1), (1, 0), (1, 1), (.23, .67), (.5, .5)):
        estimate = weighted_estimate(block, p, q)
        assert math.isclose(estimate["focal_return"], 3+5*p+7*q+11*p*q)
        assert math.isclose(estimate["response_return"], 2+13*p-3*q+2*p*q)
        assert math.isclose(sum(estimate["stratum_weights"].values()), 1)
    class CheckRollout:
        metadata = {"implementation": "internal_control_flow_fixture_only"}
        def __init__(self):
            self.calls = []
        def episode(self, *, matchup, env_seed, policy_seeds, deadline, allowance):
            self.calls.append((matchup, env_seed, tuple(policy_seeds)))
            a, b = [int(name == "stag") for name in matchup]
            return {"complete": True, "env_steps": 3, "focal_return": float(3+5*a+7*b+11*a*b),
                    "response_return": float(2+13*a-3*b+2*a*b), "seconds": .001,
                    "interaction_count": 1, "event_parse_errors": []}, []
    with tempfile.TemporaryDirectory(prefix="stratified-e5-local-check-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        root = Path(tmp)
        backend = CheckRollout()
        def arguments(phase, seed, output):
            return parser().parse_args(["--phase", phase, "--seed", str(seed), "--output", str(output),
               "--horizon", "3", "--k", "2", "--budgets", "0,1,2", "--candidate-train-episodes", "1",
               "--response-train-episodes", "1", "--stratum-replicates", "1", "--calibration-dir", str(root/"calibration")])
        cal = arguments("calibration", 11, root/"calibration"/"seed_11")
        a = run_panel(cal, backend)
        assert a["completed_physical_episodes"] == 18
        test = arguments("test", 22, root/"test"/"seed_22")
        test.max_env_steps = 3
        assert run_panel(test, backend)["status"] == "budget_stopped"
        test.max_env_steps = 100000
        b = run_panel(test, backend)
        assert b["completed_physical_episodes"] == 30 and b["physical_env_steps"] == 90
        calls = len(backend.calls)
        run_panel(test, backend)
        assert len(backend.calls) == calls
        pair = read(test.output/"audit"/"candidate_0"/"paired.json")
        assert pair["logical_query_episodes"] == 6
        assert pair["independent_response_training_replicates_per_branch"] == 1
        assert "old" not in pair and "new" not in pair
        evaluated = read(test.output/"audit"/"candidate_0"/"evaluation_block.json")
        assert len(evaluated["rows"]) == 4
        for row in evaluated["rows"]:
            assert row["signature"]["mixture_uniform_seeds"] == []
            assert row["signature"]["focal_logit"] is None
            assert row["sampled_specialists"] == row["signature"]["forced_specialists"]
        for path in (test.output/"training").glob("*.json"):
            for row in read(path)["history"]:
                native = read(test.output/"episodes"/(row["episode_id"]+".json"))
                assert native["signature"]["sampling_mode"] == "sampled_training"
        selection = read(test.output/"selection"/"candidate_0"/"paired.json")
        assert set(selection["training_seeds"]).isdisjoint(pair["training_seeds"])
        assert set(selection["evaluation_seeds"]).isdisjoint(pair["evaluation_seeds"])
        decisions = read(test.output/"decisions_frozen.json")
        assert len(decisions) == 13 and all("audit_gain" not in row for row in decisions)
        assert all(row["query_cost"] == 6*row["query_packages_used"] for row in decisions)
        verify_seal(test.output)
    print("PASS: exact p=0/1 strata and arbitrary-mixture expectation; real-training/forced-evaluation separation; v2 streams; resume/freeze/isolation; 18/30 tiny-panel episode accounting")


def main():
    p = parser()
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.phase is None or args.output is None or args.model_path is None:
        p.error("--phase, --output and --model-path are required")
    if min(args.candidate_train_episodes, args.response_train_episodes, args.stratum_replicates,
           args.horizon, args.max_env_steps) <= 0:
        p.error("episode, stratum and frame counts must be positive")
    if args.proxy_stratum_replicates is not None and args.proxy_stratum_replicates <= 0:
        p.error("proxy stratum replicates must be positive")
    values = (args.learning_rate, args.reward_scale, args.reward_clip, args.logit_clip, args.max_seconds)
    values += tuple(v for v in (args.candidate_learning_rate, args.response_learning_rate) if v is not None)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        p.error("learning and time limits must be finite and positive")
    if not math.isfinite(args.old_logit) or not math.isfinite(args.opponent_logit):
        p.error("initial logits must be finite")
    if not args.budgets or len(set(args.budgets)) != len(args.budgets) or any(b < 0 or b > args.k for b in args.budgets):
        p.error("budgets must be unique integers between 0 and K")
    from melting_reference_diagnostic import OfficialSpecialistRollout
    backend = OfficialSpecialistRollout(model_path=args.model_path, task=args.task, horizon=args.horizon,
                                       crossbench_root=args.crossbench_root)
    try:
        result = run_panel(args, backend)
    finally:
        backend.close()
    print(json.dumps({"status": result["status"], "output": str(args.output), "physical_env_steps": result["physical_env_steps"]}), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
