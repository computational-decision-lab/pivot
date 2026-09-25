#!/usr/bin/env python3
"""Read-only integrity verification for a predeclared short/long cohort.

Fresh response training and an optional reanalysis of earlier evidence have
separate provenance, CSVs, plots and conclusions. This script never runs an
environment, trains a model, or selects a PIVOT winner.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile

import numpy as np

VERSION = "melting_response_sprint_v1"
STATISTICS_SEED = 20260916
BOOTSTRAP_DRAWS = 10000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-8), message)


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def weighted(means, p, q, role):
    require(0 <= p <= 1 and 0 <= q <= 1, "invalid mixture probability")
    return float(sum((p if a else 1-p)*(q if b else 1-q)*means[f"{a}{b}"][role]
                     for a in (0, 1) for b in (0, 1)))


def mean_ci(values, name):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), "invalid bootstrap input")
    seed = int(hashlib.sha256(f"{STATISTICS_SEED}:{name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(len(x), size=(BOOTSTRAP_DRAWS, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(), "n_seeds": len(x)}


def csv_rows(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def load(self, path, expected_hash=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "evidence path escaped batch root")
        h = sha(path)
        self.hashes[str(path.relative_to(self.root))] = h
        if expected_hash is not None:
            require(h == expected_hash, f"changed evidence: {path.name}")
        return read(path)


def verify_block(evidence, path, native, replicas, config):
    block = evidence.load(path)
    require(block["complete"] and not block["training_reads_this_block"], "invalid evaluation block")
    require(block["replicates_per_stratum"] == replicas and len(block["rows"]) == 4*replicas,
            "wrong conditional evaluation count")
    groups = {}
    for row in block["rows"]:
        require(row == native[row["episode_id"]], "evaluation block differs from persisted native episode")
        sig = row["signature"]
        require(sig["sampling_mode"] == "forced_evaluation_stratum" and sig["focal_logit"] is None and sig["response_logit"] is None,
                "evaluation stratum was not exactly forced")
        a, b = sig["forced_specialists"]
        r = sig["key"][-1]
        require(a in (0, 1) and b in (0, 1) and 0 <= r < replicas and (a,b,r) not in groups,
                "duplicate/invalid evaluation stratum")
        require(row["sampled_specialists"] == [a, b], "forced Specialist identity mismatch")
        groups[a,b,r] = row
    for a in (0, 1):
        for b in (0, 1):
            for role in ("focal_return", "response_return"):
                close(block["conditional_return_means"][f"{a}{b}"][role],
                      np.mean([groups[a,b,r][role] for r in range(replicas)]), "conditional mean is not the raw-return mean")
    close(block["actual_env_frames"], sum(row["env_steps"] for row in block["rows"]), "evaluation frame count mismatch")
    return block


def inspect_panel(evidence, folder, seed, batch_protocol, source_dir):
    mechanism = batch_protocol["mechanism"]
    replicas = mechanism["eval_replicates_per_skill_stratum_per_block"]
    evaluation_count = 8 * replicas
    total_count = 96 + evaluation_count
    info = evidence.load(folder/"protocol.json")
    config = info["config"]
    require(info["version"] == VERSION and config["seed"] == seed, "wrong sprint version/seed")
    require(config["target_probabilities"] == [.25, .5, .75] and config["checkpoints"] == mechanism["checkpoints"], "wrong target/checkpoint grid")
    require(config["train_episodes"] == 32 and config["horizon"] == 2500 and config["stratum_replicates"] == replicas,
            "engineering smoke or changed response budget")
    require(config["baseline"] == "running_mean" and config["initial_response_logit"] == 0, "changed learner initialization")
    for name, expected in info["sources_sha256"].items():
        path = Path(source_dir)/Path(name).name
        require(path.is_file() and sha(path) == expected, f"source mismatch: {name}")
    state, summary = evidence.load(folder/"status.json"), evidence.load(folder/"summary.json")
    require(state["status"] == summary["status"] == "complete", "unfinished panel")
    require(summary["complete"] and summary["actual_complete_episodes"] == total_count
            and summary["actual_training_episodes"] == 96
            and summary["actual_evaluation_episodes"] == evaluation_count
            and summary["actual_partial_episodes"] == 0, "summary episode accounting mismatch")
    require(summary["protocol_sha256"] == sha(folder/"protocol.json"), "protocol seal mismatch")
    costs, registry = evidence.load(folder/"episode_costs.json"), evidence.load(folder/"seeds.json")
    require(len(costs) == total_count and all(c["complete"] for c in costs.values()), "expected configured training and evaluation episodes")
    require(len(set(registry.values())) == len(registry), "seed registry collision")
    native, contexts, training_contexts, evaluation_contexts = {}, set(), set(), set()
    counts, frames = Counter(), Counter()
    for rel, cost in costs.items():
        row = evidence.load(folder/rel)
        sig = row["signature"]
        require(row["complete"] and not row.get("event_parse_errors"), "incomplete/invalid native episode")
        require(row["episode_id"] == digest(sig["key"])[:24] and row["train_seed"] == seed,
                "native episode identity mismatch")
        close(row["env_steps"], cost["env_steps"], "native frame ledger mismatch")
        require(0 < row["env_steps"] <= 2500 and cost["key"] == sig["key"], "invalid episode length/key")
        require(all(math.isfinite(row[k]) for k in ("focal_return", "response_return")), "nonfinite native return")
        template = sig["template"]
        require(registry[json.dumps([*template,"environment"])] == sig["environment_seed"], "unregistered environment seed")
        for role in (0,1):
            require(registry[json.dumps([*template,"policy",role])] == sig["policy_seeds"][role], "unregistered low-level seed")
        context = (sig["environment_seed"], *sig["policy_seeds"])
        contexts.add(context)
        if sig["sampling_mode"] == "sampled_training":
            category = "training"
            training_contexts.add(context)
            for role, name in enumerate(("focal", "response")):
                uniform_seed = registry[json.dumps([*template,"mixture",role])]
                require(uniform_seed == sig["mixture_uniform_seeds"][role], "training mixture seed mismatch")
                u = float(np.random.default_rng(uniform_seed).random())
                close(sig["mixture_uniforms"][role], u, "training mixture draw mismatch")
                require(row["sampled_specialists"][role] == int(u < sigmoid(sig[f"{name}_logit"])), "incorrect sampled skill")
        else:
            require(sig["sampling_mode"] == "forced_evaluation_stratum", "unknown rollout role")
            category = "evaluation"
            evaluation_contexts.add(context)
        native[row["episode_id"]] = row
        counts[category] += 1
        frames[category] += row["env_steps"]
    require(counts == {"training":96,"evaluation":evaluation_count} and not training_contexts & evaluation_contexts,
            "training/evaluation overlap or wrong count")
    require(len(training_contexts) == 32 and len(evaluation_contexts) == 2*replicas,
            "paired target training or evaluation random templates changed")
    close(summary["actual_env_frames"], sum(frames.values()), "summary frame ledger mismatch")
    frozen = evidence.load(folder/"freeze_training.json")
    require(frozen["protocol_sha256"] == sha(folder/"protocol.json")
            and frozen["config_sha256"] == digest(config)
            and summary["training_seal_sha256"] == sha(folder/"freeze_training.json")
            and frozen["all_training_frozen_before_evaluation"]
            and not frozen["training_reads_evaluation"], "training freeze contract mismatch")
    # The exact serialization is checked below when a checkpoint is used; all
    # training histories are independently reconstructed from saved native rows.
    histories, checkpoints = {}, {}
    for target, p in enumerate(config["target_probabilities"]):
        path = folder/"training"/(digest(("training",target))[:24]+".json")
        history = evidence.load(path)
        require(history["complete"] and len(history["history"]) == 32 and history["role"] == "response", "incomplete response learning")
        close(history["initial_logit"], 0, "wrong initial response policy")
        close(sigmoid(history["other_logit"]), p, "wrong frozen focal target")
        theta, previous_sum = 0.0, 0.0
        checkpoints[target,0] = theta
        for i, step in enumerate(history["history"]):
            row = native[step["episode_id"]]
            require(row["signature"]["sampling_mode"] == "sampled_training" and row["signature"]["key"] == ["training",target,i],
                    "learning history references wrong trajectory")
            close(row["signature"]["response_logit"], theta, "response update not inherited")
            close(sigmoid(row["signature"]["focal_logit"]), p, "focal changed during response learning")
            scaled = float(np.clip(row["response_return"]/config["reward_scale"],-config["reward_clip"],config["reward_clip"]))
            baseline = previous_sum/i if i else 0.0
            gradient = (scaled-baseline)*(row["sampled_specialists"][1]-sigmoid(theta))
            after = float(np.clip(theta+config["response_learning_rate"]*gradient,-config["logit_clip"],config["logit_clip"]))
            for key, value in (("native_return",row["response_return"]),("baseline_before",baseline),("gradient",gradient),("logit_after",after)):
                close(step[key], value, "response REINFORCE update mismatch")
            theta, previous_sum = after, previous_sum+scaled
            if i+1 in config["checkpoints"]:
                checkpoints[target,i+1] = theta
        close(history["final_logit"],theta,"wrong final learned state")
        histories[target] = {"path":str(path.relative_to(folder)),"sha256":sha(path)}
    freeze_text = json.dumps(frozen, sort_keys=True)
    require(all(value["sha256"] in freeze_text for value in histories.values()), "training freeze omitted/changed a history")
    blocks = {name:verify_block(evidence,folder/"evaluation"/f"block_{name}.json",native,replicas,config) for name in ("A","B")}
    require(all(block["signature"]["frozen_context"]["training_seal_sha256"] == sha(folder/"freeze_training.json")
                and block["signature"]["frozen_context"]["all_probabilities_frozen_before_block"]
                for block in blocks.values()), "evaluation does not reference frozen training")
    require(set(blocks["A"]["evaluation_seeds"]).isdisjoint(blocks["B"]["evaluation_seeds"]), "evaluation blocks share seeds")
    rows = evidence.load(folder/"results.json")
    require(summary["results_sha256"] == sha(folder/"results.json"), "result file seal mismatch")
    require(len(rows) == 6*len(config["checkpoints"]), "incomplete checkpoint/target/block evaluation grid")
    keys = {(r["target_id"],r["training_episodes"],r["eval_block"]) for r in rows}
    expected_keys = {(j,t,b) for j in range(3) for t in config["checkpoints"] for b in ("A","B")}
    require(keys == expected_keys, "duplicate/missing checkpoint result")
    verified = []
    for row in rows:
        target,t,b = row["target_id"],row["training_episodes"],row["eval_block"]
        p,q = config["target_probabilities"][target],sigmoid(checkpoints[target,t])
        block = blocks[b]
        means = block["conditional_return_means"]
        require(row["seed"] == seed and row["evaluation_block_sha256"] == sha(folder/"evaluation"/f"block_{b}.json"), "checkpoint evaluation provenance mismatch")
        expected = {"target_probability":p,"response_probability":q,"q_initial":.5,
                    "focal_return":weighted(means,p,q,"focal_return"),"response_return":weighted(means,p,q,"response_return"),
                    "frozen_focal_return":weighted(means,p,.5,"focal_return"),"frozen_response_return":weighted(means,p,.5,"response_return")}
        expected["response_own_gain"] = expected["response_return"]-expected["frozen_response_return"]
        expected["focal_response_effect"] = expected["focal_return"]-expected["frozen_focal_return"]
        expected["stag_minus_hare_response_advantage"] = weighted(means,p,1,"response_return")-weighted(means,p,0,"response_return")
        for field,value in expected.items():
            close(row[field],value,f"checkpoint estimate mismatch: {field}")
        verified.append(dict(row))
    return {"seed":seed,"config":config,"rows":verified,"blocks":blocks,"contexts":contexts,
            "counts":dict(counts),"frames":dict(frames),"sources_sha256":info["sources_sha256"],"freeze_sha256":sha(folder/"freeze_training.json")}

