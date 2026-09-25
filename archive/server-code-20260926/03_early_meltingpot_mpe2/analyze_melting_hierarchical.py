#!/usr/bin/env python3
"""Validate the complete frozen v2 batch, then summarize independent test seeds.

Incomplete/invalid batches produce a partial progress report and empty result
CSVs. They never produce estimates from a convenient subset of finished seeds.
No environments, model inference, training, plotting, or network calls occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

VERSION = "official_specialist_bernoulli_stratified_e5_v2"
METHODS = ("proxy_only", "random_hf", "global_voi", "paired_lucb", "pivot_voi")
BOOTSTRAP_DRAWS = 10000
STATISTICS_SEED = 20260915
REQUIRED_SOURCES = ("melting_hierarchical_e5_stratified.py", "melting_hierarchical_e5.py",
                    "melting_reference_diagnostic.py", "melting_behavior_diagnostic.py", "melting_e5_adapter.py")
LIMITATIONS = [
    "单任务、单轮、四候选；仅学习一维每局技能混合概率，低层共用一个冻结官方网络。",
    "这是沿作者E5选择规则、K4部分验证预算2的预声明外部检验，不是PDF预算4扫描的严格复现。",
    "每候选/目标/查询或审计流只有一次独立响应训练；四种技能组合各两局是条件评估，不是四个或八个独立响应训练重复。",
    "分层均值来自新真实环境对局；按冻结概率积分仍含Monte Carlo误差，不代表已知真实收益或真实选择遗憾。",
    "种子是统计单位，推断条件于同一官方低层网络及已拟合校准模型；bootstrap未重做低层训练或校准拟合。",
    "两个主比较使用双侧配对符号翻转检验及Holm校正；其有效性需要种子独立及零假设下差值的符号可交换/对称，不是随机分配处理的因果检验。",
    "预算曲线、机制及t0至t4响应学习曲线为描述性结果；学习曲线重用同一审计block，不是五次独立响应预算实验。",
    "所有方法在各自面板内先冻结后审计；不是全批次同时冻结。没有依据测试胜负增加训练、替换种子或调整分析。",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), label)


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def weighted(means, p, q, role="focal_return"):
    require(0 <= p <= 1 and 0 <= q <= 1, "invalid mixture probability")
    return sum((p if a else 1-p)*(q if b else 1-q)*means[f"{a}{b}"][role]
               for a in (0, 1) for b in (0, 1))


def metric_seed(name):
    return int(hashlib.sha256(f"{STATISTICS_SEED}:{name}".encode()).hexdigest()[:8], 16)


def mean_ci(values, name):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "invalid mean inputs")
    rng = np.random.default_rng(metric_seed(name))
    boot = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "n_seeds": len(values)}


def ratio_ci(numerators, denominators, name):
    n, d = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    require(n.shape == d.shape and n.ndim == 1 and (d >= n).all() and (n >= 0).all(), "invalid ratio inputs")
    if not d.sum():
        return {"estimate": None, "ci95": None, "numerator": 0, "denominator": 0, "n_seeds": len(n)}
    rng = np.random.default_rng(metric_seed(name))
    indices = rng.integers(len(n), size=(BOOTSTRAP_DRAWS, len(n)))
    denominators_boot = d[indices].sum(1)
    valid = denominators_boot > 0
    boot = n[indices].sum(1)[valid]/denominators_boot[valid]
    return {"estimate": float(n.sum()/d.sum()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "numerator": int(n.sum()), "denominator": int(d.sum()), "n_seeds": len(n),
            "undefined_bootstrap_draws": int((~valid).sum())}


def exact_sign_flip(differences):
    diffs = np.asarray(differences, dtype=float)
    require(np.isfinite(diffs).all(), "nonfinite paired differences")
    nonzero = diffs[diffs != 0]
    require(len(nonzero) <= 20, "exact sign-flip test supports at most 20 nonzero pairs")
    distribution = np.zeros(1)
    for value in np.abs(nonzero):
        distribution = np.concatenate((distribution+value, distribution-value))
    threshold = abs(float(diffs.sum()))
    tolerance = 64*np.finfo(float).eps*max(1.0, float(np.abs(diffs).sum()))
    probability = float(np.mean(np.abs(distribution) >= threshold-tolerance))
    return {"p_two_sided": probability, "nonzero_pairs": len(nonzero), "zero_pairs": int((diffs == 0).sum()),
            "enumerated_sign_patterns": len(distribution), "statistic": "absolute sum of paired native-gain differences",
            "zero_tolerance": "exact zero; numerical tolerance only when comparing enumerated sums"}


def holm(probabilities):
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    adjusted, previous = [0.0]*len(order), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, probabilities[index]*(len(order)-rank)))
        adjusted[index] = previous
    return adjusted


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def load(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "evidence path escaped the formal batch")
        self.hashes[str(path.relative_to(self.root))] = sha(path)
        return read(path)

    def checked(self, path, expected_hash):
        value = self.load(path)
        require(self.hashes[str(Path(path).resolve().relative_to(self.root))] == expected_hash,
                f"hash mismatch: {path}")
        return value


def inspect_panel(evidence, panel, phase, seed, protocol, sources, author_sources):
    info = evidence.load(panel/"protocol.json")
    config = info["config"]
    require(info.get("version") == VERSION and config.get("version") == VERSION, "not stratified v2")
    require(not config.get("diagnostic_only") and config["phase"] == phase and config["seed"] == seed,
            "diagnostic/phase/seed mismatch")
    k, response_steps = protocol["candidate_count"], protocol["response_train_episodes"]
    expected = {"task": protocol["task"], "k": k, "horizon": protocol["native_max_frames"],
                "candidate_train_episodes": protocol["candidate_train_episodes"], "response_train_episodes": response_steps,
                "stratum_replicates": protocol["stratified_replicates_per_skill_pair"],
                "proxy_stratum_replicates": protocol["proxy_stratified_replicates_per_skill_pair"],
                "candidate_learning_rate": protocol["learning_rate"], "response_learning_rate": protocol["learning_rate"],
                "reward_scale": protocol["reward_scale"], "reward_clip": protocol["reward_clip"],
                "old_logit": protocol["old_logit"], "opponent_logit": protocol["opponent_logit"],
                "budgets": protocol["budgets"], "methods": list(METHODS), "audit_mechanism": True,
                "baseline": "running_mean", "logit_clip": 8.0,
                "candidate_lr_multipliers": [.5, 1.0, 2.0, 1.0]}
    for key, value in expected.items():
        require(config.get(key) == value, f"panel setting differs from registered {key}")
    require(info["sources_sha256"] == {name: sources[name] for name in REQUIRED_SOURCES}, "panel source fingerprints differ")
    require(info["official_reference"]["network_instances"] == 1, "unexpected low-level network count")
    for name, expected_hash in author_sources.items():
        require(info["e5_source"]["sources"][name]["sha256"] == expected_hash, "author source fingerprint mismatch")
    state, summary = evidence.load(panel/"status.json"), evidence.load(panel/"summary.json")
    require(state["status"] == summary["status"] == "complete", "panel not complete")
    require(summary["version"] == VERSION and summary["seed"] == seed and summary["phase"] == phase, "summary identity mismatch")
    costs = evidence.load(panel/"episode_costs.json")
    seeds = evidence.load(panel/"seeds.json")
    require(len(set(seeds.values())) == len(seeds), "within-panel seed collision")
    expected_count = protocol[f"{phase}_episodes_per_panel"]
    require(len(costs) == summary["completed_physical_episodes"] == expected_count and not summary["incomplete_physical_attempts"], "physical episode count mismatch")
    native_rows, counts, frames = {}, Counter(), Counter()
    contexts = {}
    for relative, cost in costs.items():
        row = evidence.load(panel/relative)
        signature, key = row["signature"], row["signature"]["key"]
        require(row["complete"] and cost["complete"] and not row.get("event_parse_errors"), "invalid/incomplete native episode")
        require(signature["version"] == VERSION and row["train_seed"] == seed and row["task"] == protocol["task"], "native episode provenance mismatch")
        require(cost["key"] == key and cost["env_steps"] == row["env_steps"], "native frame ledger mismatch")
        require(0 < row["env_steps"] <= protocol["native_max_frames"], "native frame count outside limit")
        require(row["episode_id"] == digest(key)[:24], "episode identity mismatch")
        for field in ("focal_return", "response_return"):
            require(math.isfinite(row[field]), "nonfinite native return")
        template = signature["template"]
        for role in (0, 1):
            require(seeds[json.dumps([*template, "policy", role])] == signature["policy_seeds"][role], "policy seed differs from registry")
        require(seeds[json.dumps([*template, "environment"])] == signature["environment_seed"], "environment seed differs from registry")
        training = key[0] == "candidate_training" or "response_training" in key
        if training:
            require(signature["sampling_mode"] == "sampled_training" and signature["forced_specialists"] is None, "training used an evaluation stratum")
            for role, name in enumerate(("focal", "response")):
                mixture_seed = seeds[json.dumps([*template, "mixture", role])]
                require(signature["mixture_uniform_seeds"][role] == mixture_seed, "mixture seed mismatch")
                uniform = float(np.random.default_rng(mixture_seed).random())
                close(signature["mixture_uniforms"][role], uniform, "mixture uniform mismatch")
                p = sigmoid(signature[f"{name}_logit"])
                close(row[f"{name}_probability"], p, "sampled mixture probability mismatch")
                require(row["sampled_specialists"][role] == int(uniform < p), "incorrect Specialist draw")
        else:
            require(signature["sampling_mode"] == "forced_evaluation_stratum", "non-stratified evaluation episode")
            require(signature["mixture_uniform_seeds"] == [] and signature["focal_logit"] is None and signature["response_logit"] is None, "forced stratum approximated by a sampled logit")
            require(row["sampled_specialists"] == signature["forced_specialists"] and all(z in (0, 1) for z in signature["forced_specialists"]), "incorrect forced Specialist")
        require(signature["matchup"] == ["stag" if z else "hare" for z in row["sampled_specialists"]], "matchup identity mismatch")
        category = "training" if training else "evaluation"
        counts[category] += 1
        counts[f"stage:{key[0]}"] += 1
        frames[category] += row["env_steps"]
        frames[f"stage:{key[0]}"] += row["env_steps"]
        native_rows[row["episode_id"]] = row
        context = (signature["environment_seed"], *signature["policy_seeds"])
        # Sharing a template within a pair/block is deliberate; a joint context
        # shared between different templates/panels is checked separately.
        contexts.setdefault(context, set()).add(tuple(template))
    require(all(len(value) == 1 for value in contexts.values()), "different templates reused a full rollout random context")
    close(sum(cost["env_steps"] for cost in costs.values()), summary["physical_env_steps"], "summary frame count mismatch")
    close(summary["physical_env_steps"], state["physical_env_steps"], "status frame count mismatch")
    response_streams = 1 if phase == "calibration" else 2
    expected_training = k*config["candidate_train_episodes"] + response_streams*2*k*response_steps
    require(counts["training"] == expected_training and counts["evaluation"] == expected_count-expected_training, "training/evaluation episode split mismatch")
    require(counts["stage:proxy"] == 4*config["proxy_stratum_replicates"], "proxy block count mismatch")
    require(counts["stage:candidate_training"] == k*config["candidate_train_episodes"], "candidate-training count mismatch")
    require(counts["stage:calibration" if phase == "calibration" else "stage:selection"] == k*(2*response_steps+4*config["stratum_replicates"]), "response-query episode count mismatch")
    if phase == "test":
        require(counts["stage:audit"] == k*(2*response_steps+4*config["stratum_replicates"]), "audit episode count mismatch")

    def history(path, role, initial, other, length, rate, expected_key, expected_template, expected_hash=None):
        h = evidence.checked(path, expected_hash) if expected_hash else evidence.load(path)
        require(h["complete"] and len(h["history"]) == length and h["role"] == role, "training history incomplete")
        require(h["optimizer"] == "plain_SGD_no_momentum" and h["baseline_uses_only_previous_training_episodes"],
                "training optimizer/baseline differs from frozen protocol")
        close(h["initial_logit"], initial, "training branch initial logit mismatch")
        close(h["other_logit"], other, "training branch opponent mismatch")
        theta, reward_sum = initial, 0.0
        for i, step in enumerate(h["history"]):
            row = native_rows[step["episode_id"]]
            require(row["signature"]["sampling_mode"] == "sampled_training", "training history references evaluation")
            require(row["signature"]["key"] == [*expected_key, i] and
                    row["signature"]["template"] == [*expected_template, i], "training episode branch/template mismatch")
            close(row["signature"][f"{role}_logit"], theta, "training trajectory does not inherit its current logit")
            close(row["signature"]["response_logit" if role == "focal" else "focal_logit"], other,
                  "training rollout opponent differs from frozen branch target")
            raw = row[f"{role}_return"]
            scaled = float(np.clip(raw/config["reward_scale"], -config["reward_clip"], config["reward_clip"]))
            baseline = reward_sum/i if i else 0.0
            p = sigmoid(theta)
            z = row["sampled_specialists"][0 if role == "focal" else 1]
            gradient = (scaled-baseline)*(z-p)
            updated = float(np.clip(theta+rate*gradient, -config["logit_clip"], config["logit_clip"]))
            for key, expected_value in (("logit_before", theta), ("probability", p), ("native_return", raw),
                ("baseline_before", baseline), ("scaled_clipped_return", scaled), ("advantage", scaled-baseline),
                ("gradient", gradient), ("learning_rate", rate), ("logit_after", updated)):
                close(step[key], expected_value, f"REINFORCE history {key} mismatch")
            theta, reward_sum = updated, reward_sum+scaled
        close(h["final_logit"], theta, "final trained logit mismatch")
        return h

    candidate = evidence.load(panel/"candidates.json")
    require(candidate["version"] == VERSION and len(candidate["candidate_logits"]) == k, "candidate bank mismatch")
    for j, logit in enumerate(candidate["candidate_logits"]):
        h = history(panel/"training"/(digest(("candidate_training", j))[:24]+".json"), "focal", config["old_logit"],
                    config["opponent_logit"], config["candidate_train_episodes"],
                    config["candidate_learning_rate"]*config["candidate_lr_multipliers"][j % 4],
                    ("candidate_training", j), ("candidate_training", j))
        close(logit, h["final_logit"], "candidate does not match trained state")
        close(candidate["candidate_probabilities"][j], sigmoid(logit), "candidate probability mismatch")

    def block(path, replicas, expected_hash=None):
        b = evidence.checked(path, expected_hash) if expected_hash else evidence.load(path)
        require(b["version"] == VERSION and b["complete"] and not b["training_reads_this_block"], "invalid evaluation block provenance")
        require(b["signature"]["config_sha256"] == digest(config), "evaluation block configuration mismatch")
        require(len(b["rows"]) == 4*replicas and b["replicates_per_stratum"] == replicas, "stratum episode count mismatch")
        groups = {}
        for row in b["rows"]:
            require(row == native_rows[row["episode_id"]], "block row differs from persisted native episode")
            require(row["signature"]["sampling_mode"] == "forced_evaluation_stratum", "block uses sampled training data")
            a, bb, r = row["signature"]["key"][-3:]
            require(row["signature"]["key"] == [*b["signature"]["key"], a, bb, r] and
                    row["signature"]["template"] == [*b["signature"]["template"], r], "evaluation block reused another stream or random template")
            require([a, bb] == row["sampled_specialists"] and 0 <= r < replicas, "stratum key mismatch")
            require((a, bb, r) not in groups, "duplicate stratum episode")
            groups[a, bb, r] = row
        for a in (0, 1):
            for bb in (0, 1):
                for role in ("focal_return", "response_return"):
                    expected_mean = float(np.mean([groups[a, bb, r][role] for r in range(replicas)]))
                    close(b["conditional_return_means"][f"{a}{bb}"][role], expected_mean, "conditional mean differs from real returns")
        close(b["actual_env_frames"], sum(row["env_steps"] for row in b["rows"]), "block actual frame count mismatch")
        return b

    proxy_block = block(panel/"proxy"/"evaluation_block.json", config["proxy_stratum_replicates"])
    proxies = evidence.load(panel/"proxy.json")
    features = evidence.load(panel/"features.json")
    require(len(proxies) == len(features) == k, "proxy/features row count mismatch")
    p_old, q_zero = sigmoid(config["old_logit"]), sigmoid(config["opponent_logit"])
    for j, p_new in enumerate(candidate["candidate_probabilities"]):
        expected_proxy = weighted(proxy_block["conditional_return_means"], p_new, q_zero)-weighted(proxy_block["conditional_return_means"], p_old, q_zero)
        close(features[j][0], expected_proxy, "proxy not derived from its independent real block")
        close(proxies[j]["delta"], expected_proxy, "proxy artifact disagreement")
        close(features[j][1], p_new-p_old, "signed mixture feature mismatch")
        close(features[j][2], abs(p_new-p_old), "absolute mixture feature mismatch")

    pairs, traces = {}, []
    streams = ("calibration",) if phase == "calibration" else ("selection", "audit")
    for stream in streams:
        for j, logit in enumerate(candidate["candidate_logits"]):
            pair_path = panel/stream/f"candidate_{j}"/"paired.json"
            pair = evidence.load(pair_path)
            require(pair["version"] == VERSION and pair["stream"] == stream and pair["candidate"] == j, "response pair identity mismatch")
            require(pair["signature"]["config_sha256"] == digest(config), "paired response configuration mismatch")
            require(pair["independent_response_training_replicates_per_branch"] == 1, "response-training replicate count mislabeled")
            require(pair["logical_query_episodes"] == 2*response_steps+4*config["stratum_replicates"], "query cost units mismatch")
            b = block(pair_path.with_name("evaluation_block.json"), config["stratum_replicates"], pair["evaluation_block"]["sha256"])
            require(pair["conditional_return_means"] == b["conditional_return_means"], "pair conditional returns differ from block")
            histories = {}
            for branch, focal_logit in (("old", config["old_logit"]), ("new", logit)):
                reference = pair["response_training_histories"][branch]
                path = panel/"training"/(digest((stream, j, "response_training", branch))[:24]+".json")
                require(Path(reference["path"]).name == path.name, "history path does not identify the expected branch")
                histories[branch] = history(path, "response", config["opponent_logit"], focal_logit,
                                            response_steps, config["response_learning_rate"],
                                            (stream, j, "response_training", branch), (stream, j, "response_training"), reference["sha256"])
                close(pair[f"{branch}_response_logit"], histories[branch]["final_logit"], "response endpoint differs from trained state")
            p_new = sigmoid(logit)
            q_old, q_new = sigmoid(pair["old_response_logit"]), sigmoid(pair["new_response_logit"])
            require(b["signature"]["key"] == b["signature"]["template"] == [stream, j, "stratified_evaluation"],
                    "response block stream differs from its pair")
            context = b["signature"]["frozen_context"]
            require(context["all_probabilities_frozen_before_block"], "evaluation probabilities not frozen")
            for name, expected_probability in (("p_old", p_old), ("p_new", p_new), ("q_old", q_old), ("q_new", q_new), ("q_initial", q_zero)):
                close(context[name], expected_probability, "evaluation block frozen probability mismatch")
            means = b["conditional_return_means"]
            for name, p, q in (("old_estimate", p_old, q_old), ("new_estimate", p_new, q_new)):
                estimate = pair[name]
                close(estimate["focal_probability"], p, "frozen focal probability mismatch")
                close(estimate["response_probability"], q, "frozen response probability mismatch")
                for role in ("focal_return", "response_return"):
                    close(estimate[role], weighted(means, p, q, role), "weighted expectation mismatch")
            expected_delta = weighted(means, p_new, q_new)-weighted(means, p_old, q_old)
            close(pair["delta"], expected_delta, "paired gain mismatch")
            frozen_delta = weighted(means, p_new, q_zero)-weighted(means, p_old, q_zero)
            training_frames = sum(cost["env_steps"] for cost in costs.values() if tuple(cost["key"][:3]) == (stream, j, "response_training"))
            close(pair["total_env_steps"], training_frames+b["actual_env_frames"], "query physical frames mismatch")
            expected_train_seeds = [seeds[json.dumps([stream, j, "response_training", i, "environment"])] for i in range(response_steps)]
            require(pair["training_seeds"] == expected_train_seeds and pair["evaluation_seeds"] == b["evaluation_seeds"], "paired seed metadata mismatch")
            if stream == "audit":
                close(pair["frozen_delta"], frozen_delta, "frozen audit delta mismatch")
                close(pair["response_effect"], expected_delta-frozen_delta, "response effect mismatch")
                for branch, p, q in (("old", p_old, q_old), ("new", p_new, q_new)):
                    close(pair[f"{branch}_response_own_gain"], weighted(means, p, q, "response_return")-weighted(means, p, q_zero, "response_return"), "response-own gain mismatch")
                close(pair["candidate_specific_effect"], weighted(means, p_new, q_new)-weighted(means, p_new, q_old), "candidate-specific effect mismatch")
                old_progress = [config["opponent_logit"]]+[step["logit_after"] for step in histories["old"]["history"]]
                new_progress = [config["opponent_logit"]]+[step["logit_after"] for step in histories["new"]["history"]]
                for t, (o, n) in enumerate(zip(old_progress, new_progress)):
                    gain = weighted(means, p_new, sigmoid(n))-weighted(means, p_old, sigmoid(o))
                    traces.append({"seed": seed, "candidate": j, "response_training_episodes": t,
                                   "p_old": p_old, "p_new": p_new, "q_old": sigmoid(o), "q_new": sigmoid(n),
                                   "audit_gain_at_saved_checkpoint": gain, "gain_change_from_t0": gain-frozen_delta,
                                   "shared_audit_block": True})
            pairs[stream, j] = pair
    base = {"phase": phase, "seed": seed, "counts": dict(counts), "frames": dict(frames),
            "physical_env_steps": summary["physical_env_steps"], "contexts": list(contexts),
            "contract": info["calibration_contract"], "info": info}
    if phase == "calibration":
        cal = evidence.load(panel/"calibration_summary.json")
        require(cal["version"] == VERSION and cal["seed"] == seed and len(cal["rows"]) == k, "calibration summary mismatch")
        require(cal["calibration_contract"] == info["calibration_contract"], "calibration contract mismatch")
        for j, row in enumerate(cal["rows"]):
            require(row["seed"] == seed and row["candidate"] == j and row["features"] == features[j], "calibration row mismatch")
            close(row["target_correction"], pairs["calibration", j]["delta"]-features[j][0], "calibration target mismatch")
        base["calibration_rows"] = cal["rows"]
        return base
    require(info["calibration_seeds"] == protocol["calibration_seeds"], "test did not use every planned calibration seed")
    expected_cal_hashes = {s: sha(evidence.root/"calibration"/f"seed_{s}"/"calibration_summary.json") for s in protocol["calibration_seeds"]}
    supplied = {int(Path(path).parent.name.removeprefix("seed_")): value for path, value in info["calibration_input_hashes"].items()}
    require(supplied == expected_cal_hashes, "test calibration input hashes mismatch")
    seal = evidence.load(panel/"selection_seal.json")
    decisions = evidence.checked(panel/"decisions_frozen.json", seal["decisions_sha256"])
    for field, name in (("candidate_sha256", "candidates.json"), ("features_sha256", "features.json"), ("posterior_sha256", "posterior.json")):
        evidence.checked(panel/name, seal[field])
    require(seal["audit_opened"] is False and seal["audit_generated"] is False and seal["n_decisions"] == 17,
            "selection seal does not cover all decisions before audit")
    require(seal["calibration_input_hashes"] == info["calibration_input_hashes"], "seal calibration identity mismatch")
    close(seal["physical_env_steps_at_freeze"], sum(cost["env_steps"] for cost in costs.values() if cost["key"][0] != "audit"), "audit/freeze physical order mismatch")
    isolation = evidence.load(panel/"isolation.json")
    require(isolation["seed_registry_sha256"] == sha(panel/"seeds.json"), "seed registry changed after isolation check")
    selection_seeds = {v for key, v in seeds.items() if json.loads(key)[0] == "selection"}
    audit_seeds = {v for key, v in seeds.items() if json.loads(key)[0] == "audit"}
    require(not selection_seeds & audit_seeds and isolation["overlap_count"] == 0, "query/audit seeds overlap")
    require(isolation["selection_seed_count"] == len(selection_seeds) and isolation["audit_seed_count"] == len(audit_seeds), "isolation count mismatch")
    expected_keys = {(method, b) for method in METHODS for b in ((0,) if method == "proxy_only" else protocol["budgets"])}
    require(len(decisions) == 17 and {(d["method"], d["budget"]) for d in decisions} == expected_keys, "incomplete/duplicate method-budget grid")
    scored = evidence.load(panel/"scored_decisions.json")
    require(len(scored) == 17 and summary["decision_records"] == scored, "scored decision summary mismatch")
    score_map = {(d["method"], d["budget"]): d for d in scored}
    require(set(score_map) == expected_keys, "scored method grid differs")
    posterior = evidence.load(panel/"posterior.json")
    raw = np.asarray(features)
    transformed = np.c_[np.ones(k), (raw[:, 1:]-posterior["feature_mean"])/posterior["feature_scale"]]
    corrected = raw[:, 0]+transformed@np.asarray(posterior["mean"])
    method_rows = []
    for d in decisions:
        require(d["version"] == VERSION and d["seed"] == seed and d["task"] == protocol["task"], "decision identity mismatch")
        ids = d["queried_ids"]
        require(len(set(ids)) == len(ids) == d["query_packages_used"] == (0 if d["method"] == "proxy_only" else d["budget"]), "fixed-budget query count mismatch")
        require(all(str(j) in map(str, range(k)) for j in ids), "invalid queried candidate")
        close(d["query_cost"], len(ids)*(2*response_steps+4*config["stratum_replicates"]), "logical query cost mismatch")
        close(d["logical_query_episodes"], d["query_cost"], "query episode units mismatch")
        close(d["logical_query_env_frames"], sum(pairs["selection", int(j)]["total_env_steps"] for j in ids), "logical frame charge mismatch")
        require(len(d["query_ledger"]) == len(ids), "query ledger count mismatch")
        for entry, j in zip(d["query_ledger"], ids):
            require(entry["transition_id"] == j, "query ledger order mismatch")
            pair = pairs["selection", int(j)]
            close(entry["delta"], pair["delta"], "query ledger response mismatch")
            reference = entry["input_files"][0]
            require(reference["sha256"] == sha(panel/"selection"/f"candidate_{j}"/"paired.json"), "query file changed after selection")
        estimates = {}
        for j in range(k):
            expected_estimate = pairs["selection", j]["delta"] if str(j) in ids else raw[j, 0] if d["method"] in ("proxy_only", "random_hf") else corrected[j]
            estimates[str(j)] = float(expected_estimate)
            close(d["estimates"][str(j)], expected_estimate, "decision used wrong unqueried-candidate prediction")
        selected = max(estimates, key=estimates.get)
        require(d["selected_candidate_id"] == selected, "selected candidate does not maximize the E5 estimates")
        s = score_map[d["method"], d["budget"]]
        require(all(s[key] == value for key, value in d.items()), "scored result mutated frozen decision")
        pair = pairs["audit", int(selected)]
        reference = s["audit_provenance"]
        require(reference["sha256"] == sha(panel/"audit"/f"candidate_{selected}"/"paired.json"), "audit source hash mismatch")
        require(reference["response_training_independent"] and reference["evaluation_independent_of_selection"], "missing independent-audit claim")
        close(s["audit_gain"], pair["delta"], "selected audit gain mismatch")
        close(s["selected_probability"], candidate["candidate_probabilities"][int(selected)], "selected probability mismatch")
        for field in ("frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect"):
            close(s[field], pair[field], "scored mechanism mismatch")
        method_rows.append({key: s[key] for key in ("seed", "method", "budget", "selected", "selected_probability", "audit_gain",
                           "query_packages_used", "logical_query_episodes", "logical_query_env_frames", "frozen_delta", "response_effect",
                           "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect")})
    candidate_rows = []
    for j in range(k):
        pair = pairs["audit", j]
        candidate_rows.append({"seed": seed, "candidate": j, "p_old": p_old, "p_new": candidate["candidate_probabilities"][j],
                               "p_change": candidate["candidate_probabilities"][j]-p_old, "proxy_delta": features[j][0],
                               "audit_gain": pair["delta"], "q_initial": q_zero,
                               "q_old_final": sigmoid(pair["old_response_logit"]), "q_new_final": sigmoid(pair["new_response_logit"]),
                               **{field: pair[field] for field in ("frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect")}})
    base.update(method_rows=method_rows, candidate_rows=candidate_rows, traces=traces, posterior=posterior)
    return base


def aggregate(test_panels, protocol, accounting):
    methods = [row for panel in test_panels for row in panel["method_rows"]]
    candidates = [row for panel in test_panels for row in panel["candidate_rows"]]
    traces = [row for panel in test_panels for row in panel["traces"]]
    method_summary = []
    for method in METHODS:
        for budget in ((0,) if method == "proxy_only" else protocol["budgets"]):
            rows = sorted([r for r in methods if r["method"] == method and r["budget"] == budget], key=lambda r: r["seed"])
            record = {"method": method, "budget": budget, "n_seeds": len(rows)}
            for field in ("audit_gain", "selected_probability", "logical_query_episodes", "logical_query_env_frames",
                          "frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect"):
                record[field] = mean_ci([r[field] for r in rows], f"method:{method}:{budget}:{field}")
            method_summary.append(record)
    index = {(r["seed"], r["method"], r["budget"]): r for r in methods}
    primary, primary_seed_rows = [], []
    for reference in ("proxy_only", "random_hf"):
        reference_budget = 0 if reference == "proxy_only" else 2
        rows = []
        for seed in protocol["test_seeds"]:
            p, r = index[seed, "pivot_voi", 2], index[seed, reference, reference_budget]
            row = {"seed": seed, "contrast": f"pivot_voi_minus_{reference}", "primary_budget": 2,
                   "pivot_audit_gain": p["audit_gain"], "reference_audit_gain": r["audit_gain"],
                   "paired_gain_difference": p["audit_gain"]-r["audit_gain"],
                   "pivot_logical_query_episodes": p["logical_query_episodes"],
                   "reference_logical_query_episodes": r["logical_query_episodes"],
                   "selection_changed": p["selected"] != r["selected"]}
            rows.append(row)
            primary_seed_rows.append(row)
        differences = [r["paired_gain_difference"] for r in rows]
        primary.append({"contrast": f"pivot_voi_minus_{reference}", "budget": 2,
                        "difference": mean_ci(differences, f"primary:{reference}"), "paired_seed_differences": differences,
                        "test_seeds": protocol["test_seeds"], "test": exact_sign_flip(differences),
                        "selection_changed_seeds": sum(r["selection_changed"] for r in rows),
                        "cost_scope": "additional validation versus zero queries" if reference == "proxy_only" else "equal declared episode query budget"})
    adjusted = holm([r["test"]["p_two_sided"] for r in primary])
    for row, value in zip(primary, adjusted):
        row["test"]["holm_p_two_tests"] = value
    metric_panels, mechanism_panels = [], []
    for panel in test_panels:
        rows = panel["candidate_rows"]
        proxy, audit = np.array([r["proxy_delta"] for r in rows]), np.array([r["audit_gain"] for r in rows])
        valid, positive = (proxy != 0) & (audit != 0), proxy > 0
        metric_panels.append({"seed": panel["seed"], "IDE": float(np.abs(proxy-audit).mean()),
                              "ISC_numerator": int(((np.sign(proxy) == np.sign(audit)) & valid).sum()), "ISC_denominator": int(valid.sum()),
                              "IRR_numerator": int((positive & (audit < 0)).sum()), "IRR_denominator": int(positive.sum())})
        mechanism_panels.append({"seed": panel["seed"], **{field: float(np.mean([r[field] for r in rows])) for field in (
            "p_change", "frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect")}})
    metrics = {"IDE": mean_ci([r["IDE"] for r in metric_panels], "candidate:IDE"),
               "ISC": ratio_ci([r["ISC_numerator"] for r in metric_panels], [r["ISC_denominator"] for r in metric_panels], "candidate:ISC"),
               "IRR": ratio_ci([r["IRR_numerator"] for r in metric_panels], [r["IRR_denominator"] for r in metric_panels], "candidate:IRR"),
               "definition": "ISC removes exact-zero proxy/audit ties; IRR counts negative noisy audit estimates among positive proxy estimates. Ratios bootstrap full seed clusters."}
    mechanism = {field: mean_ci([r[field] for r in mechanism_panels], f"mechanism:{field}") for field in mechanism_panels[0] if field != "seed"}
    progress = []
    for t in range(protocol["response_train_episodes"]+1):
        record = {"response_training_episodes": t}
        for field in ("audit_gain_at_saved_checkpoint", "gain_change_from_t0", "q_old", "q_new"):
            per_seed = [float(np.mean([r[field] for r in traces if r["seed"] == seed and r["response_training_episodes"] == t])) for seed in protocol["test_seeds"]]
            record[field] = mean_ci(per_seed, f"response_progress:{t}:{field}")
        progress.append(record)
    return {"methods": method_summary, "primary_contrasts": primary, "candidate_metrics": metrics,
            "candidate_mechanism_means": mechanism, "response_learning_progress": progress,
            "accounting": accounting, "method_seed_results": methods, "candidate_seed_results": candidates,
            "primary_seed_differences": primary_seed_rows, "candidate_metric_seed_results": metric_panels,
            "response_learning_seed_results": traces}


def analyze(batch_root, source_dir, author_root):
    evidence = Evidence(batch_root)
    errors, progress, panels = [], [], []
    report = {"status": "partial", "formal_effects_available": False, "analysis_version": "stratified_v2_analysis_1",
              "statistics": {"bootstrap_draws": BOOTSTRAP_DRAWS, "fixed_seed": STATISTICS_SEED,
                             "bootstrap_unit": "test training/environment seed", "primary_budget": 2,
                             "primary_tests": "two-sided exact paired sign flips; Holm over two predeclared contrasts"},
              "limitations": LIMITATIONS, "validation_errors": errors, "panel_progress": progress}
    try:
        protocol, state = evidence.load(evidence.root/"protocol.json"), evidence.load(evidence.root/"status.json")
        report["registered_protocol"] = protocol
        report["supervisor_status"] = state["status"]
        require(len(protocol["calibration_seeds"]) == 8 and len(protocol["test_seeds"]) == 20, "formal plan must contain 8 calibration and 20 test seeds")
        require(len(set(protocol["calibration_seeds"]+protocol["test_seeds"])) == 28, "registered seeds are duplicate or overlapping")
        require(protocol["candidate_count"] == 4 and protocol["budgets"] == [0, 1, 2, 4], "wrong formal K/budget grid")
        require(protocol["total_planned_episodes"] == 3744 and protocol["calibration_episodes_per_panel"] == 88 and protocol["test_episodes_per_panel"] == 152, "not the frozen stratified episode budget")
        require(protocol["stratified_replicates_per_skill_pair"] == protocol["proxy_stratified_replicates_per_skill_pair"] == 2, "wrong stratified replicate count")
        ready = state["status"] == "complete"
        planned_jobs = {(phase, s) for phase in ("calibration", "test") for s in protocol[f"{phase}_seeds"]}
        jobs = state.get("jobs", [])
        if len(jobs) != 28 or {(j["phase"], j["seed"]) for j in jobs} != planned_jobs or any(j["returncode"] != 0 for j in jobs):
            errors.append("supervisor does not confirm all 28 planned jobs completed successfully")
            ready = False
        for phase, seed in sorted(planned_jobs):
            panel = evidence.root/phase/f"seed_{seed}"
            path = panel/"status.json"
            status = evidence.load(path).get("status") if path.exists() else "missing"
            progress.append({"phase": phase, "seed": seed, "status": status})
            if status != "complete" or not (panel/"summary.json").exists():
                ready = False
        if not ready:
            errors.append("batch incomplete: no subset-based formal effect estimates were computed")
            return report, evidence.hashes
        sources = protocol["sources_sha256"]
        source_checks = {}
        for name, expected_hash in sources.items():
            path = Path(source_dir)/name
            require(path.is_file() and sha(path) == expected_hash, f"registered source content mismatch: {name}")
            source_checks[name] = expected_hash
        author_paths = {"experiments.v9.e5c_efficiency": Path(author_root)/"experiments/v9/e5c_efficiency.py",
                        "pivot.acquisition.pivot_voi": Path(author_root)/"src/pivot/acquisition/pivot_voi.py"}
        author_sources = {name: sha(path) for name, path in author_paths.items()}
        report["verified_source_hashes"] = source_checks
        report["verified_author_source_hashes"] = author_sources
        for phase, seed in sorted(planned_jobs):
            try:
                panels.append(inspect_panel(evidence, evidence.root/phase/f"seed_{seed}", phase, seed, protocol, sources, author_sources))
            except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
                errors.append(f"{phase}/seed_{seed}: {type(exc).__name__}: {exc}")
        if errors or len(panels) != 28:
            return report, evidence.hashes
        contract = panels[0]["contract"]
        require(all(p["contract"] == contract for p in panels), "cross-panel model/learning/evaluation contracts differ")
        seen_contexts = set()
        for panel in panels:
            contexts = set(panel["contexts"])
            require(not seen_contexts & contexts, "different independent panels reused a full rollout random context")
            seen_contexts.update(contexts)
        tests = sorted([p for p in panels if p["phase"] == "test"], key=lambda p: p["seed"])
        require(all(p["posterior"] == tests[0]["posterior"] for p in tests), "test panels do not share the frozen independent calibration posterior")
        # Independently reconstruct the author E5 Gaussian fit from calibration
        # labels only, including the adapter's add/subtract target arithmetic.
        calibration_rows = [row for panel in panels if panel["phase"] == "calibration" for row in panel["calibration_rows"]]
        raw = np.asarray([row["features"] for row in calibration_rows])
        center, scale = raw[:, 1:].mean(0), raw[:, 1:].std(0)
        scale[scale < 1e-8] = 1.0
        design = np.c_[np.ones(len(raw)), (raw[:, 1:]-center)/scale]
        target = np.asarray([(row["features"][0]+row["target_correction"])-row["features"][0] for row in calibration_rows])
        noise = max(float(np.var(target)), 1e-4)
        covariance = np.linalg.inv(np.eye(design.shape[1])+design.T@design/noise)
        mean = covariance@design.T@target/noise
        posterior = tests[0]["posterior"]
        for field, expected_array in (("feature_mean", center), ("feature_scale", scale), ("mean", mean), ("covariance", covariance)):
            require(np.allclose(posterior[field], expected_array, rtol=1e-8, atol=1e-8), "posterior was not fit to only the independent calibration records")
        close(posterior["noise_variance"], noise, "calibration noise rule mismatch")
        require(posterior["prior_precision"] == 1 and posterior["n_observations"] == 32 and not posterior["test_labels_used"],
                "posterior prior/count/test-label provenance mismatch")
        accounting = {"completed_real_episodes": sum(p["counts"]["training"]+p["counts"]["evaluation"] for p in panels),
                      "training_episodes": sum(p["counts"]["training"] for p in panels),
                      "evaluation_episodes": sum(p["counts"]["evaluation"] for p in panels),
                      "physical_env_frames": sum(p["physical_env_steps"] for p in panels),
                      "training_env_frames": sum(p["frames"]["training"] for p in panels),
                      "evaluation_env_frames": sum(p["frames"]["evaluation"] for p in panels),
                      "calibration_panels": 8, "test_panels": 20,
                      "by_phase": {phase: {"training_episodes": sum(p["counts"]["training"] for p in panels if p["phase"] == phase),
                                           "evaluation_episodes": sum(p["counts"]["evaluation"] for p in panels if p["phase"] == phase),
                                           "physical_env_frames": sum(p["physical_env_steps"] for p in panels if p["phase"] == phase)}
                                   for phase in ("calibration", "test")},
                      "cost_note": "Physical episodes count each cached rollout once. Each method pays its full logical query episode/frame cost; audit/calibration/proxy/candidate generation are separate overhead."}
        require(accounting["completed_real_episodes"] == 3744 and accounting["training_episodes"] == 1984 and accounting["evaluation_episodes"] == 1760, "batch episode allocation does not reconcile")
        report.update(aggregate(tests, protocol, accounting), status="complete", formal_effects_available=True,
                      validation_passed_panels=28, all_planned_test_seeds_used=True)
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        errors.append(f"batch: {type(exc).__name__}: {exc}")
    return report, evidence.hashes


def csv_rows(path, rows, empty_fields=("status",)):
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(empty_fields)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_methods(rows):
    result = []
    for row in rows:
        flat = {key: row[key] for key in ("method", "budget", "n_seeds")}
        for key, value in row.items():
            if isinstance(value, dict) and "ci95" in value:
                flat.update({f"{key}_mean": value["mean"], f"{key}_ci_low": value["ci95"][0], f"{key}_ci_high": value["ci95"][1]})
        result.append(flat)
    return result


def render_markdown(report):
    if report["status"] != "complete":
        complete = sum(row["status"] == "complete" for row in report["panel_progress"])
        return "# Melting Pot 分层混合策略实验：尚未完成验收\n\n" + \
               f"当前计划面板完成记录为 {complete}/28。整批完整性或来源检查尚未全部通过，因此没有计算已完成种子子集的正式方法效果。\n\n" + \
               "\n".join(f"- {error}" for error in report["validation_errors"]) + \
               "\n\n[进度与完整检查记录](summary.json)。原始对局保留在服务器。\n"
    lines = ["# Melting Pot 分层混合策略实验结果", "",
        "正式8个校准面板和20个测试面板均完成；全部计划种子、来源哈希、训练/评估分工、查询与审计隔离、选择冻结及原始回报重算检查通过。",
        "", "本批沿用作者E5选择规则，检验K4、部分验证预算2的外部表现；保留预算4全候选验证结果，不称为严格复现PDF预算4扫描。",
        "", "每局由已知概率选定stag/hare技能并保持整局，官方低层网络冻结；上层使用真实回报REINFORCE。四种技能组合各两局新真实对局形成分层均值，再按已冻结概率积分。", "",
        "| 预算2主比较（PIVOT减对照） | 部署增益差均值 [95%区间] | 双侧符号翻转p | 两比较Holm p |", "|---|---:|---:|---:|"]
    for row in report["primary_contrasts"]:
        d, t = row["difference"], row["test"]
        label = "Proxy Only（额外验证成本）" if "proxy_only" in row["contrast"] else "Random-HF（同对局查询成本）"
        lines.append(f"| {label} | {d['mean']:.6g} [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}] | {t['p_two_sided']:.6g} | {t['holm_p_two_tests']:.6g} |")
    a = report["accounting"]
    lines += ["", f"共执行 {a['completed_real_episodes']:,} 局真实环境交互：训练 {a['training_episodes']:,} 局、评估 {a['evaluation_episodes']:,} 局；合计 {a['physical_env_frames']:,} 环境帧。每个查询固定16局，实际帧数随原生终止变化；各方法分别支付完整逻辑成本，缓存节省仅体现在物理总量。", "",
              "统计以20个测试种子为单位，固定随机种子进行10,000次bootstrap。两个主检验为精确双侧配对符号翻转并做Holm校正；未显著不等于等效，正均值也不自动意味着稳定优势。", "",
              "IDE/ISC/IRR针对含噪分层审计估计。ISC移除恰为零的符号平局；IRR不是逐候选确认的真实有害升级率。机制与t0–t4学习曲线在同一独立audit block上重加权，具有相关性且没有新增对局。", "",
              "每候选、每目标分支、每查询/审计流只有一次独立响应训练；四组合各两局不能写成四次或八次独立响应训练。", "",
              "[全部方法与预算](method_summary.csv) · [逐测试种子结果](method_seed_results.csv) · [主比较逐种子差值](primary_seed_differences.csv) · [候选与机制](candidate_seed_results.csv) · [响应学习检查点](response_learning_seed_results.csv) · [完整汇总](summary.json) · [证据哈希清单](manifest.json)", ""]
    lines += ["结论范围："]+[f"- {item}" for item in LIMITATIONS]+["", "原始模型、对局和大日志保留在服务器；本汇总及CSV可独立复算主比较，不需要访问云端绝对路径。", ""]
    return "\n".join(lines)


def save_outputs(output, report, hashes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write(output/"summary.json", report)
    mappings = {"method_seed_results.csv": "method_seed_results", "candidate_seed_results.csv": "candidate_seed_results",
                "primary_seed_differences.csv": "primary_seed_differences", "candidate_metric_seed_results.csv": "candidate_metric_seed_results",
                "response_learning_seed_results.csv": "response_learning_seed_results"}
    for name, key in mappings.items():
        csv_rows(output/name, report.get(key, []))
    csv_rows(output/"method_summary.csv", flatten_methods(report.get("methods", [])))
    csv_rows(output/"panel_progress.csv", report["panel_progress"])
    (output/"中文结果.md").write_text(render_markdown(report))
    names = ["summary.json", "中文结果.md", "method_summary.csv", "panel_progress.csv", *mappings]
    write(output/"manifest.json", {"status": report["status"], "analysis_source_sha256": sha(__file__),
          "input_paths_relative_to_formal_batch": hashes, "output_sha256": {name: sha(output/name) for name in names},
          "registered_source_hashes": report.get("verified_source_hashes", {}),
          "author_source_hashes": report.get("verified_author_source_hashes", {}),
          "note": "Original input artifacts remain on the server; result CSVs and summary contain the data required to reproduce the primary comparisons."})


def self_test():
    close(exact_sign_flip([1, 1])["p_two_sided"], .5, "exact small sign-flip test")
    close(exact_sign_flip([1, -1])["p_two_sided"], 1, "two-sided centered sign-flip test")
    close(exact_sign_flip([0, 0])["p_two_sided"], 1, "all-zero paired test")
    close(exact_sign_flip([1]*20)["p_two_sided"], 2/(2**20), "20-pair exact enumeration")
    require(holm([.01, .04]) == [.02, .04], "Holm adjustment")
    require(mean_ci([3]*20, "constant")["ci95"] == [3, 3], "constant bootstrap")
    ratio = ratio_ci([1]*20, [2]*20, "constant_ratio")
    require(ratio["ci95"] == [.5, .5] and ratio["estimate"] == .5, "clustered ratio")
    means = {f"{a}{b}": {"focal_return": 3+5*a+7*b+11*a*b} for a in (0, 1) for b in (0, 1)}
    close(weighted(means, .23, .67), 3+5*.23+7*.67+11*.23*.67, "independent weighted-estimator check")
    with tempfile.TemporaryDirectory(prefix="hierarchical-analysis-selfcheck-") as tmp:
        root = Path(tmp)
        report, hashes = analyze(root, root, root)
        require(report["status"] == "partial" and not report["formal_effects_available"] and "methods" not in report, "missing batch must not produce effects")
        save_outputs(root/"outputs", report, hashes)
        require("尚未完成" in (root/"outputs"/"中文结果.md").read_text(), "partial report")
    # Exercise the actual persisted v2 schema, selection seal and provenance
    # validators. The fixture is confined to a disposable directory and cannot
    # be selected by the analysis CLI or included in user benchmark outputs.
    source_dir = Path(__file__).resolve().parent
    author_root = source_dir.parent/"colin_pivot"
    for path in (source_dir, author_root, author_root/"src"):
        sys.path.insert(0, str(path))
    import melting_hierarchical_e5_stratified as runner
    class CheckRollout:
        metadata = {"implementation": "internal_control_flow_fixture_only", "network_instances": 1}
        def episode(self, *, matchup, env_seed, policy_seeds, deadline, allowance):
            a, b = [int(name == "stag") for name in matchup]
            noise = (env_seed % 19)/19
            return {"complete": True, "env_steps": 3, "focal_return": float(3+5*a+7*b+11*a*b+noise),
                    "response_return": float(2+13*a-3*b+2*a*b+noise), "seconds": .001,
                    "interaction_count": 1, "event_parse_errors": []}, []
    with tempfile.TemporaryDirectory(prefix="analysis-complete-schema-selfcheck-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        root = Path(tmp)
        protocol = {"task": "melting_stag", "candidate_count": 4, "budgets": [0, 1, 2, 4],
                    "calibration_seeds": list(range(31000, 31008)), "test_seeds": list(range(32000, 32020)),
                    "candidate_train_episodes": 4, "response_train_episodes": 4,
                    "stratified_replicates_per_skill_pair": 2, "proxy_stratified_replicates_per_skill_pair": 2,
                    "learning_rate": .5, "reward_scale": 100, "reward_clip": 2,
                    "old_logit": 0, "opponent_logit": 0, "native_max_frames": 2500,
                    "calibration_episodes_per_panel": 88, "test_episodes_per_panel": 152,
                    "total_planned_episodes": 3744,
                    "sources_sha256": {name: sha(source_dir/name) for name in (*REQUIRED_SOURCES, "run_melting_hierarchical_batch.py")}}
        write(root/"protocol.json", protocol)
        jobs = []
        for phase in ("calibration", "test"):
            for seed in protocol[f"{phase}_seeds"]:
                args = runner.parser().parse_args(["--phase", phase, "--seed", str(seed),
                       "--output", str(root/phase/f"seed_{seed}"), "--calibration-dir", str(root/"calibration")])
                runner.run_panel(args, CheckRollout())
                jobs.append({"phase": phase, "seed": seed, "returncode": 0})
        write(root/"status.json", {"status": "complete", "jobs": jobs})
        report, hashes = analyze(root, source_dir, author_root)
        require(report["status"] == "complete", f"full schema self-check failed: {report['validation_errors']}")
        require(len(report["method_seed_results"]) == 340 and len(report["primary_seed_differences"]) == 40,
                "full schema aggregate row counts")
        require(report["accounting"]["completed_real_episodes"] == 3744 and len(report["response_learning_seed_results"]) == 400,
                "full schema accounting and progress rows")
        save_outputs(root/"outputs", report, hashes)
        # A single changed raw return must reject the whole batch even when all
        # summary files still claim completion. No subset effect is emitted.
        path = next((root/"test"/"seed_32000"/"episodes").glob("*.json"))
        while path.name.endswith(".events.json"):
            path = next(p for p in (root/"test"/"seed_32000"/"episodes").glob("*.json") if not p.name.endswith(".events.json"))
        changed = read(path)
        changed["focal_return"] += 100
        write(path, changed)
        invalid, hashes = analyze(root, source_dir, author_root)
        require(invalid["status"] == "partial" and not invalid["formal_effects_available"] and "methods" not in invalid,
                "raw-data corruption must reject all formal effects")
        save_outputs(root/"outputs", invalid, hashes)
        require(len((root/"outputs"/"method_seed_results.csv").read_text().splitlines()) == 1,
                "partial reanalysis must overwrite stale effect CSV")
    print("PASS: exact paired tests through 20 pairs, Holm, seed-cluster bootstrap, weighted expectations; full temporary 8-calibration/20-test v2 schema, 3744-episode accounting, seals, raw-return recomputation, and corruption/partial effect gating")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--author-root", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.batch_root is None or args.output is None:
        p.error("--batch-root and --output are required")
    author_root = args.author_root or args.source_dir.parent/"colin_pivot"
    report, hashes = analyze(args.batch_root, args.source_dir, author_root)
    save_outputs(args.output, report, hashes)
    print(json.dumps({"status": report["status"], "formal_effects_available": report["formal_effects_available"],
                      "validation_errors": report["validation_errors"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
