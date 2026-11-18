#!/usr/bin/env python3
"""Independent audit and paired analysis for loop-v3c outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        raise ValueError("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.percentile(sample, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(low), float(high)], "n_roots": int(len(x))}


def signflip_p(values: list[float], *, seed: int, draws: int) -> float:
    x = np.asarray(values, dtype=float)
    if np.all(np.abs(x) <= 1e-15):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(x)))
    observed = abs(float(x.mean()))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def replay_trajectories(
    input_dir: Path,
    source: Path,
    summary: dict[str, Any],
    trajectories: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    import sys

    audit_dir = Path(__file__).resolve().parent
    sys.path[:0] = [str(audit_dir), str(source), str(source / "src")]
    from run_multiround_loop import ResponseWorld, deployment_delta, make_row, propose, stable_seed
    from run_multiround_loop_v3b import (
        calibration_cv,
        condition_noisy_observation,
        fit_contextual_model,
        select_query,
    )
    from experiments.revision.highway_calibration import CalibrationModel

    protocol = summary["protocol"]
    samples = json.loads((input_dir / "calibration_training_samples.json").read_text())
    cv = calibration_cv(samples, folds=5, ridge=float(protocol["contextual_model"]["ridge"]))
    fitted = fit_contextual_model(
        samples,
        ridge=float(protocol["contextual_model"]["ridge"]),
        observation_variance=np.asarray(cv["observation_variance"], dtype=float),
        residual_variance=float(cv["model_residual_variance"]),
    )
    model_record = fitted.to_record()
    stored_model = json.loads((input_dir / "calibration_model.json").read_text())
    model_record_match = json.dumps(model_record, sort_keys=True) == json.dumps(stored_model, sort_keys=True)
    model_hash = hashlib.sha256(json.dumps(model_record, sort_keys=True).encode()).hexdigest()
    model_hash_match = model_hash == summary["calibration"]["contextual_model_hash"]

    methods = list(summary["methods"])
    method_alias = {
        "PIVOT-KG-Loop-v3c": "PIVOT-KG-Loop-v3b",
        "IVR-Online-v3c": "IVR-Online",
        "Top-Proxy-Online-v3c": "Top-Proxy-Online",
    }
    ledger_index = {
        (row["regime"], row["method"], int(row["root_id"]), int(row["round_id"])): row
        for row in ledger
    }
    trajectory_index = {
        (row["regime"], row["method"], int(row["root_id"])): row
        for row in trajectories
    }
    replay_ok = True
    replay_counts = {"candidate_panels": 0, "queries": 0, "posterior_selections": 0, "independent_evaluations": 0, "deployment_returns": 0, "state_transitions": 0, "trajectory_aggregates": 0}
    failures: list[dict[str, Any]] = []

    def check(condition: bool, *, kind: str, regime: str, root: int, method: str, round_id: int | None = None) -> None:
        nonlocal replay_ok
        replay_ok = replay_ok and bool(condition)
        if not condition and len(failures) < 25:
            failures.append({"kind": kind, "regime": regime, "root_id": root, "method": method, "round_id": round_id})

    response_order = list(protocol.get("response_regime_order", protocol["response_regimes"]))
    response_values = protocol["response_regimes"]
    for regime_index, regime in enumerate(response_order):
        response_strength = response_values[regime]
        roots = sorted({int(row["root_id"]) for row in trajectories if row["regime"] == regime})
        for root in roots:
            root_offset = root - int(protocol["root_start"]) - regime_index * 10000
            initial_pressure = 0.10 + 0.015 * (root_offset % 7)
            for method in methods:
                world = ResponseWorld(pressure=initial_pressure, response_strength=float(response_strength))
                incumbent = {"intensity": 0.18, "bias": 0.0}
                online_model = fitted.clone()
                totals = {"CISR": 0.0, "cumulative_utility": 0.0, "cumulative_deployment_return": 0.0, "harmful": 0, "retained": 0}
                for round_id in range(int(protocol["rounds"])):
                    row = ledger_index[(regime, method, root, round_id)]
                    pressure_before = float(world.pressure)
                    candidates = propose(incumbent, root=root, round_id=round_id)
                    rows = [make_row(world, incumbent, candidate, root=root, round_id=round_id, index=i) for i, candidate in enumerate(candidates)]
                    check(rows == row["candidate_panel"], kind="candidate_panel", regime=regime, root=root, method=method, round_id=round_id)
                    replay_counts["candidate_panels"] += 1
                    current_model = online_model if protocol["contextual_model"]["cross_round_update"] == "online" else fitted.clone()
                    posterior = current_model.predict(
                        rows,
                        pressure=pressure_before,
                        response_strength=float(response_strength),
                        incumbent=incumbent,
                        round_id=round_id,
                        rounds=int(protocol["rounds"]),
                        calibration_model_type=CalibrationModel,
                    )
                    expected_query_ids: list[str] = []
                    for query_order, detail in enumerate(row["query_details"]):
                        candidate_id, acquisition = select_query(
                            method_alias[method], posterior, rows, set(expected_query_ids)
                        )
                        check(candidate_id == detail["candidate_id"], kind="query_choice", regime=regime, root=root, method=method, round_id=round_id)
                        if acquisition is None:
                            check(detail.get("acquisition") is None, kind="query_score", regime=regime, root=root, method=method, round_id=round_id)
                        else:
                            check(np.isclose(float(detail["acquisition"]), float(acquisition), rtol=1e-10, atol=1e-12), kind="query_score", regime=regime, root=root, method=method, round_id=round_id)
                        candidate_index = posterior._index[candidate_id]
                        query_seed = stable_seed("loop-v3-hf", root, round_id, candidate_id)
                        value = deployment_delta(world, incumbent, rows[candidate_index]["candidate_parameters"], seed=query_seed)
                        check(int(detail["query_order"]) == query_order and int(detail["query_seed"]) == query_seed, kind="query_seed_or_order", regime=regime, root=root, method=method, round_id=round_id)
                        check(np.isclose(float(detail["observation"]), float(value), rtol=1e-10, atol=1e-12), kind="query_observation", regime=regime, root=root, method=method, round_id=round_id)
                        if protocol["contextual_model"]["cross_round_update"] == "online":
                            correction = value - float(rows[candidate_index]["delta_proxy"])
                            current_model.observe(
                                rows[candidate_index], correction, pressure=pressure_before,
                                response_strength=float(response_strength), incumbent=incumbent,
                                round_id=round_id, rounds=int(protocol["rounds"]),
                            )
                        condition_noisy_observation(posterior, candidate_id, value)
                        latent_variance = float(posterior.covariance[candidate_index, candidate_index])
                        check(np.isclose(float(detail["posterior_latent_variance"]), latent_variance, rtol=1e-9, atol=1e-11), kind="posterior_variance", regime=regime, root=root, method=method, round_id=round_id)
                        check(np.isclose(float(detail["observation_variance"]), float(posterior.observation_variance[candidate_index]), rtol=1e-10, atol=1e-12), kind="observation_variance", regime=regime, root=root, method=method, round_id=round_id)
                        expected_query_ids.append(candidate_id)
                        replay_counts["queries"] += 1
                    selected = posterior.select()
                    check(selected == row["selected_policy"], kind="posterior_selection", regime=regime, root=root, method=method, round_id=round_id)
                    replay_counts["posterior_selections"] += 1
                    evaluation_seed = stable_seed("loop-v3-independent-eval", root, round_id)
                    eval_deltas = [deployment_delta(world, incumbent, candidate, seed=evaluation_seed + i) for i, candidate in enumerate(candidates)]
                    check(int(row["evaluation_seed"]) == evaluation_seed and np.allclose(eval_deltas, row["deployment_outcome"]["independent_deltas"], rtol=1e-10, atol=1e-12), kind="independent_evaluation", regime=regime, root=root, method=method, round_id=round_id)
                    replay_counts["independent_evaluations"] += 1
                    best = max([0.0, *eval_deltas])
                    selected_delta = 0.0 if selected == "incumbent" else eval_deltas[int(selected[1:])]
                    isr = best - selected_delta
                    chosen = incumbent if selected == "incumbent" else candidates[int(selected[1:])]
                    state_before = world.state_hash()
                    deployment_seed = stable_seed("loop-v3-deployment-response", root, round_id)
                    deployment_return = world.value(chosen, seed=deployment_seed, deployment=True)
                    world.deploy(chosen, seed=deployment_seed)
                    state_after = world.state_hash()
                    check(int(row["deployment_seed"]) == deployment_seed and np.isclose(float(row["deployment_outcome"]["deployment_return"]), deployment_return, rtol=1e-10, atol=1e-12), kind="deployment_return", regime=regime, root=root, method=method, round_id=round_id)
                    check(row["initial_state_hash"] == state_before and row["next_state_hash"] == state_after and np.isclose(float(row["next_pressure"]), world.pressure, rtol=1e-10, atol=1e-12), kind="state_transition", regime=regime, root=root, method=method, round_id=round_id)
                    replay_counts["deployment_returns"] += 1
                    replay_counts["state_transitions"] += 1
                    totals["CISR"] += isr
                    totals["cumulative_utility"] += selected_delta
                    totals["cumulative_deployment_return"] += deployment_return
                    totals["harmful"] += int(selected_delta < 0.0)
                    totals["retained"] += int(selected == "incumbent")
                    incumbent = {"intensity": float(chosen["intensity"]), "bias": float(chosen["bias"])}
                traj = trajectory_index[(regime, method, root)]
                aggregate_matches = (
                    np.isclose(float(traj["CISR"]), totals["CISR"], rtol=1e-10, atol=1e-10)
                    and np.isclose(float(traj["cumulative_utility"]), totals["cumulative_utility"], rtol=1e-10, atol=1e-10)
                    and np.isclose(float(traj["cumulative_deployment_return"]), totals["cumulative_deployment_return"], rtol=1e-10, atol=1e-10)
                    and np.isclose(float(traj["harmful_adoption_rate"]), totals["harmful"] / int(protocol["rounds"]), rtol=1e-10, atol=1e-10)
                    and np.isclose(float(traj["retention_rate"]), totals["retained"] / int(protocol["rounds"]), rtol=1e-10, atol=1e-10)
                )
                check(bool(aggregate_matches), kind="trajectory_aggregate", regime=regime, root=root, method=method)
                replay_counts["trajectory_aggregates"] += 1
    return {
        "calibration_model_recomputed": bool(model_record_match),
        "calibration_model_hash_matches": bool(model_hash_match),
        "trajectory_replay_all_pass": bool(replay_ok),
        "replay_counts": replay_counts,
        "failure_count": len(failures),
        "first_failures": failures,
    }


def analyze(input_dir: Path, source: Path, output: Path, *, bootstrap_draws: int, signflip_draws: int) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"output exists; choose a new file: {output}")
    summary = json.loads((input_dir / "summary.json").read_text())
    trajectories = json.loads((input_dir / "trajectories.json").read_text())
    ledger = json.loads((input_dir / "trajectory_ledger.json").read_text())
    samples = json.loads((input_dir / "calibration_training_samples.json").read_text())
    model = json.loads((input_dir / "calibration_model.json").read_text())
    protocol = summary["protocol"]
    methods = list(summary["methods"])
    primary, *baselines = methods
    regimes = list(protocol.get("response_regime_order", protocol["response_regimes"]))
    rounds = int(protocol["rounds"])
    index = {(row["regime"], row["method"], int(row["root_id"])): row for row in trajectories}
    ledger_index = {
        (row["regime"], row["method"], int(row["root_id"]), int(row["round_id"])): row
        for row in ledger
    }
    roots_by_regime = {
        regime: sorted({int(row["root_id"]) for row in trajectories if row["regime"] == regime})
        for regime in regimes
    }

    calibration_root_ids = sorted({int(row["root_id"]) for row in samples})
    expected_calibration = list(range(int(protocol["calibration_root_start"]), int(protocol["calibration_root_start"]) + int(protocol["calibration_roots"])))
    noise_regimes = [float(value) for value in model["observation_variance_regimes"]]
    expected_noise = np.zeros((len(noise_regimes), int(protocol["candidate_count"])))
    noise_counts = np.zeros_like(expected_noise, dtype=int)
    calibration_panel: dict[int, set[int]] = {}
    calibration_noise_ok = True
    repeats = int(protocol["calibration_repeats_per_candidate"])
    for row in samples:
        values = np.asarray(row["rollout_values"], dtype=float)
        expected_seeds = [
            stable_seed("loop-v3-calibration-hf", row["root_id"], row["candidate_index"], replicate)
            for replicate in range(repeats)
        ]
        calibration_noise_ok = calibration_noise_ok and (
            row["rollout_seeds"] == expected_seeds
            and len(values) == repeats
            and np.isclose(float(row["hf_delta"]), float(values.mean()), atol=1e-12)
            and np.isclose(float(row["single_rollout_variance"]), float(values.var(ddof=1)), atol=1e-12)
            and np.isclose(float(row["training_noise_variance"]), float(values.var(ddof=1) / repeats), atol=1e-12)
        )
        ri = min(range(len(noise_regimes)), key=lambda i: abs(noise_regimes[i] - float(row["response_strength"])))
        ci = int(row["candidate_index"])
        expected_noise[ri, ci] += float(row["single_rollout_variance"])
        noise_counts[ri, ci] += 1
        calibration_panel.setdefault(int(row["root_id"]), set()).add(ci)
    expected_noise /= np.maximum(noise_counts, 1)
    recorded_noise = np.asarray(model["single_rollout_observation_variance"], dtype=float)
    test_root_ranges = [
        roots_by_regime[regime] == list(range(int(protocol["root_start"]) + i * 10000, int(protocol["root_start"]) + i * 10000 + int(protocol["roots_per_regime"])))
        for i, regime in enumerate(regimes)
    ]
    expected_trajectory_count = len(regimes) * int(protocol["roots_per_regime"]) * len(methods)
    expected_ledger_count = expected_trajectory_count * rounds
    all_test_roots = [root for roots in roots_by_regime.values() for root in roots]
    protocol_checks: dict[str, bool] = {
        "calibration_noise_rollouts_replay": bool(calibration_noise_ok),
        "calibration_noise_matrix_recomputed": bool(np.allclose(expected_noise, recorded_noise, rtol=1e-10, atol=1e-12)),
        "calibration_samples_complete": len(samples) == int(protocol["calibration_roots"]) * int(protocol["candidate_count"]),
        "calibration_root_candidate_panels_complete": len(calibration_panel) == int(protocol["calibration_roots"]) and all(values == set(range(int(protocol["candidate_count"]))) for values in calibration_panel.values()),
        "calibration_roots_match_declared_block": calibration_root_ids == expected_calibration,
        "test_roots_match_declared_blocks": all(test_root_ranges),
        "test_roots_disjoint_from_calibration": not set(all_test_roots).intersection(calibration_root_ids),
        "unique_trajectory_panel": len(index) == len(trajectories) == expected_trajectory_count,
        "unique_ledger_panel": len(ledger_index) == len(ledger) == expected_ledger_count,
        "two_queries_each_round": all(row["query_count"] == int(protocol["hf_budget_per_round"]) for row in ledger),
        "unique_query_ids_within_round": all(len(row["query_ids"]) == len(set(row["query_ids"])) for row in ledger),
        "no_sentinel_queries": all(detail["kind"] == "method_specific" for row in ledger for detail in row["query_details"]),
        "deployment_returns_recorded": all("deployment_return" in row["deployment_outcome"] for row in ledger),
        "terminal_rule_includes_incumbent": all("incumbent" in row["terminal_selection_rule"] for row in ledger),
        "state_transitions_recorded": all(row["initial_state_hash"] != row["next_state_hash"] for row in ledger),
        "deployment_seed_replay_and_sharing": all(
            row["deployment_seed"] == stable_seed("loop-v3-deployment-response", row["root_id"], row["round_id"])
            and len({ledger_index[(row["regime"], method, int(row["root_id"]), int(row["round_id"]))]["deployment_seed"] for method in methods}) == 1
            for row in ledger
        ),
        "evaluation_seed_replay_and_sharing": all(
            row["evaluation_seed"] == stable_seed("loop-v3-independent-eval", row["root_id"], row["round_id"])
            and len({ledger_index[(row["regime"], method, int(row["root_id"]), int(row["round_id"]))]["evaluation_seed"] for method in methods}) == 1
            for row in ledger
        ),
        "query_seed_replay": all(detail["query_seed"] == stable_seed("loop-v3-hf", row["root_id"], row["round_id"], detail["candidate_id"]) for row in ledger for detail in row["query_details"]),
        "terminal_isr_recomputed": all(
            np.isclose(float(row["deployment_outcome"]["ISR"]), max([0.0, *map(float, row["deployment_outcome"]["independent_deltas"])]) - float(row["deployment_outcome"]["selected_delta"]), atol=1e-12)
            for row in ledger
        ),
        "noisy_posterior_conditioning_retains_latent_variance": all(
            float(detail["observation_variance"]) > 0.0 and float(detail["posterior_latent_variance"]) > 0.0
            for row in ledger for detail in row["query_details"]
        ),
    }
    comparisons: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    method_metrics: dict[str, Any] = {}
    metric_names = ("CISR", "cumulative_utility", "cumulative_deployment_return", "harmful_adoption_rate", "retention_rate")
    for regime in regimes:
        roots = roots_by_regime[regime]
        method_metrics[regime] = {}
        for method in methods:
            rows = [index[(regime, method, root)] for root in roots]
            method_metrics[regime][method] = {
                metric: bootstrap([float(row[metric]) for row in rows], seed=stable_seed("loop-v3c-method", protocol["split"], regime, method, metric), draws=bootstrap_draws)
                for metric in metric_names
            }
        for baseline in baselines:
            isr_diffs = [float(index[(regime, baseline, root)]["CISR"]) - float(index[(regime, primary, root)]["CISR"]) for root in roots]
            label = f"{regime}:{baseline}_minus_{primary}_CISR"
            pvalues[label] = signflip_p(isr_diffs, seed=stable_seed("loop-v3c-signflip", protocol["split"], regime, baseline), draws=signflip_draws)
            return_diffs = [float(index[(regime, primary, root)]["cumulative_deployment_return"]) - float(index[(regime, baseline, root)]["cumulative_deployment_return"]) for root in roots]
            comparisons[label] = {
                "estimand": "baseline cumulative ISR minus PIVOT cumulative ISR; positive favors PIVOT",
                "paired_root_bootstrap": bootstrap(isr_diffs, seed=stable_seed("loop-v3c-isr-bootstrap", protocol["split"], regime, baseline), draws=bootstrap_draws),
                "two_sided_signflip_p": pvalues[label],
                "deployment_return_primary_minus_baseline": bootstrap(return_diffs, seed=stable_seed("loop-v3c-return-bootstrap", protocol["split"], regime, baseline), draws=bootstrap_draws),
            }
    adjusted = holm_adjust(pvalues)
    for label, value in adjusted.items():
        comparisons[label]["holm_adjusted_p_over_all_regime_baseline_contrasts"] = value

    isr_superiority_by_regime: dict[str, bool] = {}
    deployment_return_noninferior_by_regime: dict[str, bool] = {}
    for regime in regimes:
        labels = [f"{regime}:{baseline}_minus_{primary}_CISR" for baseline in baselines]
        isr_superiority_by_regime[regime] = all(
            comparisons[label]["paired_root_bootstrap"]["ci95"][0] > 0.0
            and comparisons[label]["holm_adjusted_p_over_all_regime_baseline_contrasts"] < 0.05
            for label in labels
        )
        deployment_return_noninferior_by_regime[regime] = all(
            comparisons[label]["deployment_return_primary_minus_baseline"]["ci95"][0] >= 0.0
            for label in labels
        )
    joint_success_by_regime = {
        regime: isr_superiority_by_regime[regime] and deployment_return_noninferior_by_regime[regime]
        for regime in regimes
    }
    replay = replay_trajectories(input_dir, source, summary, trajectories, ledger)
    protocol_checks["calibration_model_replay"] = replay["calibration_model_recomputed"] and replay["calibration_model_hash_matches"]
    protocol_checks["independent_full_replay"] = replay["trajectory_replay_all_pass"]
    report = {
        "analysis_kind": "paired_root_equal_budget_no_sentinel_v3c",
        "split": protocol["split"],
        "protocol_sha256": summary["protocol_sha256"],
        "source_hashes": {name: sha256_file(input_dir / filename) for name, filename in {
            "summary": "summary.json", "trajectories": "trajectories.json", "ledger": "trajectory_ledger.json",
            "calibration_training_samples": "calibration_training_samples.json", "calibration_model": "calibration_model.json",
        }.items()},
        "n_roots_per_regime": {regime: len(roots) for regime, roots in roots_by_regime.items()},
        "method_metrics": method_metrics,
        "protocol_audit_checks": {**protocol_checks, "all_pass": all(protocol_checks.values())},
        "primary_paired_cumulative_isr_contrasts": comparisons,
        "holm_family": list(pvalues),
        "isr_superiority_by_regime": isr_superiority_by_regime,
        "deployment_return_noninferior_by_regime": deployment_return_noninferior_by_regime,
        "joint_success_by_regime": joint_success_by_regime,
        "overall_success": bool(all(joint_success_by_regime.values())),
        "replay_audit": replay,
        "success_rule": "per regime, both baseline-minus-PIVOT ISR 95% CI lower bounds > 0 and Holm-adjusted sign-flip p-values < 0.05, plus both PIVOT-minus-baseline deployment-return 95% CI lower bounds >= 0",
        "limitations": [
            "The result is conditional on the declared controlled PerformativeWorld, proposer, candidate generator, calibration cohort, budget, and five-round horizon.",
            "CISR is a within-state selection-regret estimate based on one independent paired rollout per candidate per round; it is not itself an absolute deployment-return objective.",
            "Absolute deployment return is reported separately and does not telescope across changing states.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--signflip-draws", type=int, default=100000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.source.resolve(), args.output, bootstrap_draws=args.bootstrap_draws, signflip_draws=args.signflip_draws), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
