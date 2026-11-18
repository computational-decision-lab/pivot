#!/usr/bin/env python3
"""Run the preregistered equal-budget, no-sentinel loop variant (v3c).

All three selectors receive two sequential paired HF observations.  They use
the same contextual correction prior, candidate generator, response world,
terminal rule, and random streams; only the second-query acquisition rule
differs.  This variant is kept separate from v3b because removing the common
sentinel changes the information protocol and therefore the estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_multiround_loop import ResponseWorld, deployment_delta, make_row, propose, sha256_bytes, stable_seed
from run_multiround_loop_v3b import (
    RESPONSE_REGIMES,
    calibration_cv,
    fit_contextual_model,
    make_calibration_samples,
    select_query,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


METHODS = ("PIVOT-KG-Loop-v3c", "IVR-Online-v3c", "Top-Proxy-Online-v3c")
BASE_METHOD = {
    "PIVOT-KG-Loop-v3c": "PIVOT-KG-Loop-v3b",
    "IVR-Online-v3c": "IVR-Online",
    "Top-Proxy-Online-v3c": "Top-Proxy-Online",
}


def run(
    *,
    source: Path,
    output: Path,
    split: str,
    roots_per_regime: int,
    calibration_roots: int,
    rounds: int,
    bootstrap_draws: int,
    ridge: float,
    cross_round_update: str,
) -> None:
    if split not in {"development", "confirmation"}:
        raise ValueError("split must be development or confirmation")
    if cross_round_update not in {"none", "online"}:
        raise ValueError("cross_round_update must be none or online")
    if split == "confirmation" and (
        roots_per_regime != 120
        or calibration_roots != 240
        or rounds != 5
        or not np.isclose(ridge, 0.5)
        or cross_round_update != "none"
    ):
        raise ValueError(
            "confirmation is locked to 120 roots/regime, 240 calibration roots, "
            "5 rounds, ridge=0.5, and cross_round_update=none"
        )
    if output.exists():
        raise SystemExit(f"output exists; choose a new directory: {output}")

    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import CalibrationModel

    samples, calibration_records, _ = make_calibration_samples(
        source, calibration_roots=calibration_roots, rounds=rounds,
    )
    cv = calibration_cv(samples, folds=5, ridge=ridge)
    model = fit_contextual_model(
        samples,
        ridge=ridge,
        observation_variance=np.asarray(cv["observation_variance"], dtype=float),
        residual_variance=float(cv["model_residual_variance"]),
    )
    model_record = model.to_record()
    contextual_hash = sha256_bytes(json.dumps(model_record, sort_keys=True).encode())
    root_start = 860000 if split == "development" else 960000
    trajectories: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []

    for regime_index, (regime_name, response_strength) in enumerate(RESPONSE_REGIMES.items()):
        for root_offset in range(roots_per_regime):
            root = root_start + regime_index * 10000 + root_offset
            initial_pressure = 0.10 + 0.015 * (root_offset % 7)
            for method in METHODS:
                world = ResponseWorld(pressure=initial_pressure, response_strength=response_strength)
                incumbent = {"intensity": 0.18, "bias": 0.0}
                online_model = model.clone()
                cumulative_isr = 0.0
                cumulative_utility = 0.0
                cumulative_deployment_return = 0.0
                harmful = 0
                retained = 0
                for round_id in range(rounds):
                    pressure_before = float(world.pressure)
                    candidates = propose(incumbent, root=root, round_id=round_id)
                    rows = [
                        make_row(world, incumbent, candidate, root=root, round_id=round_id, index=i)
                        for i, candidate in enumerate(candidates)
                    ]
                    current_model = online_model if cross_round_update == "online" else model.clone()
                    posterior = current_model.predict(
                        rows,
                        pressure=pressure_before,
                        response_strength=response_strength,
                        incumbent=incumbent,
                        round_id=round_id,
                        rounds=rounds,
                        calibration_model_type=CalibrationModel,
                    )
                    query_details: list[dict[str, Any]] = []
                    queried: list[tuple[str, float]] = []
                    for query_order in range(2):
                        base_method = BASE_METHOD[method]
                        cid, acquisition = select_query(
                            base_method, posterior, rows, {item[0] for item in queried}
                        )
                        index = posterior._index[cid]
                        query_seed = stable_seed("loop-v3-hf", root, round_id, cid)
                        value = deployment_delta(
                            world, incumbent, rows[index]["candidate_parameters"], seed=query_seed
                        )
                        correction = value - float(rows[index]["delta_proxy"])
                        if cross_round_update == "online":
                            update = current_model.observe(
                                rows[index],
                                correction,
                                pressure=pressure_before,
                                response_strength=response_strength,
                                incumbent=incumbent,
                                round_id=round_id,
                                rounds=rounds,
                            )
                        else:
                            update = {
                                "correction": float(correction),
                                "predicted_correction": None,
                                "residual": None,
                                "standardized_residual": None,
                                "online_update": False,
                            }
                        condition_covariance_index = posterior._index[cid]
                        from run_multiround_loop_v3b import condition_noisy_observation

                        condition_noisy_observation(posterior, cid, value)
                        queried.append((cid, value))
                        query_details.append({
                            "candidate_id": cid,
                            "candidate_index": index,
                            "query_order": query_order,
                            "kind": "method_specific",
                            "acquisition": acquisition,
                            "query_seed": query_seed,
                            "observation": float(value),
                            "posterior_latent_variance": float(posterior.covariance[condition_covariance_index, condition_covariance_index]),
                            "observation_variance": float(posterior.observation_variance[condition_covariance_index]),
                            **update,
                        })

                    selected = posterior.select()
                    eval_seed = stable_seed("loop-v3-independent-eval", root, round_id)
                    eval_deltas = [
                        deployment_delta(world, incumbent, candidate, seed=eval_seed + i)
                        for i, candidate in enumerate(candidates)
                    ]
                    best = max([0.0, *eval_deltas])
                    selected_delta = 0.0 if selected == "incumbent" else eval_deltas[int(selected[1:])]
                    isr = float(best - selected_delta)
                    retained_flag = selected == "incumbent"
                    harmful_flag = selected_delta < 0.0
                    chosen_policy = incumbent if retained_flag else candidates[int(selected[1:])]
                    state_before = world.state_hash()
                    deploy_seed = stable_seed("loop-v3-deployment-response", root, round_id)
                    deployment_return = world.value(chosen_policy, seed=deploy_seed, deployment=True)
                    world.deploy(chosen_policy, seed=deploy_seed)
                    state_after = world.state_hash()
                    ledger.append({
                        "split": split,
                        "regime": regime_name,
                        "response_strength": response_strength,
                        "root_id": root,
                        "method": method,
                        "round_id": round_id,
                        "pressure_before": pressure_before,
                        "pressure_after": float(world.pressure),
                        "candidate_panel": rows,
                        "proposer_seed": stable_seed("proposer", root, round_id),
                        "evaluation_seed": eval_seed,
                        "deployment_seed": deploy_seed,
                        "calibration_hash": contextual_hash,
                        "sentinel_rule": None,
                        "query_ids": [cid for cid, _ in queried],
                        "query_count": len(queried),
                        "query_details": query_details,
                        "selected_policy": selected,
                        "terminal_selection_rule": "posterior mean over incumbent plus four candidates",
                        "retained_incumbent": retained_flag,
                        "deployment_outcome": {
                            "independent_deltas": eval_deltas,
                            "selected_delta": float(selected_delta),
                            "best_delta": float(best),
                            "ISR": isr,
                            "deployment_return": float(deployment_return),
                        },
                        "initial_state_hash": state_before,
                        "next_state_hash": state_after,
                        "next_pressure": float(world.pressure),
                    })
                    cumulative_isr += isr
                    cumulative_utility += float(selected_delta)
                    cumulative_deployment_return += float(deployment_return)
                    harmful += int(harmful_flag)
                    retained += int(retained_flag)
                    incumbent = {
                        "intensity": float(chosen_policy["intensity"]),
                        "bias": float(chosen_policy["bias"]),
                    }
                trajectories.append({
                    "split": split,
                    "regime": regime_name,
                    "root_id": root,
                    "method": method,
                    "CISR": cumulative_isr,
                    "cumulative_utility": cumulative_utility,
                    "cumulative_deployment_return": cumulative_deployment_return,
                    "harmful_adoption_rate": harmful / rounds,
                    "retention_rate": retained / rounds,
                    "rounds": rounds,
                })

    output.mkdir(parents=True)
    ledger_path = output / "trajectory_ledger.json"
    ledger_blob = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    ledger_path.write_bytes(ledger_blob)
    (output / "trajectory_ledger.sha256").write_text(sha256_bytes(ledger_blob) + "\n")
    (output / "trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n")
    (output / "calibration_model.json").write_text(json.dumps(model_record, indent=2, sort_keys=True) + "\n")
    (output / "calibration_rows.json").write_text(json.dumps(calibration_records, indent=2, sort_keys=True) + "\n")
    (output / "calibration_training_samples.json").write_text(json.dumps(samples, indent=2, sort_keys=True) + "\n")
    protocol = {
        "protocol_id": "P1-LOOP-V3C-2026-11-10",
        "split": split,
        "roots_per_regime": roots_per_regime,
        "root_start": root_start,
        "calibration_root_start": 510000,
        "rounds": rounds,
        "candidate_count": 4,
        "calibration_roots": calibration_roots,
        "calibration_repeats_per_candidate": 8,
        "hf_budget_per_round": 2,
        "sentinel_rule": None,
        "query_protocol": "two sequential method-specific queries; no common sentinel",
        "response_regimes": RESPONSE_REGIMES,
        "response_regime_order": list(RESPONSE_REGIMES),
        "initial_test_pressure": "0.10 + 0.015 * (root_offset mod 7)",
        "shared_random_streams": {
            "evaluation": "root, round, candidate index",
            "deployment": "root, round",
            "queries": "root, round, candidate id",
        },
        "terminal_rule": "posterior mean over incumbent plus four candidates",
        "acquisition": {
            "PIVOT-KG-Loop-v3c": "exact one-step Gaussian KG, sequentially updated posterior",
            "IVR-Online-v3c": "integrated variance reduction, sequentially updated posterior",
            "Top-Proxy-Online-v3c": "descending proxy delta, sequentially updated posterior",
        },
        "method_definitions": {
            "PIVOT-KG-Loop-v3c": "two sequential KG queries under contextual calibration",
            "IVR-Online-v3c": "two sequential IVR queries under the same contextual calibration",
            "Top-Proxy-Online-v3c": "two sequential proxy-ranked queries under the same contextual calibration",
        },
        "contextual_model": {
            "type": "Bayesian linear correction model",
            "ridge": ridge,
            "within_round_update": "noisy Gaussian conditioning with positive observation variance; coefficient update kept separate",
            "cross_round_update": cross_round_update,
            "observation_variance": "mean within-context variance over eight independent calibration rollouts, by response strength and candidate slot",
            "model_residual_variance": "five-fold held-out MSE minus variance of the calibration mean, floored at 1e-6",
        },
        "physical_simulator_called": False,
        "primary_estimand": "paired five-round cumulative ISR and absolute deployment return under the declared controlled world",
    }
    protocol_path = Path(__file__).with_name("P1_LOOP_V3C_PROTOCOL_2026-11-10.md")
    checks = {
        "complete_method_root_round_panel": len(ledger) == roots_per_regime * len(RESPONSE_REGIMES) * len(METHODS) * rounds,
        "two_queries_each_round": all(row["query_count"] == 2 for row in ledger),
        "unique_query_ids_within_round": all(len(row["query_ids"]) == len(set(row["query_ids"])) for row in ledger),
        "no_sentinel_queries": all(all(detail["kind"] == "method_specific" for detail in row["query_details"]) for row in ledger),
        "common_deployment_seed_across_methods": len({(row["root_id"], row["round_id"], row["deployment_seed"]) for row in ledger}) == roots_per_regime * len(RESPONSE_REGIMES) * rounds,
        "terminal_rule_includes_incumbent": all("incumbent" in row["terminal_selection_rule"] for row in ledger),
        "state_transitions_recorded": all(row["initial_state_hash"] != row["next_state_hash"] for row in ledger),
        "deployment_returns_recorded": all("deployment_return" in row["deployment_outcome"] for row in ledger),
    }
    summary: dict[str, Any] = {
        "analysis_kind": "equal_budget_no_sentinel_multiround_selector_comparison",
        "protocol": protocol,
        "protocol_sha256": sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "analyzer": sha256_file(Path(__file__).with_name("analyze_multiround_loop_v3c.py")),
            "preregistration": sha256_file(protocol_path),
            "v1_runner": sha256_file(Path(__file__).with_name("run_multiround_loop.py")),
            "v3b_runner": sha256_file(Path(__file__).with_name("run_multiround_loop_v3b.py")),
            "calibration_source": sha256_file(source / "experiments/revision/highway_calibration.py"),
            "kg_source": sha256_file(source / "experiments/revision/highway_b4_redesign.py"),
        },
        "calibration": {
            "contextual_model_hash": contextual_hash,
            "five_fold_cv": cv,
        },
        "methods": list(METHODS),
        "method_summary": {},
        "audit_checks": checks,
        "ledger_sha256": sha256_bytes(ledger_blob),
    }
    from run_multiround_loop import boot

    for regime in RESPONSE_REGIMES:
        summary["method_summary"][regime] = {}
        regime_roots = sorted({int(row["root_id"]) for row in trajectories if row["regime"] == regime})
        for method in METHODS:
            subset = [row for row in trajectories if row["regime"] == regime and row["method"] == method]
            summary["method_summary"][regime][method] = {
                metric: boot(
                    [float(row[metric]) for row in subset],
                    seed=stable_seed("loop-v3c-summary", split, regime, method, metric),
                    draws=bootstrap_draws,
                )
                for metric in (
                    "CISR",
                    "cumulative_utility",
                    "cumulative_deployment_return",
                    "harmful_adoption_rate",
                    "retention_rate",
                )
            }
        summary["method_summary"][regime]["n_roots"] = len(regime_roots)
    summary["audit_checks"]["all_pass"] = all(checks.values())
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--roots-per-regime", type=int, default=50)
    parser.add_argument("--calibration-roots", type=int, default=240)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=0.5)
    parser.add_argument("--cross-round-update", choices=("none", "online"), default="none")
    args = parser.parse_args()
    run(
        source=args.source.resolve(),
        output=args.output.resolve(),
        split=args.split,
        roots_per_regime=args.roots_per_regime,
        calibration_roots=args.calibration_roots,
        rounds=args.rounds,
        bootstrap_draws=args.bootstrap_draws,
        ridge=args.ridge,
        cross_round_update=args.cross_round_update,
    )
    print((args.output / "summary.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
