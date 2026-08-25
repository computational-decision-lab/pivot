#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.external_adaptive.mpe2_world import (
    MPE2Config,
    MPE2Policy,
    MPE2World,
    generate_mpe2_candidates,
)
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.metrics.improvement import compute_improvement_metrics
from pivot.research.state import classify_experiment
from pivot.theory.sample_complexity import required_subgaussian_samples
from pivot.transfer.differential import DifferentialModel
from pivot.transfer.global_value import GlobalValueModel, spearman_rank_correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 E4b global versus transition fidelity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-count", type=int, default=None)
    parser.add_argument("--held-out-trajectories", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--phase", choices=["development", "validation", "confirmatory"], default=None)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E4b config must be a mapping")
    trajectory_count = int(args.trajectory_count or payload.get("trajectory_count", 12))
    held_out = int(args.held_out_trajectories or payload.get("held_out_trajectories", 4))
    rounds = int(args.rounds or payload.get("rounds", 4))
    phase = str(args.phase or payload.get("phase", "development"))
    if trajectory_count <= held_out or held_out <= 0 or rounds <= 0:
        raise ValueError("E4b requires training and held-out trajectories")
    rows = _generate_rows(payload, trajectory_count, rounds)
    trajectory_ids = sorted({str(row["trajectory_id"]) for row in rows})
    test_trajectory_ids = set(trajectory_ids[-held_out:])
    train_rows = [row for row in rows if str(row["trajectory_id"]) not in test_trajectory_ids]
    test_rows = [row for row in rows if str(row["trajectory_id"]) in test_trajectory_ids]
    budget = min(int(payload.get("hf_budget", 16)), len(train_rows))
    if budget <= 0:
        raise ValueError("E4b HF budget must be positive")
    train_rows = train_rows[:budget]
    metrics = _compare(train_rows, test_rows)
    metrics.update(
        {
            "experiment": "e4b_global_vs_transition",
            "split_contract": "trajectory-disjoint",
            "train_trajectory_count": len(trajectory_ids) - held_out,
            "held_out_trajectory_count": held_out,
            "train_transition_count": len(train_rows),
            "test_transition_count": len(test_rows),
            "hf_budget": budget,
            "paired": True,
        }
    )
    minimum_effect = float(payload.get("minimum_effect_ide", 0.5))
    trajectory_effects = [
        float(item["global_ide"] - item["transition_ide"])
        for item in metrics.pop("trajectory_effects")
    ]
    effect_sigma = float(np.std(trajectory_effects, ddof=1)) if len(trajectory_effects) > 1 else 0.0
    required_trajectories = required_subgaussian_samples(
        sigma=max(effect_sigma, 1e-9),
        margin=minimum_effect,
        candidates=2,
        delta=float(payload.get("alpha", 0.05)),
    )
    metrics.update(
        {
            "trajectory_effects": trajectory_effects,
            "trajectory_effect_sigma": effect_sigma,
            "power_required_held_out_trajectories": required_trajectories,
            "minimum_effect_ide": minimum_effect,
        }
    )
    if phase != "confirmatory":
        state = classify_experiment(
            underpowered=True,
            reason=f"{phase} E4b result is diagnostic; confirmatory holdout is not yet frozen",
        )
    elif len(trajectory_effects) < required_trajectories:
        state = classify_experiment(
            underpowered=True,
            reason=(
                f"held-out trajectory count {len(trajectory_effects)} is below the registered power rule "
                f"({required_trajectories})"
            ),
        )
    else:
        state = classify_experiment(
            hypothesis_supported=bool(
                np.mean(trajectory_effects) >= minimum_effect
            ),
            confirmatory=True,
            reason="transition evaluator has lower held-out transition error by the registered margin"
            if np.mean(trajectory_effects) >= minimum_effect
            else "held-out transition evaluator does not improve over global value evaluator",
        )
    metrics["state"] = state.state.value
    metrics["state_reason"] = state.reason
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "transition_rows.jsonl", test_rows, jsonl=True)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "state.json", {"state": state.state.value, "reason": state.reason})
    _write_json(
        output / "provenance.json",
        {
            "config": payload,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "environment_source": "Farama MPE2",
            "train_trajectory_ids": sorted({str(row["trajectory_id"]) for row in train_rows}),
            "test_trajectory_ids": sorted(test_trajectory_ids),
            "paired": True,
            "phase": phase,
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(json.dumps({"experiment": "e4b_global_vs_transition", "state": state.state.value}, sort_keys=True))


def _generate_rows(payload: dict[str, Any], trajectory_count: int, rounds: int) -> list[dict[str, Any]]:
    config = MPE2Config(
        max_cycles=int(payload.get("max_cycles", 12)),
        scenario=str(payload.get("scenario", "simple_adversary_v3")),
        observation_dim=int(payload.get("observation_dim", 10)),
    )
    world = MPE2World(config)
    seeds = [int(seed) for seed in payload.get("seeds", [])][:trajectory_count]
    count = int(payload.get("candidates_per_round", 4))
    scale = float(payload.get("candidate_scale", 0.15))
    rows: list[dict[str, Any]] = []
    for trajectory_index, seed in enumerate(seeds):
        incumbent = MPE2Policy.random(seed=seed, observation_dim=config.observation_dim, action_dim=5)
        trajectory_id = f"trajectory-{trajectory_index:03d}"
        for round_id in range(rounds):
            incumbent_proxy = world.evaluate_observer(incumbent, seed=seed + round_id).value
            incumbent_true = world.evaluate_actor(incumbent, seed=seed + round_id).value
            candidates = generate_mpe2_candidates(incumbent, count=count, seed=seed + round_id, scale=scale)
            for candidate_index, candidate in enumerate(candidates):
                candidate_proxy = world.evaluate_observer(candidate, seed=seed + round_id).value
                candidate_true = world.evaluate_actor(candidate, seed=seed + round_id).value
                rows.append(
                    {
                        "transition_id": f"{trajectory_id}-{round_id}-{candidate_index}-{candidate.policy_id}",
                        "trajectory_id": trajectory_id,
                        "round_id": round_id,
                        "candidate_index": candidate_index,
                        "seed": seed,
                        "incumbent_policy_id": incumbent.policy_id,
                        "candidate_policy_id": candidate.policy_id,
                        "incumbent_parameters": _policy_mapping(incumbent),
                        "candidate_parameters": _policy_mapping(candidate),
                        "true_incumbent_value": incumbent_true,
                        "true_candidate_value": candidate_true,
                        "delta_proxy": candidate_proxy - incumbent_proxy,
                        "delta_true": candidate_true - incumbent_true,
                        "response_strength": 1.0,
                        "competition_strength": 0.0,
                        "update_footprint": incumbent.distance(candidate),
                        "hf_queried": True,
                        "paired_rollout_ids": [f"{trajectory_id}:{round_id}:{seed}"],
                    }
                )
            incumbent = candidates[0]
    if len(rows) < 4:
        raise ValueError("E4b generated too few transitions")
    return rows


def _compare(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_model = GlobalValueModel()
    global_features = [_policy_summary(row["candidate_parameters"]) for row in train_rows] + [
        _policy_summary(row["incumbent_parameters"]) for row in train_rows
    ]
    global_values = [float(row["true_candidate_value"]) for row in train_rows] + [float(row["true_incumbent_value"]) for row in train_rows]
    global_model.fit(global_features, global_values)
    differential_model = DifferentialModel()
    differential_model.fit(train_rows, [float(row["delta_true"] - row["delta_proxy"]) for row in train_rows])
    global_policy_predictions = [global_model.predict(_policy_summary(row["candidate_parameters"])) for row in test_rows]
    global_policy_truth = [float(row["true_candidate_value"]) for row in test_rows]
    global_errors = [abs(prediction - truth) for prediction, truth in zip(global_policy_predictions, global_policy_truth)]
    global_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for row in test_rows:
        global_delta = global_model.predict(_policy_summary(row["candidate_parameters"])) - global_model.predict(
            _policy_summary(row["incumbent_parameters"])
        )
        differential_prediction = differential_model.predict_correction(row).predicted_delta
        global_rows.append({**row, "delta_proxy": global_delta})
        transition_rows.append({**row, "delta_proxy": differential_prediction})
    _mark_selected(global_rows)
    _mark_selected(transition_rows)
    global_metric = compute_improvement_metrics(global_rows)
    transition_metric = compute_improvement_metrics(transition_rows)
    global_rank = spearman_rank_correlation(global_policy_predictions, global_policy_truth)
    global_transition_ide = _required_float(global_metric["ide"])
    transition_ide = _required_float(transition_metric["ide"])
    ci_low, ci_high = bootstrap_mean_ci(global_errors, seed=20260825)
    trajectory_effects: list[dict[str, float | str]] = []
    for trajectory_id in sorted({str(row["trajectory_id"]) for row in test_rows}):
        global_subset = [row for row in global_rows if str(row["trajectory_id"]) == trajectory_id]
        transition_subset = [row for row in transition_rows if str(row["trajectory_id"]) == trajectory_id]
        global_subset_metric = compute_improvement_metrics(global_subset)
        transition_subset_metric = compute_improvement_metrics(transition_subset)
        trajectory_effects.append(
            {
                "trajectory_id": trajectory_id,
                "global_ide": _required_float(global_subset_metric["ide"]),
                "transition_ide": _required_float(transition_subset_metric["ide"]),
            }
        )
    return {
        "global_value": {
            "policy_value_mae": float(np.mean(global_errors)),
            "policy_value_mae_ci": [ci_low, ci_high],
            "policy_value_spearman": global_rank,
            "transition_ide": global_transition_ide,
            "transition_isc": global_metric["isc"],
            "transition_irr": global_metric["irr"],
            "update_selection_regret": global_metric["isr"],
        },
        "transition": {
            "ide": transition_ide,
            "isc": transition_metric["isc"],
            "irr": transition_metric["irr"],
            "update_selection_regret": transition_metric["isr"],
        },
        "trajectory_effects": trajectory_effects,
    }


def _required_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("expected a numeric transition metric")
    return float(value)


def _mark_selected(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["trajectory_id"]), int(row["round_id"])), []).append(row)
    for group in groups.values():
        selected = max(group, key=lambda row: float(row["delta_proxy"]))
        for row in group:
            row["selected"] = row is selected


def _policy_mapping(policy: MPE2Policy) -> dict[str, float]:
    values = {f"w{index}": float(value) for index, value in enumerate(policy.weights.reshape(-1))}
    values.update({f"b{index}": float(value) for index, value in enumerate(policy.bias)})
    return values


def _policy_summary(parameters: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(parameters.values()), dtype=np.float64)
    return {
        "parameter_mean": float(values.mean()) if len(values) else 0.0,
        "parameter_std": float(values.std()) if len(values) else 0.0,
        "parameter_l2": float(np.linalg.norm(values)),
        "parameter_max_abs": float(np.max(np.abs(values))) if len(values) else 0.0,
    }


def _manifest(directory: Path) -> dict[str, Any]:
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}
    return {"schema_version": "v7-e4b-1", "file_count": len(files), "files": files}


def _write_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    if jsonl:
        path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
