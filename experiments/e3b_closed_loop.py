#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    score_pivot_voi,
    should_stop,
)
from pivot.acquisition.random import select_random
from pivot.acquisition.top_proxy import select_top_proxy
from pivot.environments.external_adaptive.mpe2_world import (
    MPE2Config,
    MPE2Policy,
    MPE2World,
    generate_mpe2_candidates,
)
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.research.state import classify_experiment
from pivot.research.validity import evaluate_e3b_gates
from pivot.theory.sample_complexity import required_cluster_samples, required_subgaussian_samples

FEATURE_DIM = 9


class PosteriorPredictionAdapter:
    def __init__(self, posterior: BayesianLinearDeltaPosterior) -> None:
        self.posterior = posterior

    def predict_correction(self, row: dict[str, Any]) -> Any:
        features = np.asarray([row["features"]], dtype=np.float64)
        correction = float(self.posterior.predict(features)[0])
        standard_deviation = math.sqrt(
            float(self.posterior.predictive_variance(features, include_observation=True)[0])
        )
        predicted_delta = float(row["delta_proxy"]) + correction
        sign_change_probability = 1.0 if predicted_delta < 0.0 else 0.0

        return Prediction(correction, standard_deviation, predicted_delta, sign_change_probability)


@dataclass(frozen=True)
class Prediction:
    correction: float
    standard_deviation: float
    predicted_delta: float
    sign_change_probability: float


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 E3b external closed-loop self-improvement")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "validation", "confirmatory"], default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E3b config must be a mapping")
    phase = str(args.phase or payload.get("phase", "development"))
    rounds = int(args.rounds or payload.get("rounds", 30))
    max_trajectories = args.max_trajectories
    methods = (
        [item.strip() for item in args.methods.split(",") if item.strip()]
        if args.methods is not None
        else [str(method) for method in payload.get("methods", [])]
    )
    workers = int(args.workers or payload.get("workers", 1))
    if workers <= 0:
        raise ValueError("E3b workers must be positive")
    if rounds <= 0 or not methods:
        raise ValueError("E3b rounds and methods must not be empty")
    environments = list(payload.get("environments", []))
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if phase == "confirmatory" and payload.get("confirmatory_seed_count") is not None:
        start = int(payload.get("confirmatory_seed_start", seeds[0] if seeds else 0))
        count = int(payload["confirmatory_seed_count"])
        if count <= 0:
            raise ValueError("confirmatory_seed_count must be positive")
        seeds = list(range(start, start + count))
    if not environments or not seeds:
        raise ValueError("E3b requires a non-empty environment and seed pool")
    if max_trajectories is not None:
        seeds = seeds[:max_trajectories]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    try:
        tasks: list[tuple[int, str, str, int, dict[str, Any]]] = []
        task_index = 0
        for env_index, env_payload in enumerate(environments):
            env_id = str(env_payload["id"])
            for seed in seeds:
                for method in methods:
                    tasks.append((task_index, env_id, method, seed + env_index * 10000, env_payload))
                    task_index += 1
        worker_count = min(workers, max(1, len(tasks)))
        if worker_count == 1:
            results = [_run_task(task, payload, rounds) for task in tasks]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        _run_task_from_bundle,
                        ((task, payload, rounds) for task in tasks),
                    )
                )
        for _, env_id, method, seed, result in sorted(results, key=lambda item: item[0]):
            trajectory_id = f"{env_id}|seed={seed}|method={method}"
            trajectories.append({"trajectory_id": trajectory_id, **result["summary"]})
            all_rows.extend(
                {
                    "trajectory_id": trajectory_id,
                    "environment_id": env_id,
                    "operator_id": method,
                    "method": method,
                    **row,
                }
                for row in result["rows"]
            )
            validity_rows.extend(result["rows"])
    except (ImportError, RuntimeError, ValueError, TypeError) as error:
        _write_failure(output, payload, phase, error)
        return

    validity = evaluate_e3b_gates(
        rewards=[float(row["selected_true_delta"]) for row in validity_rows if row.get("selected_true_delta") is not None],
        max_possible_reward=float(payload.get("max_cycles", 25) * 10),
        response_differences=[
            float(row["delta_true"] - row["delta_proxy"])
            for row in validity_rows
            if row.get("delta_true") is not None
        ],
        candidate_true_deltas=[
            float(row["delta_true"]) for row in validity_rows if row.get("delta_true") is not None
        ],
        proxy_deltas=[
            float(row["delta_proxy"]) for row in validity_rows if row.get("delta_true") is not None
        ],
        paired_deltas=[
            float(row["delta_true"]) for row in validity_rows if row.get("delta_true") is not None
        ],
    )
    metrics = _aggregate_metrics(trajectories, methods)
    worst_case_power_required = required_subgaussian_samples(
        sigma=max(float(metrics["cti_effect_std"]), 1e-9),
        margin=float(payload.get("minimum_effect_cti", 5.0)),
        candidates=max(2, len(methods)),
        delta=float(payload.get("alpha", 0.05)),
    )
    power_required = required_cluster_samples(
        sigma=max(float(metrics["cti_effect_std"]), 1e-9),
        margin=float(payload.get("minimum_effect_cti", 5.0)),
        alpha=float(payload.get("alpha", 0.05)),
        power=float(payload.get("power", 0.8)),
    )
    metrics.update(
        {
            "experiment": "e3b_closed_loop",
            "phase": phase,
            "uncertainty_unit": "trajectory",
            "validity": asdict(validity),
            "power_required_trajectories": power_required,
            "power_required_paired_trajectories": power_required,
            "worst_case_update_identification_trajectories": worst_case_power_required,
            # Methods are repeated on each base trajectory; only paired
            # method contrasts are independent uncertainty units.
            "observed_method_trajectory_count": len(trajectories),
            "observed_paired_trajectory_count": len(_paired_method_effects(trajectories, "PIVOT-VOI", "Proxy Only")),
            "observed_trajectory_count": len(_paired_method_effects(trajectories, "PIVOT-VOI", "Proxy Only")),
            "methods": methods,
            "hf_budget_per_round": int(payload.get("hf_budget_per_round", 2)),
            "matched_selection_budget": True,
            "workers": worker_count,
        }
    )
    supported = (
        phase == "confirmatory"
        and validity.valid
        and len(_paired_method_effects(trajectories, "PIVOT-VOI", "Proxy Only")) >= power_required
        and float(metrics.get("cti_pivot_voi_minus_proxy", 0.0)) >= float(payload.get("minimum_effect_cti", 5.0))
    )
    if not validity.valid:
        classification = classify_experiment(design_invalid=True, reason="one or more E3b construct gates failed")
    elif phase != "confirmatory":
        classification = classify_experiment(
            underpowered=True,
            reason=f"{phase} E3b result is diagnostic; confirmatory trajectory pool is not frozen",
        )
    elif len(_paired_method_effects(trajectories, "PIVOT-VOI", "Proxy Only")) < power_required:
        classification = classify_experiment(underpowered=True, reason="trajectory count is below the registered power rule")
    else:
        classification = classify_experiment(
            hypothesis_supported=supported,
            confirmatory=True,
            reason=("PIVOT-VOI exceeds Proxy Only on CTI" if supported else "powered E3b test does not support the CTI improvement claim"),
        )
    metrics["state"] = classification.state.value
    metrics["state_reason"] = classification.reason
    _write_json(output / "transition_rows.jsonl", all_rows, jsonl=True)
    _write_json(output / "trajectory_metrics.json", trajectories)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "state.json", {"state": classification.state.value, "reason": classification.reason})
    _write_json(
        output / "provenance.json",
        {
            "config": payload,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "phase": phase,
            "environment_source": "Farama MPE2",
            "paired": True,
            "trajectory_clustered": True,
            "methods": methods,
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(json.dumps({"experiment": "e3b_closed_loop", "trajectories": len(trajectories), "state": classification.state.value}, sort_keys=True))


def _run_task(
    task: tuple[int, str, str, int, dict[str, Any]], payload: dict[str, Any], rounds: int
) -> tuple[int, str, str, int, dict[str, Any]]:
    index, env_id, method, seed, env_payload = task
    config = MPE2Config(
        max_cycles=int(payload.get("max_cycles", 25)),
        scenario=str(env_payload["scenario"]),
        observation_dim=int(env_payload["observation_dim"]),
        environment_version=f"mpe2-1.1.0/{env_payload['scenario']}",
        observer_horizon=int(payload.get("observer_horizon", 3)),
    )
    result = _run_trajectory(
        method=method,
        world=MPE2World(config),
        seed=seed,
        rounds=rounds,
        candidates_per_round=int(payload.get("candidates_per_round", 8)),
        hf_budget=int(payload.get("hf_budget_per_round", 2)),
        candidate_scale=float(payload.get("candidate_scale", 0.15)),
        stop_delta=float(payload.get("stop_delta", 0.05)),
        stop_eta=float(payload.get("stop_eta", 0.01)),
        voi_fantasies=int(payload.get("voi_fantasies", 4)),
        voi_posterior_samples=int(payload.get("voi_posterior_samples", 8)),
        evaluation_cache={},
    )
    return index, env_id, method, seed, result


def _run_task_from_bundle(
    bundle: tuple[tuple[int, str, str, int, dict[str, Any]], dict[str, Any], int]
) -> tuple[int, str, str, int, dict[str, Any]]:
    task, payload, rounds = bundle
    return _run_task(task, payload, rounds)


def _run_trajectory(
    *,
    method: str,
    world: MPE2World,
    seed: int,
    rounds: int,
    candidates_per_round: int,
    hf_budget: int,
    candidate_scale: float,
    stop_delta: float,
    stop_eta: float,
    voi_fantasies: int,
    voi_posterior_samples: int,
    evaluation_cache: dict[tuple[str, str, str, int], Any],
) -> dict[str, Any]:
    incumbent = MPE2Policy.random(seed=seed, observation_dim=world.config.observation_dim, action_dim=5)
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=1.0).fit(
        np.eye(FEATURE_DIM, dtype=np.float64), np.zeros(FEATURE_DIM, dtype=np.float64)
    )
    training_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    selected_true_values: list[float] = []
    regrets: list[float] = []
    harmful = 0
    total_hf_cost = 0.0
    stop_counts: dict[str, int] = {}
    for round_id in range(rounds):
        candidates = generate_mpe2_candidates(
            incumbent, count=candidates_per_round, seed=seed + round_id, scale=candidate_scale
        )
        incumbent_proxy = _evaluate_cached(world, incumbent, seed + round_id, "observer", evaluation_cache).value
        candidate_rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            proxy_value = _evaluate_cached(world, candidate, seed + round_id, "observer", evaluation_cache).value
            proxy_delta = float(proxy_value - incumbent_proxy)
            distance = incumbent.distance(candidate)
            candidate_rows.append(
                {
                    "transition_id": f"{round_id}-{candidate_index}-{candidate.policy_id}",
                    "round_id": round_id,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "incumbent": incumbent,
                    "delta_proxy": proxy_delta,
                    "features": _features(proxy_delta, distance, candidate_index, candidates_per_round),
                    "hf_query_cost": 1.0,
                }
            )
        query_ids, _acquisition_scores, stop_reason = _select_queries(
            method,
            candidate_rows,
            posterior,
            hf_budget,
            seed + round_id,
            stop_delta=stop_delta,
            stop_eta=stop_eta,
            fantasies=voi_fantasies,
            posterior_samples=voi_posterior_samples,
        )
        if stop_reason is not None:
            stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
        queried = set(query_ids)
        oracle_mode = method == "All-HF Oracle"
        for row in candidate_rows:
            if row["transition_id"] in queried:
                incumbent_value = _evaluate_cached(world, row["incumbent"], seed + round_id, "actor", evaluation_cache).value
                candidate_value = _evaluate_cached(world, row["candidate"], seed + round_id, "actor", evaluation_cache).value
                row["delta_true"] = float(candidate_value - incumbent_value)
                row["hf_queried"] = True
                # Cost is measured in paired HF transition queries so every
                # non-oracle baseline receives the same budget unit.
                row["hf_query_cost"] = 1.0
                training_rows.append(row)
                total_hf_cost += 1.0
            else:
                correction = float(posterior.predict(np.asarray([row["features"]], dtype=np.float64))[0])
                row["predicted_delta"] = float(row["delta_proxy"] + correction)
                row["hf_queried"] = False
            if row["transition_id"] in queried:
                row["predicted_delta"] = float(row["delta_true"])
        selected_estimate = max(
            candidate_rows,
            key=lambda row: (float(row["predicted_delta"]), -int(row["candidate_index"])),
        )
        if not oracle_mode and selected_estimate["transition_id"] not in queried:
            incumbent_value = _evaluate_cached(
                world, selected_estimate["incumbent"], seed + round_id, "actor", evaluation_cache
            ).value
            candidate_value = _evaluate_cached(
                world, selected_estimate["candidate"], seed + round_id, "actor", evaluation_cache
            ).value
            selected_estimate["delta_true"] = float(candidate_value - incumbent_value)
            selected_estimate["oracle_delta_true"] = selected_estimate["delta_true"]
        for row in candidate_rows:
            if oracle_mode:
                incumbent_value = _evaluate_cached(
                    world, row["incumbent"], seed + round_id, "actor", evaluation_cache
                ).value
                candidate_value = _evaluate_cached(
                    world, row["candidate"], seed + round_id, "actor", evaluation_cache
                ).value
                row["oracle_delta_true"] = float(candidate_value - incumbent_value)
            elif row["transition_id"] in queried or (
                row is selected_estimate and row.get("delta_true") is not None
            ):
                row["oracle_delta_true"] = float(row["delta_true"])
            else:
                row["oracle_delta_true"] = None
            row["operator_shift"] = float(abs(row["delta_proxy"]))
            row["response_strength"] = 1.0
        selected = selected_estimate
        selected_true = float(selected["oracle_delta_true"])
        known_true = [
            float(row["oracle_delta_true"])
            for row in candidate_rows
            if row["oracle_delta_true"] is not None
        ]
        best_true = max(known_true) if known_true else selected_true
        selected_true_values.append(selected_true)
        regrets.append(best_true - selected_true if len(known_true) == len(candidate_rows) else 0.0)
        harmful += int(selected_true < 0.0)
        for row in candidate_rows:
            rows.append(
                {
                    "round_id": round_id,
                    "candidate_index": row["candidate_index"],
                    "transition_id": row["transition_id"],
                    "delta_proxy": row["delta_proxy"],
                    "delta_true": row["oracle_delta_true"],
                    "predicted_delta": row["predicted_delta"],
                    "hf_queried": row["hf_queried"],
                    "hf_query_cost": row["hf_query_cost"] if row["hf_queried"] else 0.0,
                    "selected": row is selected,
                    "selected_true_delta": selected_true if row is selected else None,
                    "response_strength": row["response_strength"],
                    "update_footprint": row["operator_shift"],
                    "features": row["features"],
                    "stop_reason": stop_reason,
                }
            )
        if training_rows:
            posterior.fit(
                np.asarray([row["features"] for row in training_rows], dtype=np.float64),
                np.asarray([row["delta_true"] - row["delta_proxy"] for row in training_rows], dtype=np.float64),
            )
        incumbent = selected["candidate"]
    final_value = float(_evaluate_cached(world, incumbent, seed + rounds + 1, "actor", evaluation_cache).value)
    return {
        "rows": rows,
        "summary": {
            "final_deployment_value": final_value,
            "cti": float(sum(selected_true_values)),
            "cisr": float(sum(regrets)),
            "harmful_promoted_updates": harmful,
            "reversal_acceptance_rate": float(harmful / rounds),
            "hf_cost": total_hf_cost,
            "cti_delta_std": float(np.std(selected_true_values, ddof=1)) if len(selected_true_values) > 1 else 0.0,
            "stop_counts": stop_counts,
        },
    }


def _select_queries(
    method: str,
    rows: list[dict[str, Any]],
    posterior: BayesianLinearDeltaPosterior,
    budget: int,
    seed: int,
    *,
    stop_delta: float = 0.05,
    stop_eta: float = 0.01,
    fantasies: int = 8,
    posterior_samples: int = 16,
) -> tuple[list[str], list[dict[str, Any]], str | None]:
    if method == "Proxy Only":
        return [], [], None
    if method == "All-HF Oracle":
        return [str(row["transition_id"]) for row in rows], [], None
    if method == "Random HF":
        return select_random(rows, min(budget, len(rows)), seed=seed), [], None
    if method == "Top Proxy HF":
        return select_top_proxy(rows, min(budget, len(rows))), [], None
    if method == "Uncertainty HF":
        scores = sorted(
            rows,
            key=lambda row: -float(
                posterior.predictive_variance(np.asarray([row["features"]], dtype=np.float64), include_observation=True)[0]
            ),
        )
        return [str(row["transition_id"]) for row in scores[:budget]], [], None
    if method == "PIVOT-H":
        adapter = PosteriorPredictionAdapter(posterior)
        return select_pivot(rows, adapter, min(budget, len(rows))), [], None
    if method == "PIVOT-VOI":
        scores = score_pivot_voi(
            rows,
            posterior,
            seed=seed,
            fantasies=fantasies,
            posterior_samples=posterior_samples,
        )
        selection_probability = float(scores[0]["selection_probability"]) if scores else 1.0
        max_acquisition = max((float(item["acquisition"]) for item in scores), default=0.0)
        stop, reason = should_stop(
            selection_probability=selection_probability,
            max_acquisition=max_acquisition,
            delta=stop_delta,
            eta=stop_eta,
        )
        if stop:
            return [], scores, reason
        return [str(item["transition_id"]) for item in scores[:budget]], scores, None
    raise ValueError(f"unknown E3b method: {method}")


def _features(proxy_delta: float, distance: float, candidate_index: int, candidate_count: int) -> list[float]:
    rank = float(candidate_index / max(candidate_count - 1, 1))
    return [proxy_delta, distance, distance, 1.0, 0.0, rank, float(candidate_count), 0.0, 0.0]


def _evaluate_cached(
    world: MPE2World,
    policy: MPE2Policy,
    seed: int,
    mode: str,
    cache: dict[tuple[str, str, str, int], Any],
) -> Any:
    key = (world.config.scenario, mode, policy.policy_id, int(seed))
    if key not in cache:
        cache[key] = world.evaluate_actor(policy, seed=seed) if mode == "actor" else world.evaluate_observer(policy, seed=seed)
    return cache[key]


def _aggregate_metrics(trajectories: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for trajectory in trajectories:
        method = str(trajectory["trajectory_id"]).split("|method=", 1)[-1]
        by_method.setdefault(method, []).append(trajectory)
    result: dict[str, Any] = {"methods": {}}
    cti_by_method: dict[str, float] = {}
    for method, records in by_method.items():
        cti = [float(record["cti"]) for record in records]
        cisr = [float(record["cisr"]) for record in records]
        low, high = bootstrap_mean_ci(cti, seed=20260825)
        result["methods"][method] = {
            "trajectory_count": len(records),
            "cti_mean": float(np.mean(cti)) if cti else 0.0,
            "cti_ci": [low, high],
            "cisr_mean": float(np.mean(cisr)) if cisr else 0.0,
            "final_value_mean": float(np.mean([float(record["final_deployment_value"]) for record in records])) if records else 0.0,
            "hf_cost_mean": float(np.mean([float(record["hf_cost"]) for record in records])) if records else 0.0,
        }
        cti_by_method[method] = float(np.mean(cti)) if cti else 0.0
    result["cti_pivot_voi_minus_proxy"] = cti_by_method.get("PIVOT-VOI", 0.0) - cti_by_method.get("Proxy Only", 0.0)
    paired_effects = _paired_method_effects(trajectories, "PIVOT-VOI", "Proxy Only")
    result["cti_effect_mean"] = float(np.mean(paired_effects)) if paired_effects else 0.0
    result["cti_effect_std"] = float(np.std(paired_effects, ddof=1)) if len(paired_effects) > 1 else 0.0
    result["paired_effect_count"] = len(paired_effects)
    return result


def _paired_method_effects(
    trajectories: list[dict[str, Any]], left_method: str, right_method: str
) -> list[float]:
    grouped: dict[str, dict[str, float]] = {}
    for trajectory in trajectories:
        raw_id = str(trajectory["trajectory_id"])
        base_id, separator, method = raw_id.rpartition("|method=")
        if separator:
            grouped.setdefault(base_id, {})[method] = float(trajectory["cti"])
    return [
        values[left_method] - values[right_method]
        for values in grouped.values()
        if left_method in values and right_method in values
    ]


def _write_failure(output: Path, payload: dict[str, Any], phase: str, error: Exception) -> None:
    _write_json(output / "metrics.json", {"experiment": "e3b_closed_loop", "phase": phase, "state": "IMPLEMENTATION_FAILURE", "error": str(error)})
    _write_json(output / "state.json", {"state": "IMPLEMENTATION_FAILURE", "reason": str(error)})
    _write_json(output / "provenance.json", {"config": payload, "phase": phase})


def _manifest(directory: Path) -> dict[str, Any]:
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    return {"schema_version": "v7-e3b-1", "file_count": len(files), "files": files}


def _write_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    if jsonl:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
