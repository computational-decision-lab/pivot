from __future__ import annotations

import json
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior, select_pivot_voi
from pivot.core.policy import Policy
from pivot.transfer.global_value import GlobalValueModel
from pivot.v9.artifacts import build_manifest, write_json, write_jsonl_gz, write_provenance
from pivot.v9.statistics import bootstrap_mean_ci

from .common import (
    append_failure,
    config_hash,
    initial_policy,
    load_yaml,
    paired_transition_rows,
    seed_list,
    setup_output,
    stable_seed,
    write_decision,
)


@dataclass
class TrainedModels:
    global_model: GlobalValueModel
    posterior: BayesianLinearDeltaPosterior
    calibration_rows: list[dict[str, Any]]


def run(config_path: Path, *, profile: dict[str, Any], output: Path, root: Path, resume: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    setup_output(output, resume=resume)
    experiment_id = str(config["experiment_id"])
    digest = config_hash(config)
    seeds = seed_list(config, profile)
    models = _train_models(config, profile, root, digest)
    jobs = [
        (str(environment["id"]), float(environment["response_strength"]), seed, str(method))
        for environment in config["environments"]
        for seed in seeds
        for method in config["methods"]
    ]
    all_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    requested_workers = max(1, int(profile.get("workers", 1)))
    # The pinned MPE2/PettingZoo adapter owns native simulator state that is
    # not thread-safe.  Concurrent calls can corrupt the allocator (observed
    # as SIGSEGV/corrupted double-linked list), so a mixed-world E3C run uses
    # deterministic serial execution.  Pure synthetic runs retain profile
    # parallelism.
    has_native_mpe2 = any(str(environment["id"]) == "mpe2_frozen" for environment in config["environments"])
    workers = 1 if has_native_mpe2 else requested_workers
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pivot-e3c") as executor:
        futures = [
            executor.submit(
                _run_trajectory,
                config=config,
                profile=profile,
                root=root,
                digest=digest,
                models=models,
                environment_id=environment_id,
                response_strength=response_strength,
                seed=seed,
                method=method,
            )
            for environment_id, response_strength, seed, method in jobs
        ]
        for future, (environment_id, _, seed, method) in zip(futures, jobs):
            try:
                rows, trajectory = future.result()
                all_rows.extend(rows)
                trajectories.append(trajectory)
            except (ArithmeticError, ImportError, RuntimeError, TypeError, ValueError) as error:
                append_failure(output, experiment=experiment_id, seed=seed, error=error, context={"environment": environment_id, "method": method})
    summary = _summarize_trajectories(trajectories, profile)
    validity = _validity(all_rows, config)
    minimum_seeds = int(config["statistics"]["minimum_independent_seeds"])
    powered = int(profile["seeds"]) >= minimum_seeds and all(
        int(value) >= minimum_seeds for value in summary["independent_seeds_by_environment"].values()
    )
    effect = summary["effects"].get("pivot_voi_minus_proxy_only", {})
    if not all_rows:
        status, reason = "IMPLEMENTATION_FAILURE", "no E3C trajectories completed"
    elif not validity["valid"]:
        status, reason = "DESIGN_INVALID", str(validity["reason"])
    elif not powered:
        status, reason = "UNDERPOWERED", "profile does not meet the preregistered 30-seed confirmatory rule"
    elif float(effect.get("ci_low", -math.inf)) > float(config["statistics"]["minimum_cisr_reduction"]):
        status, reason = "HYPOTHESIS_SUPPORTED", "PIVOT-VOI reduces CISR versus Proxy Only in the registered aggregate"
    else:
        status, reason = "HYPOTHESIS_NOT_SUPPORTED", "powered E3C does not support a registered PIVOT-VOI CISR reduction"
    frozen = _frozen_mpe2_reference(
        root / str(config["frozen_mpe2"]["result_root"]),
        source_result=str(config["frozen_mpe2"]["result_root"]),
    )
    write_jsonl_gz(output / "transition_rows.jsonl.gz", all_rows)
    write_jsonl_gz(output / "calibration_rows.jsonl.gz", models.calibration_rows)
    write_json(output / "trajectory_metrics.json", trajectories)
    write_json(
        output / "closed_loop_summary.json",
        {
            **summary,
            "validity": validity,
            "profile": profile,
            "requested_workers": requested_workers,
            "workers_used": workers,
            "native_simulator_serialized": has_native_mpe2,
        },
    )
    write_json(output / "mpe2_frozen_reference.json", frozen)
    write_provenance(output / "provenance.json", experiment_id=experiment_id, config=config, root=root, seed_list=seeds)
    decision = write_decision(
        output,
        experiment=experiment_id,
        status=status,
        reason=reason,
        design_valid=bool(validity["valid"]),
        powered=powered,
        allowed_claim=("No confirmatory result claim; the profile is diagnostic." if status == "UNDERPOWERED" else reason),
        forbidden_claim="PIVOT-VOI universally improves recursive self-improvement.",
        metrics={**summary, "validity": validity},
    )
    build_manifest(output, experiment_id=experiment_id, status=str(decision["status"]))
    return decision


def _train_models(config: Mapping[str, Any], profile: Mapping[str, Any], root: Path, digest: str) -> TrainedModels:
    calibration_rows: list[dict[str, Any]] = []
    start = int(config["calibration_seed_start"])
    count = min(int(config["calibration_seeds"]), max(4, int(profile["seeds"]) * 2))
    for environment in config["environments"]:
        for seed in range(start, start + count):
            calibration_rows.extend(
                paired_transition_rows(
                    experiment_id="E3C_CALIBRATION",
                    environment_id=str(environment["id"]),
                    response_strength=float(environment["response_strength"]),
                    incumbent=initial_policy(),
                    operator_family=str(config["operator_family"]),
                    operator_shift=float(config["operator_shift"]),
                    candidate_count=int(config["candidate_count"]),
                    seed=seed,
                    trajectory_id=f"calibration|{environment['id']}|{seed}",
                    round_id=0,
                    root=root,
                    config_digest=digest,
                )
            )
    global_features: list[dict[str, float]] = []
    global_values: list[float] = []
    for row in calibration_rows:
        global_features.append(_global_features(row, "incumbent"))
        global_values.append(float(row["actor_incumbent_value"]))
        global_features.append(_global_features(row, "candidate"))
        global_values.append(float(row["actor_candidate_value"]))
    global_model = GlobalValueModel()
    global_model.fit(global_features, global_values)
    features = np.asarray([row["features"] for row in calibration_rows], dtype=np.float64)
    corrections = np.asarray([float(row["delta_true"]) - float(row["delta_proxy"]) for row in calibration_rows], dtype=np.float64)
    variance = max(float(np.var(corrections, ddof=1)) if len(corrections) > 1 else 1.0, 1e-4)
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=variance).fit(features, corrections)
    return TrainedModels(global_model=global_model, posterior=posterior, calibration_rows=calibration_rows)


def _run_trajectory(
    *,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    root: Path,
    digest: str,
    models: TrainedModels,
    environment_id: str,
    response_strength: float,
    seed: int,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    incumbent = initial_policy()
    posterior = _clone_posterior(models.posterior)
    rows: list[dict[str, Any]] = []
    cti = 0.0
    cisr = 0.0
    hf_cost = 0.0
    harmful = 0
    for round_id in range(int(profile["rounds"])):
        trajectory_id = f"{environment_id}|seed={seed}|method={method}"
        candidates = paired_transition_rows(
            experiment_id="E3C",
            environment_id=environment_id,
            response_strength=response_strength,
            incumbent=incumbent,
            operator_family=str(config["operator_family"]),
            operator_shift=float(config["operator_shift"]),
            candidate_count=int(profile["candidates_per_round"]),
            seed=seed,
            trajectory_id=trajectory_id,
            round_id=round_id,
            root=root,
            config_digest=digest,
            method=method,
        )
        queried, scores = _query_and_score(method, candidates, posterior, models.global_model, config, seed + round_id)
        queried_set = set(queried)
        for order, row in enumerate(candidates):
            identifier = str(row["transition_id"])
            if identifier in queried_set:
                row["hf_queried"] = True
                row["hf_query_order"] = queried.index(identifier)
                hf_cost += float(row["hf_cost"])
                posterior = posterior.condition(np.asarray(row["features"], dtype=np.float64), float(row["delta_true"]) - float(row["delta_proxy"]))
            global_delta = _global_delta(models.global_model, row)
            pred_mean, pred_std = _posterior_delta(posterior, row)
            if method == "proxy_only":
                estimate = float(row["delta_proxy"])
            elif method == "global_value" or method == "global_voi":
                estimate = global_delta
            elif method in {"random_hf", "top_proxy_hf"}:
                estimate = float(row["delta_proxy"])
            else:
                estimate = pred_mean
            if identifier in queried_set:
                estimate = float(row["delta_true"])
            row.update(
                {
                    "estimated_delta": estimate,
                    "posterior_mean": pred_mean,
                    "posterior_std": pred_std,
                    "p_positive": _normal_positive_probability(pred_mean, pred_std),
                    "expected_regret": max(0.0, max(float(item["delta_true"]) for item in candidates) - estimate),
                    "evsi": float(scores.get(identifier, {}).get("evsi", 0.0)),
                    "evsi_per_cost": float(scores.get(identifier, {}).get("acquisition", 0.0)),
                }
            )
        selected = max(candidates, key=lambda item: float(item["estimated_delta"]))
        oracle = max(candidates, key=lambda item: float(item["delta_true"]))
        for row in candidates:
            row["selected"] = row is selected
            row["true_best"] = row is oracle
            row["CISR"] = float(oracle["delta_true"]) - float(selected["delta_true"])
            row["CTI"] = float(selected["delta_true"])
        cti += float(selected["delta_true"])
        cisr += float(oracle["delta_true"]) - float(selected["delta_true"])
        harmful += int(float(selected["delta_true"]) < 0.0)
        incumbent = Policy.from_mapping(dict(selected["candidate_parameters"]), metadata={"v9": method})
        rows.extend(candidates)
    final_row = paired_transition_rows(
        experiment_id="E3C_FINAL",
        environment_id=environment_id,
        response_strength=response_strength,
        incumbent=incumbent,
        operator_family=str(config["operator_family"]),
        operator_shift=float(config["operator_shift"]),
        candidate_count=1,
        seed=seed,
        trajectory_id=f"{environment_id}|seed={seed}|method={method}|final",
        round_id=int(profile["rounds"]),
        root=root,
        config_digest=digest,
        method=method,
    )[0]
    return rows, {
        "trajectory_id": f"{environment_id}|seed={seed}|method={method}",
        "environment_id": environment_id,
        "seed": seed,
        "method": method,
        "rounds": int(profile["rounds"]),
        "CTI": cti,
        "CISR": cisr,
        "hf_cost": hf_cost,
        "hf_queries": sum(bool(row["hf_queried"]) for row in rows),
        "harmful_promotion_rate": harmful / max(int(profile["rounds"]), 1),
        "final_deployment_value": float(final_row["actor_incumbent_value"]),
        "candidate_template_contract": "shared_update_noise_by_environment_seed_round",
    }


def _query_and_score(method: str, candidates: list[dict[str, Any]], posterior: BayesianLinearDeltaPosterior, global_model: GlobalValueModel, config: Mapping[str, Any], seed: int) -> tuple[list[str], dict[str, dict[str, float]]]:
    budget = int(config["hf_budget_per_round"])
    ids = [str(row["transition_id"]) for row in candidates]
    scores: dict[str, dict[str, float]] = {}
    if method in {"proxy_only", "global_value"}:
        return [], scores
    if method == "all_hf":
        return ids, scores
    predictions = {identifier: _posterior_delta(posterior, row)[0] for identifier, row in zip(ids, candidates)}
    stds = {identifier: _posterior_delta(posterior, row)[1] for identifier, row in zip(ids, candidates)}
    if method == "random_hf":
        rng = np.random.default_rng(seed)
        return [ids[index] for index in rng.permutation(len(ids))[:budget]], scores
    if method == "top_proxy_hf":
        return [str(row["transition_id"]) for row in sorted(candidates, key=lambda item: -float(item["delta_proxy"]))[:budget]], scores
    if method == "uncertainty_hf":
        return sorted(ids, key=lambda identifier: -stds[identifier])[:budget], scores
    if method == "paired_lucb":
        best = max(predictions.values())
        return sorted(ids, key=lambda identifier: abs(predictions[identifier] - best) / max(stds[identifier], 1e-6))[:budget], scores
    if method == "global_voi":
        values = {identifier: _global_delta(global_model, row) for identifier, row in zip(ids, candidates)}
        best = max(values.values())
        for identifier, row in zip(ids, candidates):
            gap = abs(best - values[identifier])
            cost = max(float(row["hf_cost"]), 1e-12)
            scores[identifier] = {"evsi": global_model.residual_std / (gap + 0.05), "acquisition": global_model.residual_std / ((gap + 0.05) * cost)}
        return sorted(ids, key=lambda identifier: -scores[identifier]["acquisition"])[:budget], scores
    if method == "pivot_h":
        return select_pivot(candidates, posterior, budget), scores
    if method == "pivot_voi":
        selected = select_pivot_voi(candidates, posterior, budget, seed=seed, fantasies=int(config["statistics"]["voi_fantasies"]), posterior_samples=int(config["statistics"]["voi_posterior_samples"]), cost_key="hf_cost")
        # Report the same scored EVSI surface used for selection.
        from pivot.acquisition.pivot_voi import score_pivot_voi

        for score in score_pivot_voi(candidates, posterior, seed=seed, fantasies=int(config["statistics"]["voi_fantasies"]), posterior_samples=int(config["statistics"]["voi_posterior_samples"]), cost_key="hf_cost"):
            scores[str(score["transition_id"])] = {"evsi": float(score["evsi"]), "acquisition": float(score["acquisition"])}
        return selected, scores
    raise ValueError(f"unknown E3C method: {method}")


def _global_features(row: Mapping[str, Any], role: str) -> dict[str, float]:
    parameters = row.get(f"{role}_parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    return {
        "intensity": float(parameters.get("intensity", 0.0)),
        "bias": float(parameters.get("bias", 0.0)),
        "response_strength": float(row.get("response_strength", 0.0)),
        "operator_shift": float(row.get("operator_shift", 0.0)),
    }


def _global_delta(model: GlobalValueModel, row: Mapping[str, Any]) -> float:
    return float(model.predict(_global_features(row, "candidate")) - model.predict(_global_features(row, "incumbent")))


def _posterior_delta(posterior: BayesianLinearDeltaPosterior, row: Mapping[str, Any]) -> tuple[float, float]:
    feature = np.asarray([row["features"]], dtype=np.float64)
    correction = float(posterior.predict(feature)[0])
    variance = float(posterior.predictive_variance(feature, include_observation=True)[0])
    return float(row["delta_proxy"]) + correction, math.sqrt(max(variance, 1e-12))


def _normal_positive_probability(mean: float, std: float) -> float:
    z = mean / max(std, 1e-12)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _clone_posterior(posterior: BayesianLinearDeltaPosterior) -> BayesianLinearDeltaPosterior:
    assert posterior.mean is not None and posterior.covariance is not None
    return BayesianLinearDeltaPosterior(
        prior_precision=posterior.prior_precision,
        noise_variance=posterior.noise_variance,
        mean=posterior.mean.copy(),
        covariance=posterior.covariance.copy(),
        n_observations=posterior.n_observations,
    )


def _summarize_trajectories(trajectories: list[dict[str, Any]], profile: Mapping[str, Any]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in trajectories})
    summary: dict[str, Any] = {"by_method_environment": [], "effects": {}, "independent_seeds_by_environment": {}}
    for environment in sorted({str(row["environment_id"]) for row in trajectories}):
        summary["independent_seeds_by_environment"][environment] = len({int(row["seed"]) for row in trajectories if row["environment_id"] == environment})
        for method in methods:
            subset = [row for row in trajectories if row["environment_id"] == environment and row["method"] == method]
            if not subset:
                continue
            record: dict[str, Any] = {"environment_id": environment, "method": method, "n": len(subset)}
            for metric in ("CISR", "CTI", "final_deployment_value", "hf_cost", "harmful_promotion_rate"):
                values = [float(row[metric]) for row in subset]
                low, high = bootstrap_mean_ci(values, seed=stable_seed(environment, method, metric), draws=int(profile["bootstrap_draws"]))
                record[metric] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "ci_low": low, "ci_high": high}
            summary["by_method_environment"].append(record)
    for baseline in ("proxy_only", "paired_lucb", "global_voi"):
        effects: list[float] = []
        lookup = {(str(row["environment_id"]), int(row["seed"]), str(row["method"])): row for row in trajectories}
        for environment, seed, method in list(lookup):
            if method != "pivot_voi":
                continue
            pivot = lookup[(environment, seed, method)]
            other = lookup.get((environment, seed, baseline))
            if other is not None:
                effects.append(float(other["CISR"]) - float(pivot["CISR"]))
        if effects:
            low, high = bootstrap_mean_ci(effects, seed=stable_seed(baseline), draws=int(profile["bootstrap_draws"]))
            summary["effects"][f"pivot_voi_minus_{baseline}"] = {"mean": float(np.mean(effects)), "median": float(np.median(effects)), "ci_low": low, "ci_high": high, "n": len(effects)}
    return summary


def _validity(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"valid": False, "reason": "no rows"}
    response = [float(row["delta_actor"]) - float(row["delta_direct"]) for row in rows]
    true_values = [float(row["delta_true"]) for row in rows]
    policy_distances = [float(row["policy_distance"]) for row in rows]
    response_variance = float(np.var(response))
    candidate_spread = float(np.std(true_values))
    diversity = float(np.std(policy_distances))
    floor = float(config["statistics"]["response_variance_floor"])
    spread_floor = float(config["statistics"]["candidate_spread_floor"])
    valid = response_variance > floor and candidate_spread > spread_floor and diversity > 0.0
    return {
        "valid": valid,
        "reason": "construct gates passed" if valid else "response signal or candidate diversity below preregistered floor",
        "response_difference_variance": response_variance,
        "candidate_true_spread": candidate_spread,
        "policy_distance_spread": diversity,
        "reward_ceiling": "not_applicable_no_hard_reward_ceiling",
        "hf_measurement_stability": "paired common-random-number rollout; seed is the inference cluster",
    }


def _frozen_mpe2_reference(directory: Path, *, source_result: str = "results/v7/e3b-confirmatory") -> dict[str, Any]:
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {
        "environment_id": "mpe2_frozen",
        "source_result": source_result,
        "status": state["state"],
        "reason": state["reason"],
        "cti_effect": metrics.get("cti_pivot_voi_minus_proxy"),
        "role": "preserved external adaptive null; no V9 rerun or relabeling",
    }
