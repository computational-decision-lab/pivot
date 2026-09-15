from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior, select_pivot_voi
from pivot.v9.artifacts import build_manifest, write_json, write_jsonl_gz, write_provenance
from pivot.v9.statistics import bootstrap_mean_ci

from .common import (
    config_hash,
    initial_policy,
    load_yaml,
    paired_transition_rows,
    seed_list,
    setup_output,
    stable_seed,
    write_decision,
)


def run(config_path: Path, *, profile: dict[str, Any], output: Path, root: Path, resume: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    setup_output(output, resume=resume)
    digest = config_hash(config)
    seeds = seed_list(config, profile)
    calibration_rows = _calibration_rows(config, profile, root, digest)
    posterior = _fit_posterior(calibration_rows)
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    budget_values = [int(value) for value in config["budgets"]]
    candidate_counts = [int(value) for value in config["candidate_counts"]]
    for environment in config["environments"]:
        environment_id = str(environment["id"])
        response = float(environment["response_strength"])
        for count in candidate_counts:
            for seed in seeds:
                candidates = paired_transition_rows(
                    experiment_id="E5C",
                    environment_id=environment_id,
                    response_strength=response,
                    incumbent=initial_policy(),
                    operator_family="gradient_informed",
                    operator_shift=0.7,
                    candidate_count=count,
                    seed=seed,
                    trajectory_id=f"E5C|{environment_id}|K={count}|seed={seed}",
                    round_id=0,
                    root=root,
                    config_digest=digest,
                )
                for candidate in candidates:
                    # Cost is predeclared from footprint, independent of the
                    # realized outcome used by the selector.
                    candidate["hf_query_cost"] = float(candidate["hf_cost"]) * (
                        1.0 + 0.5 * float(candidate["policy_distance"])
                    )
                for budget in budget_values:
                    for method in config["methods"]:
                        method_name = str(method)
                        selected_ids = _select(method_name, candidates, posterior, min(budget, count), seed, config)
                        selected_id, selected_true = _select_outcome(candidates, selected_ids, method_name, posterior)
                        true_best = max(float(row["delta_true"]) for row in candidates)
                        metric = {
                            "experiment_id": "E5C",
                            "environment_id": environment_id,
                            "candidate_count": count,
                            "seed": seed,
                            "budget": budget,
                            "method": method_name,
                            "selected_candidate_id": selected_id,
                            "selected_true": selected_true,
                            "true_best": true_best,
                            "CISR": true_best - selected_true,
                            "CTI": selected_true,
                            "hf_cost": float(len(selected_ids)),
                            "queries": len(selected_ids),
                            "oracle_noncomparable": method_name == "all_hf",
                            "candidate_template_id": str(candidates[0]["candidate_template_id"]),
                        }
                        groups.append(metric)
                        for row in candidates:
                            copy = dict(row)
                            copy.update(metric)
                            copy["candidate_id"] = int(row["candidate_id"])
                            copy["selected"] = str(row["transition_id"]) == selected_id
                            copy["hf_queried"] = str(row["transition_id"]) in selected_ids
                            rows.append(copy)
    frontier = _frontier(groups, profile)
    calibration_summary = {
        "posterior_sample_stability": _posterior_stability(rows, posterior, config),
        "cost_misspecification": _cost_robustness(rows, posterior, config),
    }
    min_seeds = int(config["statistics"]["minimum_independent_seeds"])
    powered = int(profile["seeds"]) >= min_seeds and len({int(row["seed"]) for row in groups}) >= min_seeds
    pivot = [row for row in groups if row["method"] == "pivot_voi" and int(row["budget"]) == 2]
    proxy = [row for row in groups if row["method"] == "proxy_only" and int(row["budget"]) == 0]
    proxy_by_key = {(str(row["environment_id"]), int(row["candidate_count"]), int(row["seed"])): float(row["CISR"]) for row in proxy}
    pivot_by_key = {(str(row["environment_id"]), int(row["candidate_count"]), int(row["seed"])): float(row["CISR"]) for row in pivot}
    paired_effects = [proxy_by_key[key] - pivot_by_key[key] for key in sorted(proxy_by_key.keys() & pivot_by_key.keys())]
    effect = float(np.mean(paired_effects)) if paired_effects else 0.0
    effect_low, effect_high = bootstrap_mean_ci(paired_effects, seed=stable_seed("E5C", "paired-effect"), draws=int(profile["bootstrap_draws"])) if paired_effects else (0.0, 0.0)
    if not powered:
        status, reason = "UNDERPOWERED", "profile does not meet the preregistered 30-seed rule"
    elif effect_low > 0.0:
        status, reason = "HYPOTHESIS_SUPPORTED", "PIVOT-VOI has lower CISR than Proxy Only at the registered budget"
    else:
        status, reason = "HYPOTHESIS_NOT_SUPPORTED", "powered E5C does not support the registered Proxy Only contrast"
    summary = {"frontier": frontier, "calibration": calibration_summary, "pivot_voi_minus_proxy_cisr_reduction": effect, "pivot_voi_minus_proxy_ci_low": effect_low, "pivot_voi_minus_proxy_ci_high": effect_high, "paired_effect_n": len(paired_effects), "independent_seed_count": len({int(row["seed"]) for row in groups}), "group_count": len(groups), "row_count": len(rows), "profile": profile}
    write_jsonl_gz(output / "transition_rows.jsonl.gz", rows)
    write_jsonl_gz(output / "group_metrics.jsonl.gz", groups)
    write_json(output / "efficiency_summary.json", summary)
    write_json(output / "calibration_robustness.json", calibration_summary)
    _write_csv(output / "efficiency_frontier.csv", frontier)
    write_provenance(output / "provenance.json", experiment_id="E5C", config=config, root=root, seed_list=seeds)
    decision = write_decision(output, experiment="E5C", status=status, reason=reason, design_valid=True, powered=powered, allowed_claim=(reason if status == "HYPOTHESIS_SUPPORTED" else "No universal efficiency claim."), forbidden_claim="PIVOT-VOI dominates every acquisition rule at every cost.", metrics=summary)
    build_manifest(output, experiment_id="E5C", status=str(decision["status"]))
    return decision


def _calibration_rows(config: Mapping[str, Any], profile: Mapping[str, Any], root: Path, digest: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = min(int(config["calibration_seeds"]), max(4, int(profile["seeds"])))
    for environment in config["environments"]:
        for seed in range(int(config["calibration_seed_start"]), int(config["calibration_seed_start"]) + count):
            rows.extend(paired_transition_rows(experiment_id="E5C_CALIBRATION", environment_id=str(environment["id"]), response_strength=float(environment["response_strength"]), incumbent=initial_policy(), operator_family="gradient_informed", operator_shift=0.7, candidate_count=8, seed=seed, trajectory_id=f"calibration|{environment['id']}|{seed}", round_id=0, root=root, config_digest=digest))
    return rows


def _fit_posterior(rows: list[dict[str, Any]]) -> BayesianLinearDeltaPosterior:
    features = np.asarray([row["features"] for row in rows], dtype=float)
    targets = np.asarray([float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows], dtype=float)
    return BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=max(float(np.var(targets)), 1e-4)).fit(features, targets)


def _select(
    method: str,
    candidates: list[dict[str, Any]],
    posterior: BayesianLinearDeltaPosterior,
    budget: int,
    seed: int,
    config: Mapping[str, Any],
) -> list[str]:
    if budget <= 0 or method == "proxy_only":
        return []
    if method == "all_hf":
        return [str(row["transition_id"]) for row in candidates]
    if method == "random_hf":
        return [str(candidates[index]["transition_id"]) for index in np.random.default_rng(seed).permutation(len(candidates))[:budget]]
    if method == "top_proxy_hf":
        return [str(row["transition_id"]) for row in sorted(candidates, key=lambda row: -float(row["delta_proxy"]))[:budget]]
    if method == "uncertainty_hf":
        return sorted([str(row["transition_id"]) for row in candidates], key=lambda identifier: -_std_for(identifier, candidates, posterior))[:budget]
    if method == "paired_lucb":
        predictions = {str(row["transition_id"]): _posterior_for(row, posterior)[0] for row in candidates}
        return sorted(predictions, key=lambda identifier: abs(predictions[identifier] - max(predictions.values())))[:budget]
    if method == "global_voi":
        return sorted([str(row["transition_id"]) for row in candidates], key=lambda identifier: -_std_for(identifier, candidates, posterior) / (1.0 + abs(_proxy_for(identifier, candidates))))[:budget]
    if method == "pivot_h":
        return select_pivot(candidates, posterior, budget, cost_key="hf_query_cost")
    if method == "pivot_voi":
        statistics = config["statistics"]
        return select_pivot_voi(
            candidates,
            posterior,
            budget,
            seed=seed,
            fantasies=int(statistics.get("voi_fantasies", 8)),
            posterior_samples=int(statistics.get("voi_posterior_samples", 32)),
            cost_key="hf_query_cost",
        )
    raise ValueError(f"unknown E5C method: {method}")


def _select_outcome(
    candidates: list[dict[str, Any]],
    queried: list[str],
    method: str,
    posterior: BayesianLinearDeltaPosterior,
) -> tuple[str, float]:
    queried_set = set(queried)
    estimates: dict[str, float] = {}
    for row in candidates:
        identifier = str(row["transition_id"])
        if identifier in queried_set:
            estimates[identifier] = float(row["delta_true"])
        elif method in {"uncertainty_hf", "paired_lucb", "global_voi", "pivot_h", "pivot_voi"}:
            estimates[identifier] = _posterior_for(row, posterior)[0]
        else:
            estimates[identifier] = float(row["delta_proxy"])
    selected = max(estimates, key=lambda identifier: estimates[identifier])
    return selected, float(next(row["delta_true"] for row in candidates if str(row["transition_id"]) == selected))


def _frontier(groups: list[dict[str, Any]], profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (method, budget, count, environment), subset in _groupby(groups, ("method", "budget", "candidate_count", "environment_id")):
        cisr = [float(row["CISR"]) for row in subset]
        cti = [float(row["CTI"]) for row in subset]
        low, high = bootstrap_mean_ci(cisr, seed=stable_seed(method, budget, count, environment), draws=int(profile["bootstrap_draws"]))
        output.append({"method": method, "budget": budget, "candidate_count": count, "environment_id": environment, "mean_CISR": float(np.mean(cisr)), "median_CISR": float(np.median(cisr)), "CISR_ci_low": low, "CISR_ci_high": high, "mean_CTI": float(np.mean(cti)), "mean_hf_cost": float(np.mean([float(row["hf_cost"]) for row in subset])), "n_seeds": len(subset), "oracle_noncomparable": bool(subset[0]["oracle_noncomparable"])})
    return output


def _groupby(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))


def _posterior_for(row: Mapping[str, Any], posterior: BayesianLinearDeltaPosterior) -> tuple[float, float]:
    feature = np.asarray([row["features"]], dtype=float)
    correction = float(posterior.predict(feature)[0])
    std = float(np.sqrt(posterior.predictive_variance(feature, include_observation=True)[0]))
    return float(row["delta_proxy"]) + correction, std


def _std_for(identifier: str, rows: list[dict[str, Any]], posterior: BayesianLinearDeltaPosterior) -> float:
    return _posterior_for(next(row for row in rows if str(row["transition_id"]) == identifier), posterior)[1]


def _proxy_for(identifier: str, rows: list[dict[str, Any]]) -> float:
    return float(next(row for row in rows if str(row["transition_id"]) == identifier)["delta_proxy"])


def _candidate_groups(rows: list[dict[str, Any]], max_groups: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["environment_id"]), int(row["candidate_count"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["transition_id"])] = row
    return [list(grouped[key].values()) for key in sorted(grouped)[:max_groups]]


def _posterior_stability(rows: list[dict[str, Any]], posterior: BayesianLinearDeltaPosterior, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = _candidate_groups(rows, int(config["statistics"].get("calibration_max_groups", 24)))
    sample_grid = [int(value) for value in config["statistics"]["posterior_samples"]]
    reference_samples = max(sample_grid)
    reference: list[set[str]] = []
    for index, candidates in enumerate(groups):
        reference.append(
            set(
                select_pivot_voi(
                    candidates,
                    posterior,
                    min(2, len(candidates)),
                    seed=stable_seed("e5c-reference", index),
                    fantasies=int(config["statistics"].get("voi_fantasies", 8)),
                    posterior_samples=reference_samples,
                    cost_key="hf_query_cost",
                )
            )
        )
    output: list[dict[str, Any]] = []
    for samples in sample_grid:
        overlaps: list[float] = []
        for index, candidates in enumerate(groups):
            selected = set(
                select_pivot_voi(
                    candidates,
                    posterior,
                    min(2, len(candidates)),
                    seed=stable_seed("e5c-stability", index, samples),
                    fantasies=int(config["statistics"].get("voi_fantasies", 8)),
                    posterior_samples=samples,
                    cost_key="hf_query_cost",
                )
            )
            overlaps.append(len(selected & reference[index]) / max(len(selected | reference[index]), 1))
        output.append({"posterior_samples": samples, "selected_set_jaccard": float(np.mean(overlaps)) if overlaps else 0.0, "groups": len(groups), "reference_samples": reference_samples})
    return output


def _cost_robustness(rows: list[dict[str, Any]], posterior: BayesianLinearDeltaPosterior, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = _candidate_groups(rows, int(config["statistics"].get("calibration_max_groups", 24)))
    statistics = config["statistics"]
    baseline_sets: list[set[str]] = []
    for index, candidates in enumerate(groups):
        baseline_sets.append(
            set(
                select_pivot_voi(
                    candidates,
                    posterior,
                    min(2, len(candidates)),
                    seed=stable_seed("e5c-cost-reference", index),
                    fantasies=int(statistics.get("voi_fantasies", 8)),
                    posterior_samples=int(statistics.get("voi_posterior_samples", 32)),
                    cost_key="hf_query_cost",
                )
            )
        )
    output: list[dict[str, Any]] = []
    for multiplier in statistics["cost_multipliers"]:
        cisr: list[float] = []
        stability: list[float] = []
        for index, candidates in enumerate(groups):
            scaled = [dict(row, hf_query_cost=float(row.get("hf_query_cost", row.get("hf_cost", 1.0))) * float(multiplier)) for row in candidates]
            queried = select_pivot_voi(
                scaled,
                posterior,
                min(2, len(scaled)),
                seed=stable_seed("e5c-cost", index, multiplier),
                fantasies=int(statistics.get("voi_fantasies", 8)),
                posterior_samples=int(statistics.get("voi_posterior_samples", 32)),
                cost_key="hf_query_cost",
            )
            _, selected_true = _select_outcome(scaled, queried, "pivot_voi", posterior)
            true_best = max(float(row["delta_true"]) for row in scaled)
            cisr.append(true_best - selected_true)
            stability.append(len(set(queried) & baseline_sets[index]) / max(len(set(queried) | baseline_sets[index]), 1))
        output.append({"cost_multiplier": float(multiplier), "mean_CISR": float(np.mean(cisr)) if cisr else 0.0, "selected_query_jaccard": float(np.mean(stability)) if stability else 0.0, "groups": len(groups)})
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
