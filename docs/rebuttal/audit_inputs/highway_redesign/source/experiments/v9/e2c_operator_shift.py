from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pivot.v9.artifacts import build_manifest, write_json, write_jsonl_gz, write_provenance
from pivot.v9.statistics import (
    bootstrap_mean_ci,
    density_diagnostics,
    improvement_metrics,
    spearman,
)

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


def run(config_path: Path, *, profile: dict[str, Any], output: Path, root: Path, resume: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    setup_output(output, resume=resume)
    experiment_id = str(config["experiment_id"])
    digest = config_hash(config)
    seeds = seed_list(config, profile)
    candidate_count = min(int(config["candidate_count"]), int(profile["candidates_per_round"]))
    rows: list[dict[str, Any]] = []
    failures = 0
    for environment in config["environments"]:
        environment_id = str(environment["id"])
        for response_strength in environment["response_strengths"]:
            for family in config["operator_families"]:
                for shift in config["shift_levels"]:
                    for seed in seeds:
                        try:
                            trajectory_id = f"{environment_id}|r={response_strength}|op={family}|shift={shift}|seed={seed}"
                            rows.extend(
                                paired_transition_rows(
                                    experiment_id=experiment_id,
                                    environment_id=environment_id,
                                    response_strength=float(response_strength),
                                    incumbent=initial_policy(),
                                    operator_family=str(family),
                                    operator_shift=float(shift),
                                    candidate_count=candidate_count,
                                    seed=seed,
                                    trajectory_id=trajectory_id,
                                    round_id=0,
                                    root=root,
                                    config_digest=digest,
                                )
                            )
                        except (ArithmeticError, ImportError, RuntimeError, TypeError, ValueError) as error:
                            failures += 1
                            append_failure(
                                output,
                                experiment=experiment_id,
                                seed=seed,
                                error=error,
                                context={"environment": environment_id, "operator": family, "shift": shift},
                            )
    summary = _summarize(rows, config, profile)
    validity = _validity(rows, float(config["statistics"]["response_variance_floor"]))
    minimum_seeds = int(config["statistics"]["minimum_independent_seeds"])
    powered = profile.get("seeds") == minimum_seeds and summary["independent_seed_count"] >= minimum_seeds
    if not rows:
        status, reason = "IMPLEMENTATION_FAILURE", "no E2C transitions completed"
    elif not validity["valid"]:
        status, reason = "DESIGN_INVALID", validity["reason"]
    elif not powered:
        status, reason = "UNDERPOWERED", "profile does not meet the preregistered 30-seed confirmatory rule"
    elif float(summary["shift_effect_ci_low"]) > float(config["statistics"]["minimum_shift_effect_if"]):
        status, reason = "HYPOTHESIS_SUPPORTED", "operator shift increases operator-relative absolute improvement error"
    else:
        status, reason = "HYPOTHESIS_NOT_SUPPORTED", "powered E2C does not support the preregistered shift effect"
    summary.update({"validity": validity, "failures": failures, "profile": profile, "config_hash": digest})
    write_jsonl_gz(output / "transition_rows.jsonl.gz", rows)
    write_json(output / "operator_shift_summary.json", summary)
    write_json(output / "density_diagnostics.json", density_diagnostics(config["shift_levels"]))
    write_provenance(output / "provenance.json", experiment_id=experiment_id, config=config, root=root, seed_list=seeds)
    decision = write_decision(
        output,
        experiment=experiment_id,
        status=status,
        reason=reason,
        design_valid=bool(validity["valid"]),
        powered=powered,
        allowed_claim=("No confirmatory result claim; the profile is diagnostic." if status == "UNDERPOWERED" else reason),
        forbidden_claim="Global evaluator quality universally determines update fidelity.",
        metrics=summary,
    )
    build_manifest(output, experiment_id=experiment_id, status=str(decision["status"]))
    return decision


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["environment_id"]),
                float(row["response_strength"]),
                str(row["operator_family"]),
                float(row["operator_shift"]),
            )
        ].append(row)
    reference_shift = float(config["statistics"]["reference_shift"])
    global_reference: dict[tuple[str, float], dict[str, Any]] = {}
    for key_tuple, subset in grouped.items():
        environment_id, response, family, shift = key_tuple
        if shift != reference_shift:
            continue
        reference_key = (environment_id, response)
        policy_errors = [abs(float(row["proxy_candidate_value"]) - float(row["actor_candidate_value"])) for row in subset]
        proxy_values = [float(row["proxy_candidate_value"]) for row in subset]
        true_values = [float(row["actor_candidate_value"]) for row in subset]
        global_reference[reference_key] = {
            "policy_mae": float(np.mean(policy_errors)),
            "policy_rmse": float(math.sqrt(np.mean(np.square(policy_errors)))),
            "policy_spearman": spearman(proxy_values, true_values),
            "policy_pearson": float(np.corrcoef(proxy_values, true_values)[0, 1]) if len(set(proxy_values)) > 1 else 0.0,
            "global_transition_ide": improvement_metrics(subset)["IDE"],
            "n": len(subset),
        }
    cells: list[dict[str, Any]] = []
    for cell_key, subset in sorted(grouped.items()):
        environment_id, response, family, shift = cell_key
        per_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            per_seed[int(row["seed"])].append(row)
        seed_if = [float(improvement_metrics(value)["IDE"] or 0.0) for value in per_seed.values()]
        low, high = bootstrap_mean_ci(seed_if, seed=stable_seed(*cell_key), draws=int(profile["bootstrap_draws"]))
        cell_metrics = improvement_metrics(subset)
        cells.append(
            {
                "environment_id": environment_id,
                "response_strength": response,
                "operator_family": family,
                "operator_shift": shift,
                "chi_square_shift": float(np.expm1(shift * shift)),
                "improvement_fidelity": cell_metrics["IDE"],
                "ISC": cell_metrics["ISC"],
                "IRR": cell_metrics["IRR"],
                "ISR": cell_metrics["ISR"],
                "if_ci_low": low,
                "if_ci_high": high,
                "independent_seed_n": len(per_seed),
                "transition_n": len(subset),
                **{f"global_{name}": value for name, value in global_reference.get((environment_id, response), {}).items()},
            }
        )
    effects: list[float] = []
    low_shift = min(float(value) for value in config["shift_levels"])
    high_shift = max(float(value) for value in config["shift_levels"])
    by_seed: dict[tuple[str, float, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[(str(row["environment_id"]), float(row["response_strength"]), str(row["operator_family"]), int(row["seed"]), float(row["operator_shift"]))].append(row)
    prefixes = {(a, b, c, d) for a, b, c, d, _ in by_seed}
    for prefix in prefixes:
        low_rows = by_seed.get((*prefix, low_shift), [])
        high_rows = by_seed.get((*prefix, high_shift), [])
        if low_rows and high_rows:
            effects.append(float(improvement_metrics(high_rows)["IDE"] or 0.0) - float(improvement_metrics(low_rows)["IDE"] or 0.0))
    effect_low, effect_high = bootstrap_mean_ci(effects, seed=202608261, draws=int(profile["bootstrap_draws"]))
    return {
        "cells": cells,
        "shift_effect_mean": float(np.mean(effects)),
        "shift_effect_median": float(np.median(effects)),
        "shift_effect_ci_low": effect_low,
        "shift_effect_ci_high": effect_high,
        "effect_size_standardized": float(np.mean(effects) / max(float(np.std(effects, ddof=1)), 1e-12)) if len(effects) > 1 else 0.0,
        "independent_seed_count": len({int(row["seed"]) for row in rows}),
        "transition_count": len(rows),
        "global_reference": {f"{key[0]}|{key[1]}": value for key, value in global_reference.items()},
    }


def _validity(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    differences = [float(row["delta_actor"]) - float(row["delta_direct"]) for row in rows]
    variance = float(np.var(differences)) if differences else 0.0
    if not rows:
        return {"valid": False, "reason": "no rows", "response_difference_variance": variance}
    if variance <= floor:
        return {"valid": False, "reason": "response signal below preregistered floor", "response_difference_variance": variance}
    return {"valid": True, "reason": "response signal and transitions available", "response_difference_variance": variance}
