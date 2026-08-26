from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pivot.v9.artifacts import build_manifest, write_json, write_jsonl_gz, write_provenance
from pivot.v9.environments import PerformativeControlWorld
from pivot.v9.operators import generate_candidate_batch, policy_distance
from pivot.v9.opponents import OPPONENT_MODES, Opponent
from pivot.v9.statistics import bootstrap_mean_ci

from .common import config_hash, initial_policy, load_yaml, seed_list, setup_output, write_decision


def run(config_path: Path, *, profile: dict[str, Any], output: Path, root: Path, resume: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    setup_output(output, resume=resume)
    digest = config_hash(config)
    seeds = seed_list(config, profile)
    rows: list[dict[str, Any]] = []
    world = PerformativeControlWorld()
    for mode in config["opponent_modes"]:
        for strength in config["adaptation_strengths"]:
            for seed in seeds:
                incumbent = initial_policy()
                transitions = generate_candidate_batch(incumbent, family=str(config["operator_family"]), shift_level=float(config["operator_shift"]), count=int(config["candidate_count"]), seed=seed, config_id="E7C")
                opponent = Opponent(str(mode), float(strength), seed)
                observer_incumbent = world.evaluate(incumbent, seed=seed, mode="observer").value
                actor_incumbent = world.evaluate(incumbent, seed=seed, mode="actor").value
                strategic_incumbent = opponent.strategic_value(incumbent, seed=seed)
                candidate_values: list[float] = []
                for index, transition in enumerate(transitions):
                    observer_candidate = world.evaluate(transition.candidate, seed=seed, mode="observer").value
                    actor_candidate = world.evaluate(transition.candidate, seed=seed, mode="actor").value
                    strategic_candidate = opponent.strategic_value(transition.candidate, seed=seed)
                    direct = float(observer_candidate - observer_incumbent)
                    actor = float(actor_candidate - actor_incumbent)
                    strategic = float(strategic_candidate - strategic_incumbent)
                    row = {
                        "experiment_id": "E7C",
                        "environment_id": "performative_control",
                        "environment_family": "performative_control",
                        "opponent_mode": str(mode),
                        "opponent_strength": float(strength),
                        "opponent_seed": seed,
                        "trajectory_id": f"E7C|{mode}|strength={strength}|seed={seed}",
                        "candidate_id": index,
                        "candidate_policy_id": transition.candidate.policy_id,
                        "delta_direct": direct,
                        "delta_actor": actor,
                        "delta_strategic": strategic,
                        "strategic_effect": strategic - actor,
                        "actor_positive": actor > 1e-9,
                        "strategic_reversal": actor > 1e-9 and strategic < -1e-9,
                        "policy_distance": policy_distance(incumbent, transition.candidate),
                        "adaptation_magnitude": abs(strategic - actor),
                        "opponent_reward_change": -strategic,
                        "opponent_policy_shift": float(strength * policy_distance(incumbent, transition.candidate)),
                        "source_commit": None,
                        "config_hash": digest,
                    }
                    rows.append(row)
                    candidate_values.append(strategic)
                _ = candidate_values
    summary = _summarize(rows, profile)
    min_seeds = int(config["statistics"]["minimum_independent_seeds"])
    powered = len({int(row["opponent_seed"]) for row in rows}) >= min_seeds and int(profile["seeds"]) >= min_seeds
    adaptive_effects = [float(value["mean_strategic_effect"]) for value in summary["by_mode"] if value["opponent_mode"] in {"best_response", "gradient_adaptive", "rl_evolutionary"}]
    mean_effect = float(np.mean(adaptive_effects)) if adaptive_effects else 0.0
    if not powered:
        status, reason = "UNDERPOWERED", "profile does not meet the preregistered 30-seed cluster rule"
    elif mean_effect < float(config["statistics"]["minimum_strategic_effect"]):
        status, reason = "HYPOTHESIS_SUPPORTED", "independently adaptive opponent modes reduce focal update value"
    else:
        status, reason = "HYPOTHESIS_NOT_SUPPORTED", "strategic effect is not negative in the registered adaptive modes"
    write_jsonl_gz(output / "strategic_rows.jsonl.gz", rows)
    write_json(output / "strategic_summary.json", summary)
    write_provenance(output / "provenance.json", experiment_id="E7C", config=config, root=root, seed_list=seeds)
    _write_csv(output / "strategic_summary.csv", summary["by_mode"])
    decision = write_decision(output, experiment="E7C", status=status, reason=reason, design_valid=True, powered=powered, allowed_claim=("Held-out opponent modes only." if status == "HYPOTHESIS_SUPPORTED" else reason), forbidden_claim="General equilibrium or all strategic environments reverse self-improvement updates.", metrics=summary)
    build_manifest(output, experiment_id="E7C", status=str(decision["status"]))
    return decision


def _summarize(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opponent_mode"])].append(row)
    by_mode: list[dict[str, Any]] = []
    for mode in OPPONENT_MODES:
        subset = grouped.get(mode, [])
        cluster_effects = _cluster_means(subset, "strategic_effect")
        cluster_reversals = _cluster_means(subset, "strategic_reversal", denominator_key="actor_positive")
        clusters = len(cluster_effects)
        low, high = bootstrap_mean_ci(cluster_effects, seed=202608267 + OPPONENT_MODES.index(mode), draws=int(profile["bootstrap_draws"])) if cluster_effects else (0.0, 0.0)
        by_mode.append({"opponent_mode": mode, "mean_strategic_effect": float(np.mean(cluster_effects)) if cluster_effects else 0.0, "median_strategic_effect": float(np.median(cluster_effects)) if cluster_effects else 0.0, "strategic_effect_ci_low": low, "strategic_effect_ci_high": high, "SIRR": float(np.mean(cluster_reversals)) if cluster_reversals else 0.0, "actor_positive_n": sum(int(row["actor_positive"]) for row in subset), "cluster_n": clusters, "row_n": len(subset)})
    adaptive = [row for row in rows if row["opponent_mode"] in {"best_response", "gradient_adaptive", "rl_evolutionary"}]
    adaptive_cluster_effects = _cluster_means(adaptive, "strategic_effect")
    low, high = bootstrap_mean_ci(adaptive_cluster_effects, seed=202608299, draws=int(profile["bootstrap_draws"])) if adaptive_cluster_effects else (0.0, 0.0)
    return {"by_mode": by_mode, "adaptive_effect_mean": float(np.mean(adaptive_cluster_effects)) if adaptive_cluster_effects else 0.0, "adaptive_effect_ci_low": low, "adaptive_effect_ci_high": high, "independent_seed_count": len({int(row["opponent_seed"]) for row in rows}), "row_count": len(rows), "cluster_unit": "opponent_seed"}


def _cluster_means(rows: list[dict[str, Any]], value_key: str, *, denominator_key: str | None = None) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if denominator_key is not None and not bool(row[denominator_key]):
            continue
        grouped[int(row["opponent_seed"])].append(float(bool(row[value_key])) if denominator_key is not None else float(row[value_key]))
    return [float(np.mean(values)) for values in grouped.values()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
