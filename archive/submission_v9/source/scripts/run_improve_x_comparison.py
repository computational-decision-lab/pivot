#!/usr/bin/env python3
"""Run a matched-budget proxy versus PIVOT-X controlled comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from improve_x.acquisition.pivot_x import select_pivot_x
from improve_x.benchmark.dataset import ImprovementBenchDataset, ImprovementBenchRow
from improve_x.controlled import evaluate_transition, make_contexts, policy_from_config
from pivot.acquisition.random import select_random
from pivot.acquisition.top_proxy import select_top_proxy
from pivot.core.policy import Policy
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.transfer.differential import DifferentialModel

METHODS = ("proxy_only", "random_hf", "top_proxy_hf", "pivot_x")
WORLD_FIELD = "delta_strategic"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_id(row: ImprovementBenchRow) -> str:
    value = row.metadata.get("source_transition_id")
    return str(value) if value is not None else row.transition_id.removesuffix(f"-{row.world_level}")


def _strategic_rows(dataset: ImprovementBenchDataset, split: str) -> list[ImprovementBenchRow]:
    return [
        row
        for row in dataset.rows_for_split(split)
        if row.world_level == "strategic" and row.deployment_delta is not None
    ]


def _fit_differential_model(rows: Sequence[ImprovementBenchRow]) -> DifferentialModel:
    model = DifferentialModel()
    usable = [row for row in rows if row.deployment_delta is not None and row.proxy_delta is not None]
    model.fit(
        [row.to_record() for row in usable],
        [_correction_target(row) for row in usable],
        [_source_id(row) for row in usable],
    )
    return model


def _correction_target(row: ImprovementBenchRow) -> float:
    deployment = row.deployment_delta
    proxy = row.proxy_delta
    if deployment is None or proxy is None:
        raise ValueError("differential calibration requires proxy and deployment deltas")
    return float(deployment - proxy)


def _query_ids(
    method: str,
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    model: DifferentialModel,
    seed: int,
) -> tuple[str, ...]:
    if method == "proxy_only":
        return ()
    if method == "random_hf":
        return tuple(select_random(candidates, budget, seed=seed))
    if method == "top_proxy_hf":
        return tuple(select_top_proxy(candidates, budget))
    if method == "pivot_x":
        return tuple(select_pivot_x(candidates, model, budget))
    raise ValueError(f"unsupported comparison method: {method}")


def _estimate_delta(
    method: str,
    row: Mapping[str, Any],
    queried: set[str],
    model: DifferentialModel,
) -> float:
    identifier = str(row["transition_id"])
    if identifier in queried:
        value = row[WORLD_FIELD]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{WORLD_FIELD} must be numeric")
        return float(value)
    if method == "pivot_x":
        return float(model.predict_correction(row).predicted_delta)
    return float(row["delta_proxy"])


def _candidate_records(
    world: PerformativeWorld,
    candidates: Sequence[tuple[Any, float, int, int]],
    *,
    seed: int,
    round_id: int,
    contexts_per_transition: int,
    trajectory_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    contexts = make_contexts(seed + round_id, contexts_per_transition, f"comparison-{trajectory_id}-{round_id}")
    for transition, scale, scale_index, proposal_seed in candidates:
        evaluated = evaluate_transition(world, transition, contexts)
        evaluated = {
            **evaluated,
            "delta_true": evaluated["delta_strategic"],
            "true_incumbent_value": evaluated["strategic_incumbent_value"],
            "true_candidate_value": evaluated["strategic_candidate_value"],
        }
        records.append(
            {
                **transition.to_record(),
                **evaluated,
                "timestamp": f"comparison-round-{round_id}",
                "operator_scale": scale,
                "scale_index": scale_index,
                "proposal_seed": proposal_seed,
                "trajectory_id": trajectory_id,
                "round_id": round_id,
                "delta_true": evaluated[WORLD_FIELD],
            }
        )
    return records


def _run_method(
    method: str,
    world: PerformativeWorld,
    initial_policy: Policy,
    *,
    seed: int,
    rounds: int,
    scales: Sequence[float],
    contexts_per_transition: int,
    final_contexts: int,
    hf_queries_per_round: int,
    model: DifferentialModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from build_improvementbench import _propose_v2_candidates

    incumbent = initial_policy
    all_rows: list[dict[str, Any]] = []
    proxy_total = 0.0
    true_total = 0.0
    actor_total = 0.0
    strategic_total = 0.0
    proxy_curve = [0.0]
    true_curve = [0.0]
    actor_curve = [0.0]
    strategic_curve = [0.0]
    query_ledger: list[dict[str, Any]] = []
    trajectory_id = f"comparison-{method}-{seed}"
    parents: tuple[Policy, ...] = ()
    for round_id in range(rounds):
        generated = _propose_v2_candidates(
            incumbent,
            round_id=round_id,
            seed=seed,
            scales=scales,
            parents=parents,
            config_id=world.config.config_id,
        )
        records = _candidate_records(
            world,
            generated,
            seed=seed,
            round_id=round_id,
            contexts_per_transition=contexts_per_transition,
            trajectory_id=trajectory_id,
        )
        query_budget = 0 if method == "proxy_only" else hf_queries_per_round
        query_ids = _query_ids(method, records, query_budget, model, seed + round_id)
        queried = set(query_ids)
        query_ledger.append(
            {
                "method": method,
                "round_id": round_id,
                "budget": query_budget,
                "queried_ids": list(query_ids),
                "reason": "no_high_fidelity_query" if not query_ids else f"{method}_decision_preservation",
            }
        )
        estimates = {
            str(row["transition_id"]): _estimate_delta(method, row, queried, model)
            for row in records
        }
        selected_id = max(estimates, key=lambda identifier: (estimates[identifier], identifier))
        selected = next(row for row in records if str(row["transition_id"]) == selected_id)
        for row in records:
            identifier = str(row["transition_id"])
            row.update(
                {
                    "method": method,
                    "selected": identifier == selected_id,
                    "estimated_delta": estimates[identifier],
                    "hf_queried": identifier in queried,
                    "hf_query_reason": (
                        "no_high_fidelity_query"
                        if identifier not in queried
                        else f"{method}_decision_preservation"
                    ),
                    "hf_query_cost": 0.0 if identifier not in queried else 1.0,
                    "collection_policy": "method_specific_estimate",
                }
            )
            all_rows.append(row)
        proxy_total += float(selected["delta_proxy"])
        true_total += float(selected["delta_true"])
        actor_total += float(selected["delta_actor"])
        strategic_total += float(selected["delta_strategic"])
        proxy_curve.append(proxy_total)
        true_curve.append(true_total)
        actor_curve.append(actor_total)
        strategic_curve.append(strategic_total)
        parents = (incumbent, *parents)[:2]
        incumbent = Policy.from_mapping(
            selected["candidate_parameters"],
        )
    final_contexts_value = make_contexts(seed + rounds, final_contexts, f"comparison-final-{method}")
    final_true_performance = sum(
        world.evaluate(incumbent, context, mode="strategic").value for context in final_contexts_value
    ) / len(final_contexts_value)
    return all_rows, {
        "method": method,
        "seed": seed,
        "rounds": rounds,
        "proxy_curve": proxy_curve,
        "true_curve": true_curve,
        "actor_curve": actor_curve,
        "strategic_curve": strategic_curve,
        "final_policy_id": incumbent.policy_id,
        "final_true_performance": float(final_true_performance),
        "query_ledger": query_ledger,
        "total_hf_queries": sum(len(item["queried_ids"]) for item in query_ledger),
    }


def run_comparison(
    config: Mapping[str, Any], dataset: ImprovementBenchDataset
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_split = str(config.get("calibration_split", "train"))
    evaluation_split = str(config.get("evaluation_split", "test"))
    calibration_rows = _strategic_rows(dataset, calibration_split)
    evaluation_rows = _strategic_rows(dataset, evaluation_split)
    if not calibration_rows or not evaluation_rows:
        raise ValueError("calibration and evaluation splits must contain strategic rows")
    raw_splits = dataset.metadata.get("splits", {})
    if not isinstance(raw_splits, Mapping):
        raise TypeError("benchmark metadata must expose a split mapping")
    raw_split_seeds = raw_splits.get(evaluation_split, [])
    if not isinstance(raw_split_seeds, Sequence) or isinstance(raw_split_seeds, (str, bytes)):
        raise TypeError("benchmark metadata must expose split seed lists")
    seeds = tuple(sorted(int(seed) for seed in raw_split_seeds))
    if not seeds:
        raise ValueError("evaluation split has no seeds")
    model = _fit_differential_model(calibration_rows)
    world = PerformativeWorld(PerformativeConfig(**dict(config.get("world", {}))))
    rounds = int(config.get("rounds", 1))
    scales = tuple(float(value) for value in config.get("candidate_scales", (0.1,)))
    contexts_per_transition = int(config.get("contexts_per_transition", 1))
    final_contexts = int(config.get("final_contexts", contexts_per_transition))
    hf_queries_per_round = int(config.get("hf_queries_per_round", 1))
    if rounds <= 0 or not scales or hf_queries_per_round < 0:
        raise ValueError("comparison rounds, scales, and budget are invalid")
    requested_methods = tuple(str(value) for value in config.get("methods", METHODS))
    if not requested_methods or any(method not in METHODS for method in requested_methods):
        raise ValueError(f"methods must be selected from {METHODS}")
    initial_policy = policy_from_config(config)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for method in requested_methods:
        method_runs: list[dict[str, Any]] = []
        for seed in seeds:
            method_rows, result = _run_method(
                method,
                world,
                initial_policy,
                seed=seed,
                rounds=rounds,
                scales=scales,
                contexts_per_transition=contexts_per_transition,
                final_contexts=final_contexts,
                hf_queries_per_round=hf_queries_per_round,
                model=model,
            )
            rows.extend(method_rows)
            method_runs.append(result)
        results[method] = _aggregate_runs(method_runs)
    return rows, {
        "schema_version": "improve-x-comparison-v1",
        "calibration_split": calibration_split,
        "evaluation_split": evaluation_split,
        "calibration_transition_count": len(calibration_rows),
        "evaluation_transition_count": len(evaluation_rows),
        "evaluation_seeds": list(seeds),
        "rounds": rounds,
        "hf_queries_per_round": hf_queries_per_round,
        "methods": list(requested_methods),
        "results": results,
    }


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one evaluation run is required")
    first = dict(runs[0])
    for field in ("proxy_curve", "true_curve", "actor_curve", "strategic_curve"):
        curves = [tuple(float(value) for value in run[field]) for run in runs]
        first[field] = [sum(values) / len(values) for values in zip(*curves)]
    first["seeds"] = [int(run["seed"]) for run in runs]
    first["final_true_performance"] = sum(float(run["final_true_performance"]) for run in runs) / len(runs)
    first["total_hf_queries"] = sum(int(run["total_hf_queries"]) for run in runs)
    first["query_ledger"] = [entry for run in runs for entry in run["query_ledger"]]
    first["n_runs"] = len(runs)
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare proxy-only and PIVOT-X self-improvement")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("comparison config must be a mapping")
    dataset = ImprovementBenchDataset.read(args.benchmark)
    validation = dataset.validate()
    if not bool(validation["valid"]):
        raise ValueError(f"invalid benchmark: {validation['errors']}")
    rows, summary = run_comparison(config, dataset)
    manifest_sha256 = hashlib.sha256((args.benchmark / "manifest.json").read_bytes()).hexdigest()
    summary["benchmark_manifest_sha256"] = manifest_sha256
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rounds.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "benchmark": str(args.benchmark),
                "benchmark_manifest_sha256": manifest_sha256,
                "calibration_policy": "strategic rows in calibration split",
                "truth_policy": "all-world evaluation is retained for diagnostics; only hf_queried rows affect selection",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files = ("comparison.json", "rounds.jsonl", "provenance.json")
    manifest = {
        "schema_version": "improve-x-comparison-manifest-v1",
        "source_commit": _git_commit(),
        "benchmark_manifest_sha256": manifest_sha256,
        "row_count": len(rows),
        "files": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"methods": summary["methods"], "rows": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
