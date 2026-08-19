#!/usr/bin/env python3
"""Run the frozen twelve-ablation PIVOT coverage suite.

The suite is deliberately a small, dependency-light controlled harness.  It
does not tune a simulator or select a favorable subset of transitions: all
filters, train/test seeds, response levels, candidate prefixes, and query
budgets are read from the frozen YAML snapshot and are written to provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.random import select_random
from pivot.acquisition.top_proxy import select_top_proxy
from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.execution_replay import ExecutionReplayWorld
from pivot.environments.finance_backtest.config import FinanceConfig
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone
from pivot.environments.performative.world import PerformativeWorld
from pivot.environments.strategic_market.config import StrategicMarketConfig
from pivot.environments.strategic_market.world import StrategicMarketWorld
from pivot.evaluation.paired import PairedEvaluator
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.evaluation.unpaired import UnpairedEvaluator
from pivot.metrics.improvement import compute_improvement_metrics
from pivot.transfer.differential import DifferentialModel
from pivot.transfer.reversal import compare_global_vs_local
from pivot.transfer.sampling import stratified_transition_sample

REQUIRED_ABLATIONS = (
    "paired_vs_unpaired",
    "transition_vs_global_value",
    "footprint_vs_no_footprint",
    "active_vs_random_hf",
    "pivot_vs_top_proxy",
    "small_vs_large_updates",
    "weak_vs_strong_response",
    "f1_vs_f2",
    "fixed_vs_adaptive_competitors",
    "single_vs_multiple_response_models",
    "candidate_count",
    "hf_budget",
)


def run_suite(config_path: Path, output: Path) -> dict[str, Any]:
    """Run all twelve ablations into a new, self-contained output directory."""

    output = Path(output)
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        # The registered runner materializes the immutable config beside the
        # experiment output.  It is safe to retain that one input file; every
        # other pre-existing artifact means this run would overwrite evidence.
        if existing - {"config.yaml", "stdout.log", "stderr.log", "run_manifest.json"}:
            raise FileExistsError(f"refusing to overwrite non-empty ablation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_config = Path(config_path).read_bytes()
    payload = yaml.safe_load(raw_config)
    if not isinstance(payload, dict):
        raise TypeError("ablation config must be a mapping")
    seeds = tuple(int(value) for value in payload.get("seeds", ()))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("ablation config needs at least two unique seeds")

    source_dir = output / "source"
    manifest = run_first_milestone(
        source_dir,
        PerformativeConfig(**payload.get("world", {})),
        seeds,
        payload.get("candidate_scales", []),
        response_strengths=payload.get("response_strengths"),
        optimization_strengths=payload.get("optimization_strengths"),
    )
    rows = _read_jsonl(source_dir / "transitions.jsonl")
    train_seeds, test_seeds = _split_seeds(seeds, float(payload.get("train_fraction", 0.5)))
    train_rows = [row for row in rows if int(row["seed"]) in train_seeds]
    test_rows = [row for row in rows if int(row["seed"]) in test_seeds]
    if not train_rows or not test_rows:
        raise ValueError("ablation train/test split produced an empty partition")

    raw_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    summaries["paired_vs_unpaired"] = _paired_unpaired(
        payload, seeds, raw_records
    )
    summaries["transition_vs_global_value"] = _transition_global(
        train_rows, test_rows, payload, raw_records
    )
    summaries["footprint_vs_no_footprint"] = _footprint(
        train_rows, test_rows, payload, raw_records
    )
    summaries["active_vs_random_hf"] = _acquisition_contrast(
        train_rows, test_rows, payload, raw_records, left="pivot", right="random_hf"
    )
    summaries["pivot_vs_top_proxy"] = _acquisition_contrast(
        train_rows, test_rows, payload, raw_records, left="pivot", right="top_proxy_hf"
    )
    summaries["small_vs_large_updates"] = _partition_metric(
        test_rows,
        payload,
        raw_records,
        name="update_footprint",
        threshold=float(payload.get("small_update_max", 0.2)),
        low_label="small",
        high_label="large",
    )
    summaries["weak_vs_strong_response"] = _partition_metric(
        test_rows,
        payload,
        raw_records,
        name="response_strength",
        threshold=float(payload.get("weak_response_max", 0.3)),
        low_label="weak",
        high_label="strong",
        strong_min=float(payload.get("strong_response_min", 0.7)),
    )
    summaries["f1_vs_f2"] = _finance_ablation(payload, seeds, raw_records)
    summaries["fixed_vs_adaptive_competitors"] = _strategic_ablation(
        payload, seeds, raw_records
    )
    summaries["single_vs_multiple_response_models"] = _response_model_ablation(
        train_rows, test_rows, payload, raw_records
    )
    summaries["candidate_count"] = _candidate_count_ablation(
        train_rows, test_rows, payload, raw_records
    )
    summaries["hf_budget"] = _budget_ablation(
        train_rows, test_rows, payload, raw_records
    )

    if set(summaries) != set(REQUIRED_ABLATIONS):
        raise AssertionError("ablation suite did not produce the required twelve IDs")
    (output / "ablation_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_records),
        encoding="utf-8",
    )
    (output / "ablation_summary.json").write_text(
        json.dumps(
            {
                "suite_version": "controlled-ablation-v1",
                "ablation_count": len(summaries),
                "ablations": summaries,
                "train_seeds": list(train_seeds),
                "test_seeds": list(test_seeds),
                "source_row_count": len(rows),
                "source_manifest": asdict(manifest),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output / "failed_runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "suite_version": "controlled-ablation-v1",
                "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                "config_path": str(Path(config_path).resolve()),
                "seeds": list(seeds),
                "train_seeds": list(train_seeds),
                "test_seeds": list(test_seeds),
                "paired_rollouts": int(payload.get("paired_rollouts", 16)),
                "hf_budget_matched": True,
                "ground_truth_for_endogenous_response": False,
                "live_orders": False,
                "source_dataset": "synthetic-performative-v1 plus local virtual-finance-fixture-v1",
                "git_commit": _git_commit(),
                "failure_count": len(failures),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "suite_version": "controlled-ablation-v1",
        "ablation_count": len(summaries),
        "ablations": summaries,
        "failure_count": len(failures),
    }


def _paired_unpaired(payload: Mapping[str, Any], seeds: Sequence[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    world = PerformativeWorld(PerformativeConfig(**payload.get("world", {})))
    transition = _controlled_transition()
    rollouts = int(payload.get("paired_rollouts", 16))
    paired_values: dict[str, list[float]] = defaultdict(list)
    unpaired_values: dict[str, list[float]] = defaultdict(list)
    for replicate, seed in enumerate(seeds):
        base = int(seed) * 10000
        incumbent_contexts = [RolloutContext(seed=base + i, scenario_id=f"paired-{seed}-{i}") for i in range(rollouts)]
        candidate_contexts = [RolloutContext(seed=base + 1000 + i, scenario_id=f"unpaired-{seed}-{i}") for i in range(rollouts)]
        paired = PairedEvaluator(world, mode="actor").evaluate(transition, incumbent_contexts)
        unpaired = UnpairedEvaluator(world, mode="actor").evaluate(transition, incumbent_contexts, candidate_contexts)
        for variant, result in (("paired", paired), ("unpaired", unpaired)):
            for metric, value in (("delta", result.delta), ("standard_error", result.standard_error)):
                records.append(_record("paired_vs_unpaired", variant, metric, value, replicate))
                (paired_values if variant == "paired" else unpaired_values)[metric].append(float(value))
    variants = {
        variant: {metric: _stats(values) for metric, values in values_by_metric.items()}
        for variant, values_by_metric in (("paired", paired_values), ("unpaired", unpaired_values))
    }
    return {"variants": variants, "contrast": _difference_stats(paired_values["standard_error"], unpaired_values["standard_error"]), "paired_design": False}


def _transition_global(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    budget = _training_budget(train_rows, payload)
    result = compare_global_vs_local(train_rows, test_rows, budget)
    metrics = {
        "ide": (result["improvement_differential_error"], result["global_improvement_differential_error"]),
        "isc": (result["improvement_sign_consistency"], result["global_improvement_sign_consistency"]),
        "irr": (result["improvement_reversal_rate"], result["global_improvement_reversal_rate"]),
        "isr": (result["update_selection_regret"], result["global_update_selection_regret"]),
    }
    variants: dict[str, dict[str, Any]] = {"transition": {}, "global_value": {}}
    for metric, (local, global_value) in metrics.items():
        for variant, value in (("transition", local), ("global_value", global_value)):
            if value is not None:
                records.append(_record("transition_vs_global_value", variant, metric, value, 0))
                variants[variant][metric] = _stats([float(value)])
    return {"variants": variants, "hf_budget": budget, "paired_design": True}


def _footprint(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    budget = _training_budget(train_rows, payload)
    selected = stratified_transition_sample(train_rows, budget)
    variants: dict[str, dict[str, Any]] = {}
    for label, include in (("with_footprint", True), ("without_footprint", False)):
        model = _fit_model(selected, include_footprint=include)
        evaluated = _model_rows(model, test_rows)
        metrics = compute_improvement_metrics(evaluated)
        variants[label] = {}
        for metric in ("ide", "isc", "irr", "isr"):
            value = metrics.get(metric)
            if value is not None:
                records.append(_record("footprint_vs_no_footprint", label, metric, value, 0))
                variants[label][metric] = _stats([float(value)])
    return {"variants": variants, "hf_budget": budget, "paired_design": True}


def _acquisition_contrast(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]], *, left: str, right: str) -> dict[str, Any]:
    budget = int(payload.get("query_budget", 1))
    model = _fit_model(stratified_transition_sample(train_rows, _training_budget(train_rows, payload)))
    values: dict[str, dict[str, list[float]]] = {left: defaultdict(list), right: defaultdict(list)}
    for method in (left, right):
        group_metrics = _evaluate_selection(method, test_rows, model, budget)
        for metric in ("cti", "isr"):
            values[method][metric].extend(float(row[metric]) for row in group_metrics)
            for index, value in enumerate(values[method][metric][-len(group_metrics):]):
                records.append(_record("active_vs_random_hf" if right == "random_hf" else "pivot_vs_top_proxy", method, metric, value, index))
    variants = {method: {metric: _stats(metric_values) for metric, metric_values in metrics.items()} for method, metrics in values.items()}
    return {"variants": variants, "query_budget": budget, "hf_budget_matched": True, "paired_design": True}


def _partition_metric(rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]], *, name: str, threshold: float, low_label: str, high_label: str, strong_min: float | None = None) -> dict[str, Any]:
    if strong_min is None:
        low = [row for row in rows if float(row[name]) <= threshold and float(row.get("delta_proxy", 0.0)) > 0]
        high = [row for row in rows if float(row[name]) > threshold and float(row.get("delta_proxy", 0.0)) > 0]
    else:
        low = [row for row in rows if float(row[name]) <= threshold and float(row.get("delta_proxy", 0.0)) > 0]
        high = [row for row in rows if float(row[name]) >= strong_min and float(row.get("delta_proxy", 0.0)) > 0]
    variants: dict[str, Any] = {}
    for label, subset in ((low_label, low), (high_label, high)):
        value = _irr(subset)
        if value is not None:
            records.append(_record("small_vs_large_updates" if name == "update_footprint" else "weak_vs_strong_response", label, "irr", value, 0))
            variants[label] = {"irr": _stats([value]), "n_positive_proxy": len(subset)}
    return {"variants": variants, "threshold": threshold, "paired_design": True}


def _finance_ablation(payload: Mapping[str, Any], seeds: Sequence[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    finance = FinanceConfig(**payload.get("finance", {}))
    transition = _finance_transition()
    f1 = ExecutionReplayWorld(finance)
    f2_config = InteractiveMarketConfig(finance=finance, participation_rate=float(payload.get("finance_participation", 0.05)))
    f2 = InteractiveMarketWorld(f2_config)
    values: dict[str, list[float]] = {"f1_replay": [], "f2_interactive": [], "f2_minus_f1": []}
    for index, seed in enumerate(seeds):
        context = [RolloutContext(seed=int(seed), scenario_id=f"ablation-finance-{seed}")]
        f1_delta = PairedEvaluator(f1).evaluate(transition, context).delta
        f2_delta = PairedEvaluator(f2, mode="actor").evaluate(transition, context).delta
        values["f1_replay"].append(f1_delta); values["f2_interactive"].append(f2_delta); values["f2_minus_f1"].append(f2_delta - f1_delta)
    variants: dict[str, Any] = {}
    for variant, metric_values in values.items():
        records.extend(_record("f1_vs_f2", variant, "delta", value, index) for index, value in enumerate(metric_values))
        variants[variant] = {"delta": _stats(metric_values)}
    return {"variants": variants, "participation_rate": f2_config.participation_rate, "fixture_only": True, "paired_design": True}


def _strategic_ablation(payload: Mapping[str, Any], seeds: Sequence[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    finance = FinanceConfig(**payload.get("finance", {}))
    interactive = InteractiveMarketConfig(finance=finance, participation_rate=float(payload.get("finance_participation", 0.05)))
    transition = _finance_transition()
    configs = {
        "fixed": StrategicMarketConfig(interactive=interactive, opponent_mode="fixed", opponent_count=1),
        "adaptive": StrategicMarketConfig(interactive=interactive, opponent_mode="adaptive", opponent_count=1, adaptation_steps=5, learning_rate=0.2, market_share_sensitivity=0.04),
    }
    values: dict[str, dict[str, list[float]]] = {label: {"delta": [], "sirr": []} for label in configs}
    actor_values: list[float] = []
    for seed in seeds:
        context = [RolloutContext(seed=int(seed), scenario_id=f"ablation-strategic-{seed}")]
        actor = PairedEvaluator(InteractiveMarketWorld(interactive), mode="actor").evaluate(transition, context).delta
        actor_values.append(actor)
        for label, config in configs.items():
            delta = PairedEvaluator(StrategicMarketWorld(config), mode="strategic").evaluate(transition, context).delta
            values[label]["delta"].append(delta)
            values[label]["sirr"].append(float(actor > 0 and delta < 0))
    variants: dict[str, Any] = {}
    for label, metrics in values.items():
        variants[label] = {}
        for metric, metric_values in metrics.items():
            records.extend(_record("fixed_vs_adaptive_competitors", label, metric, value, index) for index, value in enumerate(metric_values))
            variants[label][metric] = _stats(metric_values)
    return {"variants": variants, "fixture_only": True, "paired_design": True}


def _response_model_ablation(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    single_level = float(payload.get("single_response_strength", 0.3))
    multi_levels = {float(value) for value in payload.get("multiple_response_strengths", [0.0, 0.3, 0.7])}
    heldout = float(payload.get("heldout_response_strength", 0.9))
    budget = _training_budget(train_rows, payload)
    variants: dict[str, Any] = {}
    for label, selected_rows in (("single", [row for row in train_rows if float(row["response_strength"]) == single_level]), ("multiple", [row for row in train_rows if float(row["response_strength"]) in multi_levels])):
        selected = stratified_transition_sample(selected_rows, min(budget, len(selected_rows)))
        model = _fit_model(selected)
        evaluated = _model_rows(model, [row for row in test_rows if float(row["response_strength"]) == heldout])
        metrics = compute_improvement_metrics(evaluated)
        variants[label] = {}
        for metric in ("ide", "isc", "irr", "isr"):
            value = metrics.get(metric)
            if value is not None:
                records.append(_record("single_vs_multiple_response_models", label, metric, value, 0))
                variants[label][metric] = _stats([float(value)])
    return {"variants": variants, "train_response": {"single": single_level, "multiple": sorted(multi_levels)}, "heldout_response": heldout, "hf_budget_matched": True, "paired_design": True}


def _candidate_count_ablation(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    model = _fit_model(stratified_transition_sample(train_rows, _training_budget(train_rows, payload)))
    budget = int(payload.get("query_budget", 1))
    variants: dict[str, Any] = {}
    for count in [int(value) for value in payload.get("candidate_counts", [2, 4])]:
        restricted = [row for row in test_rows if int(row.get("candidate_index", 0)) < count]
        group_metrics = _evaluate_selection("pivot", restricted, model, min(budget, count))
        cti = [float(row["cti"]) for row in group_metrics]
        isr = [float(row["isr"]) for row in group_metrics]
        label = f"n={count}"
        variants[label] = {"cti": _stats(cti), "isr": _stats(isr), "n_groups": len(group_metrics)}
        for metric, values in (("cti", cti), ("isr", isr)):
            records.extend(_record("candidate_count", label, metric, value, index) for index, value in enumerate(values))
    return {"variants": variants, "query_budget": budget, "hf_budget_matched": True, "paired_design": True}


def _budget_ablation(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], payload: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    model = _fit_model(stratified_transition_sample(train_rows, _training_budget(train_rows, payload)))
    budgets = [int(value) for value in payload.get("query_budgets", [0, 1, 2, 3, 4])]
    variants: dict[str, Any] = {}
    for budget in budgets:
        for method in ("pivot", "random_hf", "top_proxy_hf"):
            group_metrics = _evaluate_selection(method, test_rows, model, min(budget, 4))
            cti = [float(row["cti"]) for row in group_metrics]
            isr = [float(row["isr"]) for row in group_metrics]
            label = f"{method}@{budget}"
            variants[label] = {"cti": _stats(cti), "isr": _stats(isr), "budget": budget}
            for metric, values in (("cti", cti), ("isr", isr)):
                records.extend(_record("hf_budget", label, metric, value, index) for index, value in enumerate(values))
    return {"variants": variants, "budgets": budgets, "hf_budget_matched": True, "paired_design": True}


def _evaluate_selection(method: str, rows: Sequence[Mapping[str, Any]], model: DifferentialModel, budget: int) -> list[dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    results: list[dict[str, float]] = []
    for group_id, group in sorted(groups.items()):
        if not group:
            continue
        effective = min(max(0, budget), len(group))
        if method == "pivot":
            queried = set(select_pivot(group, model, effective)) if effective else set()
        elif method == "random_hf":
            queried = set(select_random(group, effective, seed=_stable_seed(group_id))) if effective else set()
        elif method == "top_proxy_hf":
            queried = set(select_top_proxy(group, effective)) if effective else set()
        else:
            raise ValueError(f"unknown ablation selection method: {method}")
        estimates: dict[str, float] = {}
        truth: dict[str, float] = {}
        for row in group:
            identifier = str(row["transition_id"])
            truth[identifier] = float(row["delta_true"])
            if identifier in queried:
                estimates[identifier] = truth[identifier]
            elif method == "pivot":
                estimates[identifier] = model.predict_correction(row).predicted_delta
            else:
                estimates[identifier] = float(row["delta_proxy"])
        selected = max(estimates, key=estimates.get)
        results.append({"cti": truth[selected], "isr": max(truth.values()) - truth[selected]})
    return results


def _model_rows(model: DifferentialModel, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        prediction = model.predict_correction(row).predicted_delta
        row["predicted_delta"] = prediction
        row["delta_proxy"] = prediction
        groups[_group_key(row)].append(row)
    for group in groups.values():
        selected = max(group, key=lambda row: float(row["delta_proxy"]))
        for row in group:
            row["selected"] = row is selected
    return result


def _fit_model(rows: Sequence[Mapping[str, Any]], *, include_footprint: bool = True) -> DifferentialModel:
    model = DifferentialModel(include_footprint=include_footprint)
    model.fit(rows, [float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows], [str(row["transition_id"]) for row in rows])
    return model


def _training_budget(rows: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> int:
    return max(1, min(len(rows), int(payload.get("calibration_budget", 24))))


def _record(ablation_id: str, variant: str, metric: str, value: float, replicate: int) -> dict[str, Any]:
    return {"ablation_id": ablation_id, "variant": variant, "metric": metric, "value": float(value), "replicate": int(replicate)}


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    low, high = bootstrap_mean_ci(values, seed=20260819 + len(values))
    return {"estimate": mean(float(value) for value in values), "ci_low": low, "ci_high": high, "n": len(values)}


def _difference_stats(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    return _stats([float(a) - float(b) for a, b in zip(left, right)])


def _irr(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["delta_true"]) < 0 for row in rows) / len(rows)


def _group_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row.get(key)) for key in ("seed", "response_strength", "optimization_strength"))


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _split_seeds(seeds: Sequence[int], fraction: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    cut = max(1, min(len(seeds) - 1, round(len(seeds) * fraction)))
    return tuple(seeds[:cut]), tuple(seeds[cut:])


def _controlled_transition() -> PolicyTransition:
    return PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6}),
        round_id=0,
        candidate_index=0,
        improvement_operator="ablation-controlled",
        edit_type="intensity",
        config_id="controlled-ablation-v1",
    )


def _finance_transition() -> PolicyTransition:
    return PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2, "position_size": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6, "position_size": 0.6}),
        round_id=0,
        candidate_index=0,
        improvement_operator="ablation-finance-fixture",
        edit_type="intensity_and_position_size",
        config_id="finance-fixture-v1",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen PIVOT ablation suite")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_suite(args.config, args.output)
    print(json.dumps({"ablation_count": summary["ablation_count"], "failure_count": summary["failure_count"]}))


if __name__ == "__main__":
    main()
