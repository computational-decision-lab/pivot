#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.acquisition.footprint import select_largest_footprint
from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.random import select_random
from pivot.acquisition.top_proxy import select_top_proxy
from pivot.acquisition.uncertainty import select_uncertainty
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.transfer.differential import DifferentialModel
from pivot.transfer.sampling import stratified_transition_sample

METHODS = ("proxy_only", "random_hf", "top_proxy_hf", "largest_footprint_hf", "uncertainty_hf", "pivot", "all_hf_oracle")


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E5 matched-budget frontier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e5-budget-frontier"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "transitions"
    manifest = run_first_milestone(
        source_dir,
        PerformativeConfig(**payload.get("world", {})),
        payload.get("seeds", []),
        payload.get("candidate_scales", []),
        response_strengths=payload.get("response_strengths"),
        optimization_strengths=payload.get("optimization_strengths"),
    )
    rows = [json.loads(line) for line in (source_dir / "transitions.jsonl").read_text().splitlines()]
    calibration_budget = max(1, min(len(rows) - 1, int(payload.get("calibration_budget", len(rows) // 2))))
    sampling = str(payload.get("calibration_sampling", "stratified_round_robin"))
    if sampling != "stratified_round_robin":
        raise ValueError(f"unknown calibration sampling strategy: {sampling}")
    calibration = list(stratified_transition_sample(rows, calibration_budget))
    calibration_ids = {str(row["transition_id"]) for row in calibration}
    calibration_groups = {
        (int(row["seed"]), float(row["response_strength"]), float(row["optimization_strength"]))
        for row in calibration
    }
    test_rows = [
        row
        for row in rows
        if (
            int(row["seed"]),
            float(row["response_strength"]),
            float(row["optimization_strength"]),
        )
        not in calibration_groups
    ]
    model = DifferentialModel()
    model.fit(calibration, [float(row["delta_true"]) - float(row["delta_proxy"]) for row in calibration], [str(row["transition_id"]) for row in calibration])
    groups: dict[tuple[int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        groups[(int(row["seed"]), float(row["response_strength"]), float(row["optimization_strength"]))].append(row)
    budgets = [int(value) for value in payload.get("query_budgets", [0, 1, 2, 3, 4])]
    records: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    for budget in budgets:
        for method in METHODS:
            if method == "proxy_only" and budget != min(budgets):
                continue
            if method == "all_hf_oracle" and budget != max(budgets):
                continue
            group_results = []
            for group_key, group in sorted(groups.items()):
                effective_budget = min(budget, len(group))
                result = _evaluate_method(method, group, effective_budget, model, seed=group_key[0])
                group_results.append(result)
                ledgers.append({"method": method, "budget": effective_budget, "group": group_key, "queried_ids": result["queried_ids"]})
            if group_results:
                cti_values = [float(item["cti"]) for item in group_results]
                isr_values = [float(item["isr"]) for item in group_results]
                cti_low, cti_high = bootstrap_mean_ci(cti_values, seed=20260819)
                isr_low, isr_high = bootstrap_mean_ci(isr_values, seed=20260819)
                records.append(
                    {
                        "method": method,
                        "budget": budget,
                        "mean_cti": sum(cti_values) / len(group_results),
                        "cti_ci_low": cti_low,
                        "cti_ci_high": cti_high,
                        "mean_isr": sum(isr_values) / len(group_results),
                        "isr_ci_low": isr_low,
                        "isr_ci_high": isr_high,
                        "mean_queries": sum(int(item["queries"]) for item in group_results) / len(group_results),
                        "n_groups": len(group_results),
                        "budget_comparable": method != "all_hf_oracle",
                    }
                )
    (args.output / "budget_frontier.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "query_ledger.json").write_text(json.dumps(ledgers, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output / "budget_frontier.csv", records)
    (args.output / "provenance.json").write_text(
        json.dumps({"source_manifest": manifest.__dict__, "calibration_budget": calibration_budget, "calibration_sampling": payload.get("calibration_sampling", "stratified_round_robin"), "calibration_ids": sorted(calibration_ids), "excluded_calibration_groups": sorted(calibration_groups), "methods": METHODS, "budgets": budgets}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"E5 rows={len(rows)} groups={len(groups)} calibration={calibration_budget} output={args.output}")


def _evaluate_method(method: str, group: list[dict[str, Any]], budget: int, model: DifferentialModel, seed: int) -> dict[str, Any]:
    if method == "proxy_only":
        queried: list[str] = []
    elif method == "random_hf":
        queried = select_random(group, budget, seed=seed)
    elif method == "top_proxy_hf":
        queried = select_top_proxy(group, budget)
    elif method == "largest_footprint_hf":
        queried = select_largest_footprint(group, budget)
    elif method == "uncertainty_hf":
        queried = select_uncertainty(group, model, budget)
    elif method == "pivot":
        queried = select_pivot(group, model, budget)
    elif method == "all_hf_oracle":
        queried = [str(row["transition_id"]) for row in group]
    else:
        raise ValueError(f"unknown method: {method}")
    queried_set = set(queried)
    estimates: dict[str, float] = {}
    true_values = {str(row["transition_id"]): float(row["delta_true"]) for row in group}
    for row in group:
        identifier = str(row["transition_id"])
        if identifier in queried_set:
            estimates[identifier] = true_values[identifier]
        elif method in {"pivot", "uncertainty_hf"}:
            estimates[identifier] = model.predict_correction(row).predicted_delta
        else:
            estimates[identifier] = float(row["delta_proxy"])
    selected = max(estimates, key=estimates.get)
    selected_true = true_values[selected]
    oracle = max(true_values.values())
    return {"selected": selected, "selected_true": selected_true, "cti": selected_true, "isr": oracle - selected_true, "queries": len(queried), "queried_ids": queried}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
