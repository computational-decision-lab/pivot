#!/usr/bin/env python3
"""Paired-root bootstrap and randomization analysis for loop-v2 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.percentile(sample, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(low), float(high)], "n_roots": int(len(x))}


def signflip_p(values: list[float], *, seed: int, draws: int) -> float:
    x = np.asarray(values, dtype=float)
    if np.all(np.abs(x) <= 1e-15):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(x)))
    observed = abs(float(x.mean()))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def analyze(input_dir: Path, output: Path, *, bootstrap_draws: int, signflip_draws: int) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"output exists; choose a new file: {output}")
    summary = json.loads((input_dir / "summary.json").read_text())
    trajectories = json.loads((input_dir / "trajectories.json").read_text())
    ledger = json.loads((input_dir / "trajectory_ledger.json").read_text())
    methods = list(summary["methods"])
    primary = "PIVOT-KG-Loop-v2"
    baselines = ("IVR-Online", "Top-Proxy-Online")
    regimes = list(summary["protocol"]["response_regimes"])
    index = {(row["regime"], row["method"], int(row["root_id"])): row for row in trajectories}
    ledger_index = {(row["regime"], row["method"], int(row["root_id"]), int(row["round_id"])): row for row in ledger}
    all_roots = {
        regime: sorted({int(row["root_id"]) for row in trajectories if row["regime"] == regime})
        for regime in regimes
    }
    rounds = int(summary["protocol"]["rounds"])
    expected_trajectory_count = len(regimes) * int(summary["protocol"]["roots_per_regime"]) * len(methods)
    expected_ledger_count = expected_trajectory_count * rounds
    protocol_checks = {
        "unique_regime_method_root_trajectories": len(index) == len(trajectories) == expected_trajectory_count,
        "unique_regime_method_root_round_ledger": len(ledger_index) == len(ledger) == expected_ledger_count,
        "complete_method_root_round_panel": len(ledger_index) == expected_ledger_count,
        "test_roots_disjoint_from_calibration_roots": all(
            root < 500000 or root >= 500000 + int(summary["protocol"]["calibration_roots"])
            for roots in all_roots.values() for root in roots
        ),
        "every_method_uses_same_total_query_budget": all(
            row["query_count"] == int(summary["protocol"]["hf_budget_per_round"])
            and len(row["query_ids"]) == len(set(row["query_ids"]))
            and len(row["query_details"]) == row["query_count"]
            for row in ledger
        ),
        "online_sentinel_and_method_specific_query_order": all(
            row["query_details"][0]["kind"] == "common_sentinel"
            and row["query_details"][1]["kind"] == "method_specific"
            and row["query_details"][0]["candidate_id"] == f"c{int(row['round_id']) % 4}"
            for row in ledger
            if row["method"] in {"PIVOT-KG-Loop-v2", *baselines}
        ),
        "deployment_seed_replay_and_cross_method_sharing": all(
            row["deployment_seed"] == stable_seed("loop-v2-deployment-response", row["root_id"], row["round_id"])
            and len({
                ledger_index[(row["regime"], method, int(row["root_id"]), int(row["round_id"]))]["deployment_seed"]
                for method in methods
            }) == 1
            for row in ledger
        ),
        "evaluation_seed_replay_and_cross_method_sharing": all(
            row["evaluation_seed"] == stable_seed("loop-v2-independent-eval", row["root_id"], row["round_id"])
            and len({
                ledger_index[(row["regime"], method, int(row["root_id"]), int(row["round_id"]))]["evaluation_seed"]
                for method in methods
            }) == 1
            for row in ledger
        ),
        "query_seed_replay": all(
            detail["query_seed"] == stable_seed("loop-v2-hf", row["root_id"], row["round_id"], detail["candidate_id"])
            for row in ledger for detail in row["query_details"]
        ),
        "terminal_isr_recomputed": all(
            np.isclose(
                float(row["deployment_outcome"]["ISR"]),
                max([0.0, *[float(value) for value in row["deployment_outcome"]["independent_deltas"]]])
                - float(row["deployment_outcome"]["selected_delta"]),
                atol=1e-12,
            )
            for row in ledger
        ),
    }
    protocol_checks["all_pass"] = all(protocol_checks.values())
    comparisons: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    method_metrics: dict[str, Any] = {}
    for regime in regimes:
        roots = all_roots[regime]
        method_metrics[regime] = {}
        for method in methods:
            values = [index[(regime, method, root)] for root in roots]
            method_metrics[regime][method] = {
                metric: bootstrap(
                    [float(row[metric]) for row in values],
                    seed=stable_seed("loop-v2-method", summary["protocol"]["split"], regime, method, metric),
                    draws=bootstrap_draws,
                )
                for metric in ("CISR", "cumulative_utility", "harmful_adoption_rate", "retention_rate")
            }
        for baseline in baselines:
            diffs = [
                float(index[(regime, baseline, root)]["CISR"])
                - float(index[(regime, primary, root)]["CISR"])
                for root in roots
            ]
            label = f"{regime}:{baseline}_minus_{primary}_CISR"
            pvalues[label] = signflip_p(
                diffs,
                seed=stable_seed("loop-v2-signflip", summary["protocol"]["split"], regime, baseline),
                draws=signflip_draws,
            )
            utility_diffs = [
                float(index[(regime, primary, root)]["cumulative_utility"])
                - float(index[(regime, baseline, root)]["cumulative_utility"])
                for root in roots
            ]
            comparisons[label] = {
                "estimand": "baseline cumulative ISR minus PIVOT-KG-Loop-v2 cumulative ISR; positive favors PIVOT-KG-Loop-v2",
                "paired_root_bootstrap": bootstrap(
                    diffs,
                    seed=stable_seed("loop-v2-paired-bootstrap", summary["protocol"]["split"], regime, baseline),
                    draws=bootstrap_draws,
                ),
                "two_sided_signflip_p": pvalues[label],
                "utility_primary_minus_baseline": bootstrap(
                    utility_diffs,
                    seed=stable_seed("loop-v2-utility-bootstrap", summary["protocol"]["split"], regime, baseline),
                    draws=bootstrap_draws,
                ),
            }
    adjusted = holm_adjust(pvalues)
    for label, value in adjusted.items():
        comparisons[label]["holm_adjusted_p_over_all_regime_baseline_contrasts"] = value

    per_round: dict[str, Any] = {}
    for regime in regimes:
        per_round[regime] = {}
        for round_id in range(rounds):
            per_round[regime][str(round_id + 1)] = {}
            for method in methods:
                values = [
                    float(ledger_index[(regime, method, root, round_id)]["deployment_outcome"]["ISR"])
                    for root in all_roots[regime]
                ]
                per_round[regime][str(round_id + 1)][method] = bootstrap(
                    values,
                    seed=stable_seed("loop-v2-round", summary["protocol"]["split"], regime, round_id, method),
                    draws=bootstrap_draws,
                )

    report = {
        "analysis_kind": "paired_root_loop_v2_confirmation_or_development",
        "split": summary["protocol"]["split"],
        "protocol_sha256": summary["protocol_sha256"],
        "source_hashes": {
            "summary": sha256_file(input_dir / "summary.json"),
            "trajectories": sha256_file(input_dir / "trajectories.json"),
            "ledger": sha256_file(input_dir / "trajectory_ledger.json"),
        },
        "n_roots_per_regime": {regime: len(roots) for regime, roots in all_roots.items()},
        "method_metrics": method_metrics,
        "protocol_audit_checks": protocol_checks,
        "primary_paired_cumulative_isr_contrasts": comparisons,
        "holm_family": list(pvalues),
        "per_round_isr": per_round,
        "success_rule": "within each declared regime, both baseline-minus-PIVOT paired ISR 95% CI lower bounds exceed zero and both Holm-adjusted sign-flip p-values are below 0.05",
        "limitations": [
            "Development results are exploratory and cannot support a confirmatory superiority claim.",
            "Evidence is limited to the declared PerformativeWorld response regimes, proposer, candidate generator, and five-round horizon.",
            "The online contextual model is a linear correction model; no general response model is estimated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--signflip-draws", type=int, default=100000)
    args = parser.parse_args()
    report = analyze(args.input, args.output, bootstrap_draws=args.bootstrap_draws, signflip_draws=args.signflip_draws)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
