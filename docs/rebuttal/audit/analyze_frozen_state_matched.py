#!/usr/bin/env python3
"""Independently replay and analyze the frozen-state matched selector run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_multiround_loop import ResponseWorld, deployment_delta, make_row, propose, sha256_bytes, stable_seed
from run_multiround_loop_v3b import (
    RESPONSE_REGIMES,
    calibration_cv,
    condition_noisy_observation,
    fit_contextual_model,
    select_query,
)
from run_frozen_state_matched import BASE_METHOD, METHODS, ROOT_START


def bootstrap(values: list[float], seed: int, draws: int = 10000) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(np.mean(x)), "ci95": [float(low), float(high)], "n_roots": int(len(x))}


def signflip(values: list[float], seed: int, draws: int = 100000) -> float:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(x)))
    observed = abs(float(np.mean(x)))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1))


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    result: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[name] = running
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    input_dir = args.input.resolve()
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import CalibrationModel

    summary = json.loads((input_dir / "summary.json").read_text())
    ledger = json.loads((input_dir / "trajectory_ledger.json").read_text())
    samples = json.loads((input_dir / "calibration_training_samples.json").read_text())
    model_record = json.loads((input_dir / "calibration_model.json").read_text())
    protocol = summary["protocol"]
    cv = calibration_cv(samples, folds=5, ridge=0.5)
    model = fit_contextual_model(
        samples,
        ridge=0.5,
        observation_variance=np.asarray(cv["observation_variance"], dtype=float),
        residual_variance=float(cv["model_residual_variance"]),
    )
    replay_model = model.to_record()
    model_hash = sha256_bytes(json.dumps(replay_model, sort_keys=True).encode())
    checks: dict[str, bool] = {
        "model_record_replayed": replay_model == model_record,
        "model_hash_matches": model_hash == summary["calibration_model_hash"],
        "ledger_hash_matches": hashlib.sha256((input_dir / "trajectory_ledger.json").read_bytes()).hexdigest()
        == summary["ledger_sha256"],
        "protocol_has_no_cross_round_update": protocol["cross_round_update"] == "none",
    }
    index = {(row["regime"], row["method"], int(row["root_id"])): row for row in ledger}
    checks["ledger_panel_unique"] = len(index) == len(ledger) == 1080
    checks["root_blocks_match"] = all(
        sorted({int(row["root_id"]) for row in ledger if row["regime"] == regime})
        == list(range(ROOT_START[regime], ROOT_START[regime] + 120))
        for regime in RESPONSE_REGIMES
    )
    checks["roots_disjoint_from_calibration"] = not any(
        int(row["root_id"]) in range(510000, 510240) for row in ledger
    )
    checks["equal_budget_and_noisy_conditioning"] = all(
        row["query_count"] == 2
        and len(row["query_ids"]) == len(set(row["query_ids"]))
        and all(detail["observation_variance"] > 0 and detail["posterior_latent_variance"] > 0 for detail in row["query_details"])
        for row in ledger
    )

    replayed = 0
    shared_panel = True
    for regime, strength in RESPONSE_REGIMES.items():
        for root_offset in range(120):
            root = ROOT_START[regime] + root_offset
            pressure = 0.10 + 0.015 * (root_offset % 7)
            incumbent = {"intensity": 0.18, "bias": 0.0}
            world = ResponseWorld(pressure=pressure, response_strength=strength)
            candidates = propose(incumbent, root=root, round_id=0)
            rows = [make_row(world, incumbent, candidate, root=root, round_id=0, index=i) for i, candidate in enumerate(candidates)]
            panel_hash = sha256_bytes(json.dumps(rows, sort_keys=True).encode())
            eval_seed = stable_seed("frozen-state-v3c-independent-eval", root)
            deployment_seed = stable_seed("frozen-state-v3c-deployment-return", root)
            eval_deltas = [deployment_delta(world, incumbent, candidate, seed=eval_seed + i) for i, candidate in enumerate(candidates)]
            for method in METHODS:
                stored = index[(regime, method, root)]
                shared_panel &= (
                    stored["panel_hash"] == panel_hash
                    and stored["candidate_panel"] == rows
                    and stored["evaluation_seed"] == eval_seed
                    and stored["deployment_seed"] == deployment_seed
                )
                posterior = model.clone().predict(
                    rows,
                    pressure=pressure,
                    response_strength=strength,
                    incumbent=incumbent,
                    round_id=0,
                    rounds=5,
                    calibration_model_type=CalibrationModel,
                )
                queried: list[str] = []
                for query_order, detail in enumerate(stored["query_details"]):
                    candidate_id, acquisition = select_query(BASE_METHOD[method], posterior, rows, set(queried))
                    replayed += 1
                    if candidate_id != detail["candidate_id"] or query_order != detail["query_order"]:
                        checks["query_choice_replay"] = False
                    query_seed = stable_seed("frozen-state-v3c-hf", root, candidate_id)
                    value = deployment_delta(world, incumbent, rows[posterior._index[candidate_id]]["candidate_parameters"], seed=query_seed)
                    if query_seed != detail["query_seed"] or not np.isclose(value, detail["observation"], atol=1e-12):
                        checks["query_observation_replay"] = False
                    condition_noisy_observation(posterior, candidate_id, value)
                    if not np.isclose(float(posterior.covariance[posterior._index[candidate_id], posterior._index[candidate_id]]), detail["posterior_latent_variance"], atol=1e-12):
                        checks["posterior_update_replay"] = False
                    queried.append(candidate_id)
                selected = posterior.select()
                selected_delta = 0.0 if selected == "incumbent" else float(eval_deltas[int(selected[1:])])
                best_delta = max([0.0, *eval_deltas])
                chosen = incumbent if selected == "incumbent" else candidates[int(selected[1:])]
                deployment_return = world.value(chosen, seed=deployment_seed, deployment=True)
                outcome = stored["deployment_outcome"]
                if selected != stored["selected_policy"]:
                    checks["terminal_selection_replay"] = False
                if not np.isclose(float(outcome["ISR"]), best_delta - selected_delta, atol=1e-12):
                    checks["isr_replay"] = False
                if not np.isclose(float(outcome["deployment_return"]), deployment_return, atol=1e-12):
                    checks["deployment_return_replay"] = False
    checks["all_rows_replayed"] = replayed == 2160
    checks["shared_frozen_panel"] = bool(shared_panel)

    root_rows: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in ledger:
        root_rows[(row["regime"], row["method"])][int(row["root_id"])] = row
    metrics: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    primary = "PIVOT-KG-Loop-v3c"
    baselines = ("IVR-Online-v3c", "Top-Proxy-Online-v3c")
    for regime in RESPONSE_REGIMES:
        metrics[regime] = {}
        roots = sorted(root_rows[(regime, primary)])
        for method in METHODS:
            method_rows = [root_rows[(regime, method)][root] for root in roots]
            metrics[regime][method] = {
                metric: bootstrap(
                    [float(row["deployment_outcome"][metric]) for row in method_rows],
                    stable_seed("frozen-state-method", regime, method, metric),
                )
                for metric in ("ISR", "deployment_return")
            }
        for baseline in baselines:
            isr_diff = [
                float(root_rows[(regime, baseline)][root]["deployment_outcome"]["ISR"])
                - float(root_rows[(regime, primary)][root]["deployment_outcome"]["ISR"])
                for root in roots
            ]
            return_diff = [
                float(root_rows[(regime, primary)][root]["deployment_outcome"]["deployment_return"])
                - float(root_rows[(regime, baseline)][root]["deployment_outcome"]["deployment_return"])
                for root in roots
            ]
            label = f"{regime}:{baseline}_minus_PIVOT_ISR"
            raw_p[label] = signflip(isr_diff, stable_seed("frozen-state-signflip", regime, baseline))
            contrasts[label] = {
                "ISR_baseline_minus_PIVOT": bootstrap(isr_diff, stable_seed("frozen-state-isr-bootstrap", regime, baseline)),
                "deployment_return_PIVOT_minus_baseline": bootstrap(return_diff, stable_seed("frozen-state-return-bootstrap", regime, baseline)),
                "two_sided_signflip_p": raw_p[label],
            }
    adjusted = holm(raw_p)
    for label in contrasts:
        contrasts[label]["holm_adjusted_p_over_six_frozen_state_contrasts"] = adjusted[label]
    checks["all_replay_checks_pass"] = all(checks.values())
    analysis = {
        "status": "PASS" if checks["all_replay_checks_pass"] else "FAIL",
        "analysis_kind": "post_review_frozen_state_matched_diagnostic",
        "ledger_sha256": summary["ledger_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "checks": checks,
        "replay_counts": {"panels": 360, "queries": replayed, "terminal_decisions": 1080, "evaluation_panels": 360},
        "method_metrics": metrics,
        "paired_contrasts": contrasts,
        "holm_family": list(raw_p),
        "interpretation": "A post-review auxiliary comparison at matched initial states; results do not replace or extend the v3c closed-loop confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Frozen-State Matched Selector Diagnostic",
        "",
        "This post-review auxiliary experiment holds the decision state, incumbent, candidate panel, proxy values, calibration prior, query budget, evaluation seeds, and terminal rule fixed across selectors. It is separate from the v3c closed-loop confirmation.",
        "",
        "## Mean ISR",
        "",
        "| Response | PIVOT-KG | IVR | Top-Proxy | IVR - PIVOT (95% CI) | Top-Proxy - PIVOT (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime in RESPONSE_REGIMES:
        ivr = contrasts[f"{regime}:IVR-Online-v3c_minus_PIVOT_ISR"]["ISR_baseline_minus_PIVOT"]
        top = contrasts[f"{regime}:Top-Proxy-Online-v3c_minus_PIVOT_ISR"]["ISR_baseline_minus_PIVOT"]
        means = [metrics[regime][method]["ISR"]["mean"] for method in METHODS]
        lines.append(
            f"| {regime} | {means[0]:.5f} | {means[1]:.5f} | {means[2]:.5f} | {ivr['mean']:.5f} [{ivr['ci95'][0]:.5f}, {ivr['ci95'][1]:.5f}] | {top['mean']:.5f} [{top['ci95'][0]:.5f}, {top['ci95'][1]:.5f}] |"
        )
    lines += ["", "## Paired contrasts", "", "Positive ISR contrast favors PIVOT; positive deployment-return contrast also favors PIVOT.", "", "| Contrast | ISR difference (95% CI) | Holm p | Deployment-return difference (95% CI) |", "|---|---:|---:|---:|"]
    for label, item in contrasts.items():
        isr = item["ISR_baseline_minus_PIVOT"]
        ret = item["deployment_return_PIVOT_minus_baseline"]
        lines.append(f"| {label} | {isr['mean']:.5f} [{isr['ci95'][0]:.5f}, {isr['ci95'][1]:.5f}] | {item['holm_adjusted_p_over_six_frozen_state_contrasts']:.5g} | {ret['mean']:.5f} [{ret['ci95'][0]:.5f}, {ret['ci95'][1]:.5f}] |")
    lines += [
        "",
        "This is an auxiliary diagnostic specified after the earlier v1-v4a and v3c results. Its tests do not extend the v3c confirmation claim or correct for that preceding variant search. All roots are retained.",
        "",
        f"Independent replay checks: `{analysis['status']}`; {replayed} query decisions replayed.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": analysis["status"], "output": str(args.output), "report": str(args.report)}, indent=2))
    return 0 if analysis["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
