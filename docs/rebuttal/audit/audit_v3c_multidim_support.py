#!/usr/bin/env python3
"""Audit contextual-calibration support and residuals for the frozen v3c ledger.

This is a descriptive post-hoc audit.  It reconstructs the feature vectors used
by the sealed contextual model, but never filters roots, changes decisions, or
refits the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "pressure",
    "response_strength",
    "incumbent_intensity",
    "incumbent_bias",
    "delta_intensity",
    "delta_bias",
    "proxy_delta",
    "pressure_x_delta_intensity",
    "response_x_delta_intensity",
    "round_fraction",
)
METHODS = (
    "PIVOT-KG-Loop-v3c",
    "IVR-Online-v3c",
    "Top-Proxy-Online-v3c",
)
REGIMES = ("low", "medium", "high")
RESPONSE_STRENGTH = {"low": 0.35, "medium": 0.60, "high": 0.85}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_vector(
    candidate: dict[str, Any],
    *,
    pressure: float,
    response_strength: float,
    incumbent: dict[str, float],
    round_id: int,
    rounds: int,
) -> np.ndarray:
    params = candidate["candidate_parameters"]
    delta_intensity = float(params["intensity"] - incumbent["intensity"])
    delta_bias = float(params["bias"] - incumbent["bias"])
    return np.asarray(
        [
            pressure,
            response_strength,
            incumbent["intensity"],
            incumbent["bias"],
            delta_intensity,
            delta_bias,
            float(candidate["delta_proxy"]),
            pressure * delta_intensity,
            response_strength * delta_intensity,
            round_id / max(1, rounds - 1),
        ],
        dtype=float,
    )


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def describe(values: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "unique_values": sorted({float(value) for value in values})
        if values.size <= 20
        else None,
    }


def reconstruct_test_rows(
    ledger: list[dict[str, Any]], rounds: int
) -> list[dict[str, Any]]:
    """Reconstruct incumbent-dependent feature vectors in ledger order."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[(str(row["regime"]), str(row["method"]), int(row["root_id"]))].append(row)

    records: list[dict[str, Any]] = []
    for regime in REGIMES:
        for method in METHODS:
            roots = sorted(
                root_id for (row_regime, row_method, root_id) in grouped
                if row_regime == regime and row_method == method
            )
            for root_id in roots:
                rows = grouped[(regime, method, root_id)]
                incumbent = {"intensity": 0.18, "bias": 0.0}
                for row in sorted(rows, key=lambda item: int(item["round_id"])):
                    round_id = int(row["round_id"])
                    pressure = float(row["pressure_before"])
                    strength = float(row["response_strength"])
                    candidates = sorted(row["candidate_panel"], key=lambda item: int(item["candidate_index"]))
                    for index, candidate in enumerate(candidates):
                        raw = feature_vector(
                            candidate,
                            pressure=pressure,
                            response_strength=strength,
                            incumbent=incumbent,
                            round_id=round_id,
                            rounds=rounds,
                        )
                        actual_correction = float(
                            row["deployment_outcome"]["independent_deltas"][index]
                            - candidate["delta_proxy"]
                        )
                        records.append(
                            {
                                "regime": regime,
                                "method": method,
                                "root_id": root_id,
                                "round_id": round_id,
                                "candidate_index": index,
                                "features": raw,
                                "actual_correction": actual_correction,
                                "isr": float(row["deployment_outcome"]["ISR"]),
                            }
                        )
                    selected = str(row["selected_policy"])
                    if selected != "incumbent":
                        index = int(selected[1:])
                        chosen = candidates[index]["candidate_parameters"]
                        incumbent = {
                            "intensity": float(chosen["intensity"]),
                            "bias": float(chosen["bias"]),
                        }
    return records


def model_prediction(raw: np.ndarray, model: dict[str, Any], regime: str, candidate_index: int) -> tuple[float, float]:
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    beta = np.asarray(model["beta_mean"], dtype=float)
    beta_cov = np.asarray(model["beta_covariance"], dtype=float)
    design = np.concatenate(([1.0], (raw - mean) / scale))
    predicted = float(design @ beta)
    strength_index = min(
        range(len(model["observation_variance_regimes"])),
        key=lambda index: abs(float(model["observation_variance_regimes"][index]) - RESPONSE_STRENGTH[regime]),
    )
    observation = float(model["single_rollout_observation_variance"][strength_index][candidate_index])
    variance = float(design @ beta_cov @ design)
    variance += float(model["residual_variance"]) + observation
    return predicted, max(variance, 1e-12)


def calibration_cv_audit(samples: list[dict[str, Any]], script_dir: Path) -> dict[str, Any]:
    """Recompute root-held-out five-fold residuals from calibration samples."""
    sys.path.insert(0, str(script_dir))
    from run_multiround_loop_v3b import fit_contextual_model

    roots = sorted({int(row["root_id"]) for row in samples})
    errors: list[float] = []
    covered: list[bool] = []
    by_response: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_roots = {root for index, root in enumerate(roots) if index % 5 == fold}
        train_rows = [row for row in samples if int(row["root_id"]) not in validation_roots]
        validation_rows = [row for row in samples if int(row["root_id"]) in validation_roots]
        model = fit_contextual_model(train_rows, ridge=0.5)
        fold_errors: list[float] = []
        fold_covered: list[bool] = []
        for row in validation_rows:
            raw = np.asarray(row["features"], dtype=float)
            design = model.design(raw.reshape(1, -1)).reshape(-1)
            prediction = float(design @ model.beta_mean)
            error = float(row["correction"] - prediction)
            predictive_variance = float(design @ model.beta_covariance @ design)
            predictive_variance += float(model.residual_variance)
            predictive_variance += float(row["training_noise_variance"])
            inside = abs(error) <= 1.96 * np.sqrt(max(predictive_variance, 1e-12))
            errors.append(error)
            covered.append(inside)
            fold_errors.append(error)
            fold_covered.append(inside)
            by_response[str(float(row["response_strength"]))].append((error, inside))
        fold_rows.append(
            {
                "fold": fold,
                "validation_roots": len(validation_roots),
                "candidate_observations": len(validation_rows),
                "rmse": float(np.sqrt(np.mean(np.square(fold_errors)))),
                "mae": float(np.mean(np.abs(fold_errors))),
                "working_95pct_prediction_interval_coverage": float(np.mean(fold_covered)),
            }
        )
    error_array = np.asarray(errors, dtype=float)
    return {
        "method": "recomputed five-fold root-held-out Bayesian linear regression",
        "folds": fold_rows,
        "overall": {
            "n_candidate_vectors": int(error_array.size),
            "rmse": float(np.sqrt(np.mean(error_array**2))),
            "mae": float(np.mean(np.abs(error_array))),
            "bias_prediction_minus_actual": float(-np.mean(error_array)),
            "working_95pct_prediction_interval_coverage": float(np.mean(covered)),
        },
        "by_response_strength": {
            strength: {
                "n_candidate_vectors": len(values),
                "rmse": float(np.sqrt(np.mean([error**2 for error, _ in values]))),
                "mae": float(np.mean([abs(error) for error, _ in values])),
                "working_95pct_prediction_interval_coverage": float(np.mean([inside for _, inside in values])),
            }
            for strength, values in sorted(by_response.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input.resolve()
    samples = json.loads((input_dir / "calibration_training_samples.json").read_text())
    ledger = json.loads((input_dir / "trajectory_ledger.json").read_text())
    model = json.loads((input_dir / "calibration_model.json").read_text())
    paired = json.loads((input_dir / "paired_analysis.json").read_text())
    summary = json.loads((input_dir / "summary.json").read_text())
    rounds = max(int(row["round_id"]) for row in ledger) + 1

    train = np.asarray([row["features"] for row in samples], dtype=float)
    test_records = reconstruct_test_rows(ledger, rounds)
    test = np.vstack([record["features"] for record in test_records])
    if train.shape[1] != len(FEATURE_NAMES) or test.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected v3c feature dimensionality")

    train_stats = {name: describe(train[:, index]) for index, name in enumerate(FEATURE_NAMES)}
    test_stats = {name: describe(test[:, index]) for index, name in enumerate(FEATURE_NAMES)}
    train_min = np.min(train, axis=0)
    train_max = np.max(train, axis=0)
    feature_ood = (test < train_min - 1e-12) | (test > train_max + 1e-12)

    # Use a fixed ridge for the support distance because round_fraction is
    # constant in calibration and makes the empirical covariance singular.
    train_center = train.mean(axis=0)
    train_scale = train.std(axis=0, ddof=1)
    train_scale[train_scale < 1e-12] = 1.0
    train_z = (train - train_center) / train_scale
    test_z = (test - train_center) / train_scale
    covariance = np.cov(train_z, rowvar=False)
    covariance = (covariance + covariance.T) / 2 + np.eye(len(FEATURE_NAMES)) * 1e-3
    inverse_covariance = np.linalg.pinv(covariance)
    train_distance = np.einsum("ij,jk,ik->i", train_z, inverse_covariance, train_z)
    test_distance = np.einsum("ij,jk,ik->i", test_z, inverse_covariance, test_z)
    distance_p95 = percentile(train_distance, 0.95)
    distance_p99 = percentile(train_distance, 0.99)

    index = 0
    coverage: dict[str, Any] = {}
    residuals: dict[str, Any] = {}
    ood_isr: dict[str, Any] = {}
    for regime in REGIMES:
        coverage[regime] = {}
        residuals[regime] = {}
        ood_isr[regime] = {}
        for method in METHODS:
            coverage[regime][method] = {}
            residuals[regime][method] = {}
            ood_isr[regime][method] = {}
            for round_id in range(rounds):
                records = [
                    record
                    for record in test_records
                    if record["regime"] == regime
                    and record["method"] == method
                    and record["round_id"] == round_id
                ]
                rows = np.vstack([record["features"] for record in records])
                ood_rows = (rows < train_min - 1e-12) | (rows > train_max + 1e-12)
                # The global reconstruction is root-major, while this slice is
                # round-major. Recompute distances for the selected group to
                # avoid relying on an ordering coincidence.
                group_z = (rows - train_center) / train_scale
                distances = np.einsum("ij,jk,ik->i", group_z, inverse_covariance, group_z)
                index += len(records)
                predictions = []
                errors = []
                covered = []
                for record in records:
                    predicted, variance = model_prediction(
                        record["features"], model, regime, record["candidate_index"]
                    )
                    error = record["actual_correction"] - predicted
                    predictions.append(predicted)
                    errors.append(error)
                    covered.append(abs(error) <= 1.96 * np.sqrt(variance))
                errors_array = np.asarray(errors, dtype=float)
                if len(records) % 4:
                    raise AssertionError("each ledger state must contain four candidate vectors")
                state_ood_cells = ood_rows.reshape(-1, 4, len(FEATURE_NAMES))
                state_ood = np.any(state_ood_cells, axis=(1, 2))
                state_ood_excluding_round = np.any(state_ood_cells[:, :, :9], axis=(1, 2))
                isr_values = np.asarray(
                    [records[offset]["isr"] for offset in range(0, len(records), 4)], dtype=float
                )
                coverage[regime][method][str(round_id)] = {
                    "n_states": int(len(records) // 4),
                    "n_candidate_vectors": int(len(records)),
                    "feature_ood_rate": float(np.mean(ood_rows)),
                    "state_any_feature_ood_rate": float(np.mean(state_ood)),
                    "state_any_non_round_feature_ood_rate": float(np.mean(state_ood_excluding_round)),
                    "mean_mahalanobis_squared": float(np.mean(distances)),
                    "mahalanobis_squared_gt_train_p95_rate": float(np.mean(distances > distance_p95)),
                    "mahalanobis_squared_gt_train_p99_rate": float(np.mean(distances > distance_p99)),
                }
                residuals[regime][method][str(round_id)] = {
                    "n_candidate_vectors": int(len(errors_array)),
                    "mae": float(np.mean(np.abs(errors_array))),
                    "rmse": float(np.sqrt(np.mean(errors_array**2))),
                    "bias_actual_minus_predicted": float(np.mean(errors_array)),
                    "working_95pct_prediction_interval_coverage": float(np.mean(covered)),
                }
                in_support = isr_values[~state_ood]
                out_support = isr_values[state_ood]
                ood_isr[regime][method][str(round_id)] = {
                    "in_support_states": int(in_support.size),
                    "in_support_mean_isr": float(np.mean(in_support)) if in_support.size else None,
                    "ood_states": int(out_support.size),
                    "ood_mean_isr": float(np.mean(out_support)) if out_support.size else None,
                }

    if index != len(test_records):
        raise AssertionError("internal test-record indexing error")
    constant_features = {
        name: [float(train_center[index])]
        for index, name in enumerate(FEATURE_NAMES)
        if float(np.ptp(train[:, index])) <= 1e-12
    }
    cv_audit = calibration_cv_audit(samples, Path(__file__).resolve().parent)
    recorded_cv = summary["calibration"]["five_fold_cv"]
    if not np.isclose(cv_audit["overall"]["rmse"], recorded_cv["rmse"], atol=1e-10):
        raise AssertionError("recomputed calibration CV RMSE does not match sealed summary")
    report = {
        "status": "PASS",
        "diagnostic_id": "V3C-MULTIDIM-SUPPORT-2026-11-10",
        "purpose": "Descriptive calibration-support, distance, and residual audit; no root filtering or protocol changes.",
        "input_hashes": {
            "calibration_training_samples": sha256_file(input_dir / "calibration_training_samples.json"),
            "calibration_model": sha256_file(input_dir / "calibration_model.json"),
            "trajectory_ledger": sha256_file(input_dir / "trajectory_ledger.json"),
            "paired_analysis": sha256_file(input_dir / "paired_analysis.json"),
        },
        "counts": {
            "calibration_rows": int(len(samples)),
            "calibration_roots": int(len({int(row["root_id"]) for row in samples})),
            "ledger_rows": int(len(ledger)),
            "test_candidate_vectors": int(len(test_records)),
            "test_states": int(len(ledger)),
            "rounds": rounds,
        },
        "feature_names": list(FEATURE_NAMES),
        "calibration_feature_support": train_stats,
        "test_feature_ranges": test_stats,
        "constant_calibration_features": constant_features,
        "support_limits": {
            name: {"min": float(train_min[index]), "max": float(train_max[index])}
            for index, name in enumerate(FEATURE_NAMES)
        },
        "multivariate_support": {
            "distance": "ridge-regularized squared Mahalanobis distance on standardized features",
            "ridge": 1e-3,
            "training_distance_p95": distance_p95,
            "training_distance_p99": distance_p99,
            "test_distance_mean": float(np.mean(test_distance)),
            "test_distance_max": float(np.max(test_distance)),
            "gt_training_p95_rate": float(np.mean(test_distance > distance_p95)),
            "gt_training_p99_rate": float(np.mean(test_distance > distance_p99)),
        },
        "coverage_by_regime_method_round": coverage,
        "residuals_by_regime_method_round": residuals,
        "calibration_root_heldout_residuals": cv_audit,
        "ood_isr_association_by_regime_method_round": ood_isr,
        "identification_limits": [
            "round_fraction is constant at zero in the calibration samples; its later-round effect is not identified by the fitted calibration model.",
            "Residuals use the stored independent deployment evaluation delta and are descriptive; they do not refit or alter the sealed model.",
            "Support and OOD rates are computed on all ledger rows and are not used to exclude roots or select a favorable protocol.",
            "OOD-stratified ISR is descriptive because OOD status is method-induced by the evolving selector trajectory; this comparison does not establish a causal mechanism for ISR differences.",
            "The six v3c Holm tests remain the prespecified paired contrasts in paired_analysis.json; this diagnostic is not an additional hypothesis test.",
        ],
        "paired_analysis_overall_success": bool(paired.get("overall_success", False)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V3C Multidimensional Calibration Support Audit",
        "",
        "This is a descriptive audit of the sealed v3c calibration and ledger. It does not filter roots, refit the model, or change any selector result.",
        "",
        f"- Calibration rows: {len(samples)} ({report['counts']['calibration_roots']} roots); ledger rows: {len(ledger)}.",
        f"- Multivariate distance uses standardized features with ridge `1e-3`; training distance thresholds are p95 `{distance_p95:.4f}` and p99 `{distance_p99:.4f}`.",
        f"- Test candidate vectors above those thresholds: {report['multivariate_support']['gt_training_p95_rate']:.1%} (p95), {report['multivariate_support']['gt_training_p99_rate']:.1%} (p99).",
        f"- Recomputed five-fold root-held-out calibration correction RMSE/MAE: {cv_audit['overall']['rmse']:.4f}/{cv_audit['overall']['mae']:.4f}; working 95% interval coverage: {cv_audit['overall']['working_95pct_prediction_interval_coverage']:.1%}.",
        "",
        "## Feature support",
        "",
        "| Feature | Calibration min | Calibration max | Test min | Test max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in FEATURE_NAMES:
        train_row = train_stats[name]
        test_row = test_stats[name]
        lines.append(
            f"| `{name}` | {train_row['min']:.6g} | {train_row['max']:.6g} | {test_row['min']:.6g} | {test_row['max']:.6g} |"
        )
    lines += [
        "",
        "The calibration feature `round_fraction` is constant at zero, so the model has no calibration variation from which to identify its later-round effect. Pressure, incumbent parameters, proxy deltas, and interaction features also show test values outside their one-dimensional calibration ranges on some trajectories.",
        "",
        "## OOD and residual summary",
        "",
        "| Regime | Method | Round | Candidate feature-cell OOD | State any OOD (incl. round) | State any OOD (excl. round) | Mean squared distance | Residual RMSE | 95% interval coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        for method in METHODS:
            for round_id in range(rounds):
                cell = coverage[regime][method][str(round_id)]
                residual_cell = residuals[regime][method][str(round_id)]
                lines.append(
                    f"| {regime} | {method} | {round_id + 1} | {cell['feature_ood_rate']:.1%} | {cell['state_any_feature_ood_rate']:.1%} | {cell['state_any_non_round_feature_ood_rate']:.1%} | {cell['mean_mahalanobis_squared']:.3f} | {residual_cell['rmse']:.3f} | {residual_cell['working_95pct_prediction_interval_coverage']:.1%} |"
                )
    lines += [
        "",
        "The state-OOD column including `round_fraction` is 100% after round 1 because that feature is constant at zero in calibration but nonzero later. The companion column excludes `round_fraction` and is the more informative state-support diagnostic. OOD rows are a property of the method-induced trajectory and therefore cannot be interpreted as an independent covariate: a selector can change its own future state distribution. The residual and interval columns use the stored independent deployment evaluation and are reported for calibration diagnosis only.",
        "",
        "## OOD-stratified ISR",
        "",
        "ISR is the state-level deployment-selection regret. The groups below use the full-feature support definition (any candidate feature outside its calibration min/max, including `round_fraction`); entries give root counts and mean ISR. Round-1 OOD groups are small (1-7 roots). In rounds 2-5, every state is OOD because calibration `round_fraction` is fixed at zero. Since selectors induce their own evolving state distributions, these descriptive group means do not identify a causal mechanism for ISR differences.",
        "",
        "| Regime | Method | Round | In-support states; mean ISR | OOD states; mean ISR |",
        "|---|---|---:|---:|---:|",
    ]
    for regime in REGIMES:
        for method in METHODS:
            for round_id in range(rounds):
                cell = ood_isr[regime][method][str(round_id)]
                in_support = (
                    "-"
                    if cell["in_support_states"] == 0
                    else f"{cell['in_support_states']}; {cell['in_support_mean_isr']:.4f}"
                )
                out_support = (
                    "-"
                    if cell["ood_states"] == 0
                    else f"{cell['ood_states']}; {cell['ood_mean_isr']:.4f}"
                )
                lines.append(
                    f"| {regime} | {method} | {round_id + 1} | {in_support} | {out_support} |"
                )
    lines += [
        "",
        "## Limits",
        "",
        "- The six Holm-adjusted v3c comparisons remain the prespecified primary tests; this audit adds no success claim.",
        "- OOD rates do not justify removing unfavorable roots or changing response regimes.",
        "- Persistent cross-round calibration learning was not part of v3c (`cross_round_update=none`).",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
