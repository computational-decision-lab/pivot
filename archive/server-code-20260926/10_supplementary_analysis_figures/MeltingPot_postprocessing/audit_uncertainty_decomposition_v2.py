"""Post-result uncertainty decomposition for the frozen Melting Pot cohort.

This is a development-only audit.  It never changes frozen decisions and it
must not be used to relabel the 44000--44029 cohort as confirmation evidence
for a revised method.
"""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np


GRID = np.asarray([0.125, 0.225, 0.325, 0.425, 0.575, 0.675, 0.775, 0.875])
FEATURES = np.column_stack(((GRID - 0.5) / 0.25, np.abs(GRID - 0.5) / 0.25))


def _read_json(tar: tarfile.TarFile, name: str):
    member = tar.getmember(name)
    with tar.extractfile(member) as stream:
        assert stream is not None
        return json.load(stream)


def _psd_summary(matrix: np.ndarray) -> dict:
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    clipped = np.maximum(values, 0.0)
    projected = (vectors * clipped) @ vectors.T
    negative_mass = float(np.maximum(-values, 0.0).sum())
    scale = float(np.abs(values).sum())
    return {
        "raw_eigenvalues": values.tolist(),
        "projected_eigenvalues": clipped.tolist(),
        "negative_eigenvalue_mass_fraction": negative_mass / scale if scale else 0.0,
        "raw": matrix.tolist(),
        "psd_projection": projected.tolist(),
    }


def _target_decomposition(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """observed is [root, candidate, block], predicted is [root, candidate]."""
    residual = observed - predicted[:, :, None]
    difference = observed[:, :, 0] - observed[:, :, 1]
    mean_residual = residual.mean(axis=2)
    roots = observed.shape[0]

    block_noise_by_candidate = np.mean(difference**2, axis=0) / 2.0
    mean_noise_by_candidate = block_noise_by_candidate / 2.0
    systematic_bias = mean_residual.mean(axis=0)
    between_variance = np.var(mean_residual, axis=0, ddof=1)
    world_variance = between_variance - mean_noise_by_candidate
    bias_sq_unbiased = systematic_bias**2 - between_variance / roots

    centered = mean_residual - systematic_bias
    between_covariance = centered.T @ centered / (roots - 1)
    block_noise_covariance = difference.T @ difference / (2.0 * roots)
    mean_noise_covariance = block_noise_covariance / 2.0
    world_covariance = between_covariance - mean_noise_covariance

    return {
        "latent_mse_cross_block": float(np.mean(residual[:, :, 0] * residual[:, :, 1])),
        "observation_variance_per_block": float(np.mean(block_noise_by_candidate)),
        "observation_variance_of_ab_mean": float(np.mean(mean_noise_by_candidate)),
        "systematic_bias_by_candidate": systematic_bias.tolist(),
        "systematic_bias_squared_naive": float(np.mean(systematic_bias**2)),
        "systematic_bias_squared_mom": float(np.mean(bias_sq_unbiased)),
        "between_root_variance_of_ab_mean": float(np.mean(between_variance)),
        "cross_world_variance_mom": float(np.mean(world_variance)),
        "cross_world_variance_by_candidate": world_variance.tolist(),
        "block_noise_covariance": block_noise_covariance.tolist(),
        "cross_world_covariance": _psd_summary(world_covariance),
    }


def _bootstrap_scalar(data, statistic, *, draws=5000, seed=20260917):
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    values = np.empty(draws)
    for i in range(draws):
        values[i] = statistic(data[rng.integers(n, size=n)])
    point = float(statistic(data))
    return {"estimate": point, "ci95": np.quantile(values, [0.025, 0.975]).tolist()}


def _formal_data(archive: Path, predictions_csv: Path, calibration: dict):
    predictions = {}
    with predictions_csv.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            predictions[(int(row["seed"]), int(row["adaptation"]), int(row["candidate"]))] = float(row["predicted_delta"])

    output = {}
    root = "melting_method_confirm_20260917"
    with tarfile.open(archive, "r:gz") as tar:
        for h in (4, 32):
            deployment = np.empty((30, 8, 2))
            proxy = np.empty((30, 8, 2))
            selection_observation = np.empty((30, 8))
            predicted_deployment = np.empty((30, 8))
            correction_mean = None
            for r, seed in enumerate(range(44000, 44030)):
                summary = _read_json(tar, f"{root}/seed_{seed}/summary.json")
                posterior = _read_json(tar, f"{root}/seed_{seed}/posterior.json")
                decisions = _read_json(tar, f"{root}/seed_{seed}/decisions_frozen.json")
                beta = np.asarray(posterior[str(h)]["mean"], dtype=float)
                candidate_correction = FEATURES @ beta
                if correction_mean is None:
                    correction_mean = candidate_correction
                else:
                    np.testing.assert_allclose(correction_mean, candidate_correction)
                rows = {
                    (int(row["candidate"]), row["block"]): row
                    for row in summary["mechanism_audit"]
                    if int(row["adaptation"]) == h
                }
                for c in range(8):
                    for b, block in enumerate(("A", "B")):
                        deployment[r, c, b] = float(rows[c, block]["deployment_delta_audit"])
                        proxy[r, c, b] = float(rows[c, block]["proxy_delta_audit"])
                    predicted_deployment[r, c] = predictions[seed, h, c]
                all_hf = [
                    row
                    for row in decisions
                    if row["method"] == "all_hf_reference" and int(row["adaptation"]) == h
                ]
                assert len(all_hf) == 1
                for c in range(8):
                    selection_observation[r, c] = float(all_hf[0]["observed"][str(c)])

            assert correction_mean is not None
            predicted_correction = np.broadcast_to(correction_mean, (30, 8))
            predicted_proxy = predicted_deployment - predicted_correction
            gap = deployment - proxy

            dep = _target_decomposition(deployment, predicted_deployment)
            gap_result = _target_decomposition(gap, predicted_correction)
            proxy_result = _target_decomposition(proxy, predicted_proxy)

            hierarchical_beta = np.asarray(calibration[str(h)]["beta_population_mean"], dtype=float)
            hierarchical_correction = np.broadcast_to(FEATURES @ hierarchical_beta, (30, 8))
            hierarchical_gap_result = _target_decomposition(gap, hierarchical_correction)
            hierarchical_deployment_result = _target_decomposition(
                deployment, predicted_proxy + hierarchical_correction
            )
            sigma_beta = np.asarray(
                calibration[str(h)]["beta_cross_world_covariance"]["psd_projection"], dtype=float
            )
            mean_covariance = np.asarray(
                calibration[str(h)]["population_mean_estimation_covariance"], dtype=float
            )
            candidate_covariance = FEATURES @ (sigma_beta + mean_covariance) @ FEATURES.T
            formal_world_covariance = np.asarray(
                hierarchical_gap_result["cross_world_covariance"]["psd_projection"], dtype=float
            )
            projector = FEATURES @ np.linalg.inv(FEATURES.T @ FEATURES) @ FEATURES.T
            total_trace = float(np.trace(formal_world_covariance))
            projected_trace = float(np.trace(projector @ formal_world_covariance @ projector))

            dep_residual = deployment - predicted_deployment[:, :, None]
            gap_residual = gap - predicted_correction[:, :, None]
            proxy_residual = proxy - predicted_proxy[:, :, None]
            identity_error = dep_residual - gap_residual - proxy_residual

            per_root = np.column_stack(
                (
                    np.mean(dep_residual[:, :, 0] * dep_residual[:, :, 1], axis=1),
                    np.mean(gap_residual[:, :, 0] * gap_residual[:, :, 1], axis=1),
                    np.mean(proxy_residual[:, :, 0] * proxy_residual[:, :, 1], axis=1),
                )
            )
            covariance_term = (per_root[:, 0] - per_root[:, 1] - per_root[:, 2]) / 2.0
            selection_query_noise_by_candidate = np.mean(
                (selection_observation - deployment[:, :, 0])
                * (selection_observation - deployment[:, :, 1]),
                axis=0,
            )
            proxy_query_noise_by_candidate = np.mean(
                (predicted_proxy - proxy[:, :, 0]) * (predicted_proxy - proxy[:, :, 1]),
                axis=0,
            )
            correction_query_noise_by_candidate = (
                selection_query_noise_by_candidate + proxy_query_noise_by_candidate
            )
            output[str(h)] = {
                "deployment_prediction": dep,
                "correction_gap_prediction": gap_result,
                "online_proxy_measurement": proxy_result,
                "direct_operational_query_noise": {
                    "selection_deployment_variance_by_candidate": selection_query_noise_by_candidate.tolist(),
                    "proxy_common_variance_by_candidate": proxy_query_noise_by_candidate.tolist(),
                    "correction_variance_for_independent_selection_minus_proxy_by_candidate": correction_query_noise_by_candidate.tolist(),
                    "mean_selection_deployment_variance": float(np.mean(selection_query_noise_by_candidate)),
                    "mean_proxy_common_variance": float(np.mean(proxy_query_noise_by_candidate)),
                    "mean_correction_variance": float(np.mean(correction_query_noise_by_candidate)),
                    "selection_minus_audit_mean_bias_by_candidate": np.mean(
                        selection_observation - deployment.mean(axis=2), axis=0
                    ).tolist(),
                    "estimator": "For selection S and independent audits A/B, mean[(S-A)(S-B)] estimates selection-block MSE. The same identity estimates proxy/common MSE. Their sum is correction-observation variance because selection and proxy blocks are independent.",
                    "warning": "These 30-root estimates are development-only and individual method-of-moments variances may be negative; freeze a documented nonnegative shrinkage rule before new seeds.",
                },
                "hierarchical_calibration_mean_diagnostic": {
                    "correction_gap_prediction": hierarchical_gap_result,
                    "deployment_prediction": hierarchical_deployment_result,
                    "candidate_latent_covariance_from_12_calibration_roots": candidate_covariance.tolist(),
                    "average_candidate_latent_variance": float(np.mean(np.diag(candidate_covariance))),
                    "formal_gap_world_variance_fraction_in_two_feature_subspace": projected_trace / total_trace if total_trace else 0.0,
                    "warning": "The feature-subspace fraction uses the 30-root cohort after outcomes and is diagnostic only.",
                },
                "operational_mse_identity": {
                    "deployment_mse": _bootstrap_scalar(per_root[:, 0:1], lambda x: np.mean(x)),
                    "correction_gap_mse": _bootstrap_scalar(per_root[:, 1:2], lambda x: np.mean(x)),
                    "online_proxy_mse": _bootstrap_scalar(per_root[:, 2:3], lambda x: np.mean(x)),
                    "twice_cross_covariance": _bootstrap_scalar(covariance_term[:, None], lambda x: 2.0 * np.mean(x)),
                    "max_absolute_row_identity_error": float(np.max(np.abs(identity_error))),
                    "formula": "deployment MSE = gap-correction MSE + online proxy MSE + 2*cross covariance",
                },
            }
    return output


def _calibration_random_coefficients(archive: Path):
    root = "melting_mechanism_20260917"
    endpoint_features = np.asarray([[-1.0, 1.0], [1.0, 1.0]])
    inverse = np.linalg.inv(endpoint_features)
    output = {}
    with tarfile.open(archive, "r:gz") as tar:
        for h in (4, 32):
            beta = np.empty((12, 2, 2))
            deployment = np.empty((12, 2, 2))
            proxy = np.empty((12, 2, 2))
            for r, seed in enumerate(range(43000, 43012)):
                rows = _read_json(tar, f"{root}/seed_{seed}/results.json")
                indexed = {(float(row["target_probability"]), int(row["training_episodes"]), row["eval_block"]): row for row in rows}
                for b, block in enumerate(("A", "B")):
                    old = indexed[0.5, h, block]
                    for c, p in enumerate((0.25, 0.75)):
                        row = indexed[p, h, block]
                        deployment[r, c, b] = float(row["focal_return"] - old["focal_return"])
                        proxy[r, c, b] = float(row["frozen_focal_return"] - old["frozen_focal_return"])
                    beta[r, :, b] = inverse @ (deployment[r, :, b] - proxy[r, :, b])

            beta_mean = beta.mean(axis=2)
            beta_difference = beta[:, :, 0] - beta[:, :, 1]
            omega_beta_block = beta_difference.T @ beta_difference / (2.0 * len(beta))
            omega_beta_mean = omega_beta_block / 2.0
            centered = beta_mean - beta_mean.mean(axis=0)
            between = centered.T @ centered / (len(beta) - 1)
            sigma_world = between - omega_beta_mean

            dep_difference = deployment[:, :, 0] - deployment[:, :, 1]
            proxy_difference = proxy[:, :, 0] - proxy[:, :, 1]
            gap_difference = dep_difference - proxy_difference
            omega_dep_16 = dep_difference.T @ dep_difference / (2.0 * len(beta))
            omega_proxy_16 = proxy_difference.T @ proxy_difference / (2.0 * len(beta))
            omega_gap_same_block_16 = gap_difference.T @ gap_difference / (2.0 * len(beta))
            actual_independent_correction_noise_8 = 2.0 * (omega_dep_16 + omega_proxy_16)
            incorrectly_scaled_same_block_gap_8 = 2.0 * omega_gap_same_block_16

            output[str(h)] = {
                "beta_population_mean": beta_mean.mean(axis=0).tolist(),
                "beta_ab_difference_mean": beta_difference.mean(axis=0).tolist(),
                "beta_measurement_covariance_per_16rep_block": omega_beta_block.tolist(),
                "beta_measurement_covariance_of_ab_mean": omega_beta_mean.tolist(),
                "beta_between_root_covariance_of_ab_mean": between.tolist(),
                "beta_cross_world_covariance": _psd_summary(sigma_world),
                "population_mean_estimation_covariance": (between / len(beta)).tolist(),
                "endpoint_query_noise": {
                    "deployment_noise_8rep_direct_likelihood": (2.0 * omega_dep_16).tolist(),
                    "proxy_noise_8rep": (2.0 * omega_proxy_16).tolist(),
                    "correction_noise_8rep_independent_selection_and_proxy": actual_independent_correction_noise_8.tolist(),
                    "same_block_gap_noise_scaled_16_to_8_incorrect_for_current_query": incorrectly_scaled_same_block_gap_8.tolist(),
                    "difference_due_to_crn_covariance": (incorrectly_scaled_same_block_gap_8 - actual_independent_correction_noise_8).tolist(),
                },
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-archive", type=Path, required=True)
    parser.add_argument("--mechanism-archive", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "scope": "Post-result development audit. No new simulation and no changes to frozen decisions. The 44000--44029 cohort cannot confirm a revised posterior after this audit.",
        "identifiability": {
            "identified": [
                "evaluation noise conditional on a frozen response world, from independent A/B blocks",
                "systematic candidate-grid bias of the frozen predictor, under exchangeable root worlds",
                "remaining between-root latent predictive variation after subtracting A/B mean noise",
            ],
            "not_identified": [
                "pure response-optimizer randomness separated from all other root-seed world variation",
                "off-grid nonlinear discrepancy from the two calibration endpoints alone",
            ],
        },
        "calibration_random_coefficients": None,
        "formal_development_decomposition": None,
    }
    calibration = _calibration_random_coefficients(args.mechanism_archive)
    result["calibration_random_coefficients"] = calibration
    result["formal_development_decomposition"] = _formal_data(
        args.formal_archive, args.predictions, calibration
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
