#!/usr/bin/env python3
"""Numerical checks for v3b/v3c noisy conditioning and exact KG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_b4_redesign import exact_gaussian_kg_score, exact_kg_next
    from experiments.revision.highway_calibration import CalibrationModel, GaussianPosterior
    from run_multiround_loop_v3b import condition_noisy_observation

    rng = np.random.default_rng(20261110)
    mean = np.asarray([0.2, -0.1, 0.35, 0.05], dtype=float)
    raw = rng.normal(size=(4, 4))
    covariance = raw @ raw.T
    covariance += np.eye(4) * 0.4
    observation_variance = np.asarray([0.03, 0.07, 0.02, 0.05], dtype=float)
    rows = [
        {"candidate_id": f"c{i}", "candidate_index": i, "delta_proxy": 0.0, "highway_action": f"slot{i}"}
        for i in range(4)
    ]
    model = CalibrationModel(
        candidate_ids=tuple(f"c{i}" for i in range(4)),
        actions=tuple(f"slot{i}" for i in range(4)),
        correction_mean=tuple(float(v) for v in mean),
        correction_covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        observation_variance=tuple(float(v) for v in observation_variance),
        calibration_seed_count=8,
        covariance_shrinkage=0.0,
        observation_noise="synthetic",
    )
    values = np.asarray([0.41, -0.22], dtype=float)
    queried = [0, 2]
    sequential = GaussianPosterior(rows, model)
    for index, value in zip(queried, values, strict=True):
        condition_noisy_observation(sequential, f"c{index}", float(value))
    observation_covariance = covariance[np.ix_(queried, queried)] + np.diag(observation_variance[queried])
    cross = covariance[:, queried]
    batch_mean = mean + cross @ np.linalg.solve(observation_covariance, values - mean[queried])
    batch_covariance = covariance - cross @ np.linalg.solve(observation_covariance, cross.T)
    batch_covariance = (batch_covariance + batch_covariance.T) / 2
    checks = {
        "sequential_batch_mean_agree": bool(np.allclose(sequential.means, batch_mean, rtol=1e-10, atol=1e-10)),
        "sequential_batch_covariance_agree": bool(np.allclose(sequential.covariance, batch_covariance, rtol=1e-10, atol=1e-10)),
        "observed_latent_variances_positive": bool(all(sequential.covariance[index, index] > 0.0 for index in queried)),
        "unqueried_variances_positive": bool(all(sequential.covariance[index, index] > 0.0 for index in {1, 3})),
    }
    # The exact KG score is checked against a common-normal Monte Carlo estimate
    # using the same one-dimensional predictive fantasy construction.
    posterior = GaussianPosterior(rows, model)
    candidate, exact_score = exact_kg_next(posterior)
    candidate_index = posterior._index[candidate]
    current_value = max([0.0, *posterior.means.tolist()])
    covariance_column = posterior.covariance[:, candidate_index]
    predictive_sd = float(np.sqrt(posterior.covariance[candidate_index, candidate_index] + posterior.observation_variance[candidate_index]))
    normals = rng.standard_normal(200000)
    fantasy_values = posterior.means[candidate_index] + predictive_sd * normals
    affine_values = np.maximum(0.0, np.max(posterior.means[:, None] + (covariance_column / max(predictive_sd, 1e-15))[:, None] * normals[None, :], axis=0))
    monte_carlo_score = max(0.0, float(affine_values.mean() - current_value))
    checks["exact_kg_matches_monte_carlo"] = bool(abs(float(exact_score) - monte_carlo_score) < 0.01)
    checks["exact_kg_nonnegative"] = bool(float(exact_score) >= 0.0)
    checks["all_pass"] = all(checks.values())
    report = {
        "check_id": "V3B-NUMERICAL-2026-11-10",
        "source": str(source),
        "checks": checks,
        "conditioning": {
            "queried_indices": queried,
            "observation_variance": observation_variance.tolist(),
            "max_mean_abs_error": float(np.max(np.abs(sequential.means - batch_mean))),
            "max_covariance_abs_error": float(np.max(np.abs(sequential.covariance - batch_covariance))),
        },
        "kg": {
            "selected_candidate": candidate,
            "exact_score": float(exact_score),
            "monte_carlo_score": monte_carlo_score,
            "monte_carlo_draws": int(len(normals)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
