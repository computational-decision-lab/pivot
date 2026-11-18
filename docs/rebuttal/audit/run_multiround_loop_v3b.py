#!/usr/bin/env python3
"""Run the noisy-observation corrected multi-round study (v3b)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from run_multiround_loop import (
    ResponseWorld,
    boot,
    deployment_delta,
    make_row,
    propose,
    sha256_bytes,
    stable_seed,
)


RESPONSE_REGIMES = {"low": 0.35, "medium": 0.60, "high": 0.85}
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_vector(
    row: dict[str, Any],
    *,
    pressure: float,
    response_strength: float,
    incumbent: dict[str, float],
    round_id: int,
    rounds: int,
) -> np.ndarray:
    candidate = row["candidate_parameters"]
    delta_intensity = float(candidate["intensity"] - incumbent["intensity"])
    delta_bias = float(candidate["bias"] - incumbent["bias"])
    return np.asarray([
        pressure,
        response_strength,
        incumbent["intensity"],
        incumbent["bias"],
        delta_intensity,
        delta_bias,
        float(row["delta_proxy"]),
        pressure * delta_intensity,
        response_strength * delta_intensity,
        round_id / max(1, rounds - 1),
    ], dtype=float)


@dataclass
class ContextualCorrectionModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    beta_mean: np.ndarray
    beta_covariance: np.ndarray
    observation_variance: np.ndarray
    residual_variance: float
    calibration_roots: int
    ridge: float

    def clone(self) -> ContextualCorrectionModel:
        return ContextualCorrectionModel(
            self.feature_mean.copy(), self.feature_scale.copy(),
            self.beta_mean.copy(), self.beta_covariance.copy(),
            self.observation_variance.copy(), float(self.residual_variance), self.calibration_roots,
            float(self.ridge),
        )

    def design(self, raw: np.ndarray) -> np.ndarray:
        standardized = (raw - self.feature_mean) / self.feature_scale
        return np.column_stack([np.ones(len(raw)), standardized])

    def predict(
        self,
        rows: list[dict[str, Any]],
        *,
        pressure: float,
        response_strength: float,
        incumbent: dict[str, float],
        round_id: int,
        rounds: int,
        calibration_model_type: Any,
    ) -> Any:
        from experiments.revision.highway_calibration import GaussianPosterior

        ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
        raw = np.vstack([
            feature_vector(
                row, pressure=pressure, response_strength=response_strength,
                incumbent=incumbent, round_id=round_id, rounds=rounds,
            )
            for row in ordered
        ])
        design = self.design(raw)
        correction_mean = design @ self.beta_mean
        correction_covariance = design @ self.beta_covariance @ design.T
        correction_covariance = (correction_covariance + correction_covariance.T) / 2
        correction_covariance += np.eye(len(ordered)) * self.residual_variance
        response_index = min(
            range(len(RESPONSE_REGIMES)),
            key=lambda i: abs(list(RESPONSE_REGIMES.values())[i] - response_strength),
        )
        measurement_variances = tuple(
            float(self.observation_variance[response_index, int(row["candidate_index"])])
            for row in ordered
        )
        model = calibration_model_type(
            candidate_ids=tuple(str(row["candidate_id"]) for row in ordered),
            actions=tuple(str(row.get("highway_action", "")) for row in ordered),
            correction_mean=tuple(float(value) for value in correction_mean),
            correction_covariance=tuple(tuple(float(value) for value in r) for r in correction_covariance),
            observation_variance=measurement_variances,
            calibration_seed_count=self.calibration_roots,
            covariance_shrinkage=0.0,
            observation_noise="repeated_calibration_rollout_variance_by_response_and_candidate",
        )
        return GaussianPosterior(ordered, model)

    def observe(
        self,
        row: dict[str, Any],
        correction: float,
        *,
        pressure: float,
        response_strength: float,
        incumbent: dict[str, float],
        round_id: int,
        rounds: int,
    ) -> dict[str, float]:
        raw = feature_vector(
            row, pressure=pressure, response_strength=response_strength,
            incumbent=incumbent, round_id=round_id, rounds=rounds,
        )
        x = self.design(raw.reshape(1, -1)).reshape(-1)
        px = self.beta_covariance @ x
        response_index = min(
            range(len(RESPONSE_REGIMES)),
            key=lambda i: abs(list(RESPONSE_REGIMES.values())[i] - response_strength),
        )
        measurement_variance = float(self.observation_variance[response_index, int(row["candidate_index"])])
        predictive_variance = float(x @ px + self.residual_variance + measurement_variance)
        residual = float(correction - x @ self.beta_mean)
        z_score = residual / np.sqrt(max(predictive_variance, 1e-12))
        gain = px / max(predictive_variance, 1e-12)
        self.beta_mean = self.beta_mean + gain * residual
        self.beta_covariance = self.beta_covariance - np.outer(gain, x @ self.beta_covariance)
        self.beta_covariance = (self.beta_covariance + self.beta_covariance.T) / 2
        eig_min = float(np.linalg.eigvalsh(self.beta_covariance).min())
        if eig_min < 1e-12:
            self.beta_covariance += np.eye(len(self.beta_covariance)) * (1e-12 - eig_min)
        return {"correction": float(correction), "predicted_correction": float(x @ (self.beta_mean - gain * residual)), "residual": residual, "standardized_residual": float(z_score)}

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_names": ["intercept", *FEATURE_NAMES],
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "beta_mean": self.beta_mean.tolist(),
            "beta_covariance": self.beta_covariance.tolist(),
            "observation_variance_regimes": list(RESPONSE_REGIMES.values()),
            "candidate_slots": [0, 1, 2, 3],
            "single_rollout_observation_variance": self.observation_variance.tolist(),
            "residual_variance": self.residual_variance,
            "calibration_roots": self.calibration_roots,
            "ridge": self.ridge,
        }


def fit_contextual_model(
    samples: list[dict[str, Any]], *, ridge: float = 0.5,
    observation_variance: np.ndarray | None = None,
    residual_variance: float | None = None,
) -> ContextualCorrectionModel:
    raw = np.vstack([np.asarray(row["features"], dtype=float) for row in samples])
    y = np.asarray([float(row["correction"]) for row in samples], dtype=float)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 1e-8
    precision = design.T @ design + penalty
    covariance_unscaled = np.linalg.inv(precision)
    beta = covariance_unscaled @ design.T @ y
    residual = y - design @ beta
    training_variance = float(np.sum(residual ** 2) / max(1, len(y) - design.shape[1]))
    if observation_variance is None:
        observation_variance = estimate_observation_variance(samples)
    mean_label_noise = float(np.mean([row["training_noise_variance"] for row in samples]))
    model_variance = max(
        float(residual_variance if residual_variance is not None else training_variance - mean_label_noise),
        1e-6,
    )
    beta_covariance = covariance_unscaled * model_variance
    return ContextualCorrectionModel(
        feature_mean=mean,
        feature_scale=scale,
        beta_mean=beta,
        beta_covariance=beta_covariance,
        observation_variance=np.asarray(observation_variance, dtype=float),
        residual_variance=model_variance,
        calibration_roots=len({int(row["root_id"]) for row in samples}),
        ridge=ridge,
    )


def estimate_observation_variance(samples: list[dict[str, Any]]) -> np.ndarray:
    strengths = list(RESPONSE_REGIMES.values())
    estimates = np.zeros((len(strengths), 4), dtype=float)
    for regime_index, strength in enumerate(strengths):
        for candidate_index in range(4):
            values = [
                float(row["single_rollout_variance"])
                for row in samples
                if int(row["candidate_index"]) == candidate_index
                and np.isclose(float(row["response_strength"]), strength)
            ]
            if not values:
                raise ValueError(f"no calibration noise samples for regime={strength}, candidate={candidate_index}")
            estimates[regime_index, candidate_index] = max(float(np.mean(values)), 1e-8)
    return estimates


def make_calibration_samples(
    source: Path,
    *,
    calibration_roots: int,
    rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    sys.path[:0] = [str(source), str(source / "src")]
    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    root_context: dict[int, dict[str, Any]] = {}
    regime_values = list(RESPONSE_REGIMES.values())
    calibration_repeats = 8
    for offset in range(calibration_roots):
        root = 510000 + offset
        regime = regime_values[offset % len(regime_values)]
        within_regime = offset // len(regime_values)
        count_per_regime = int(np.ceil(calibration_roots / len(regime_values)))
        pressure = 0.05 + 0.65 * ((within_regime + 0.5) / count_per_regime)
        rng = np.random.default_rng(stable_seed("loop-v3b-calibration-context", root))
        incumbent = {
            "intensity": float(rng.uniform(-0.30, 0.35)),
            "bias": float(rng.uniform(-0.12, 0.12)),
        }
        world = ResponseWorld(pressure=pressure, response_strength=regime)
        candidates = propose(incumbent, root=root, round_id=0)
        rows = [make_row(world, incumbent, candidate, root=root, round_id=0, index=i) for i, candidate in enumerate(candidates)]
        root_context[root] = {"pressure": pressure, "response_strength": regime, "incumbent": incumbent}
        truth: dict[str, float] = {}
        for i, row in enumerate(rows):
            rollout_values = [
                deployment_delta(
                    world, incumbent, candidates[i],
                seed=stable_seed("loop-v3-calibration-hf", root, i, replicate),
                )
                for replicate in range(calibration_repeats)
            ]
            hf_delta = float(np.mean(rollout_values))
            single_rollout_variance = float(np.var(rollout_values, ddof=1))
            correction = hf_delta - float(row["delta_proxy"])
            raw_features = feature_vector(
                row, pressure=pressure, response_strength=regime,
                incumbent=incumbent, round_id=0, rounds=rounds,
            )
            rollout_seeds = [stable_seed("loop-v3-calibration-hf", root, i, replicate) for replicate in range(calibration_repeats)]
            samples.append({
                "root_id": root, "candidate_id": row["candidate_id"],
                "candidate_index": i, "response_strength": regime,
                "features": raw_features.tolist(), "correction": correction,
                "hf_delta": hf_delta, "proxy_delta": float(row["delta_proxy"]),
                "rollout_seeds": rollout_seeds,
                "rollout_values": rollout_values,
                "single_rollout_variance": single_rollout_variance,
                "training_noise_variance": single_rollout_variance / calibration_repeats,
            })
            truth[row["candidate_id"]] = hf_delta
        records.append({
            "root_id": root,
            "pressure": pressure,
            "response_strength": regime,
            "incumbent": incumbent,
            "proxy_rows": rows,
            "truth": truth,
        })
    return samples, records, root_context


def calibration_cv(samples: list[dict[str, Any]], *, folds: int = 5, ridge: float = 0.5) -> dict[str, Any]:
    root_ids = sorted({int(row["root_id"]) for row in samples})
    heldout: list[dict[str, Any]] = []
    for fold in range(folds):
        validation_roots = {root for i, root in enumerate(root_ids) if i % folds == fold}
        train = [row for row in samples if int(row["root_id"]) not in validation_roots]
        test = [row for row in samples if int(row["root_id"]) in validation_roots]
        model = fit_contextual_model(train, ridge=ridge)
        for row in test:
            raw = np.asarray(row["features"], dtype=float)
            prediction = float((model.design(raw.reshape(1, -1)) @ model.beta_mean).reshape(-1)[0])
            heldout.append({
                "root_id": int(row["root_id"]),
                "pressure": float(raw[0]),
                "response_strength": float(raw[1]),
                "error": prediction - float(row["correction"]),
                "training_noise_variance": float(row["training_noise_variance"]),
            })
    errors = np.asarray([row["error"] for row in heldout])
    pressure_bins: dict[str, Any] = {}
    for name, predicate in (
        ("low_pressure_le_0_25", lambda value: value <= 0.25),
        ("mid_pressure_0_25_to_0_50", lambda value: 0.25 < value <= 0.50),
        ("high_pressure_gt_0_50", lambda value: value > 0.50),
    ):
        subset = np.asarray([row["error"] for row in heldout if predicate(row["pressure"])])
        if len(subset):
            pressure_bins[name] = {
                "n_observations": int(len(subset)),
                "rmse": float(np.sqrt(np.mean(subset ** 2))),
                "mae": float(np.mean(np.abs(subset))),
                "bias": float(np.mean(subset)),
            }
    by_response: dict[str, Any] = {}
    for strength in sorted({row["response_strength"] for row in heldout}):
        subset = np.asarray([row["error"] for row in heldout if row["response_strength"] == strength])
        by_response[str(strength)] = {
            "n_observations": int(len(subset)),
            "rmse": float(np.sqrt(np.mean(subset ** 2))),
            "mae": float(np.mean(np.abs(subset))),
            "bias": float(np.mean(subset)),
        }
    return {
        "folds": folds,
        "roots": len(root_ids),
        "candidate_observations": len(heldout),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "by_pressure_bin": pressure_bins,
        "by_response_strength": by_response,
        "observation_variance": estimate_observation_variance(samples).tolist(),
        "label_noise_variance": float(np.mean([row["training_noise_variance"] for row in samples])),
        "model_residual_variance": max(
            float(np.mean(errors ** 2) - np.mean([row["training_noise_variance"] for row in heldout])),
            1e-6,
        ),
    }


def fit_static_model(source: Path, *, roots: int = 40) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    from run_multiround_loop import fit_model

    return fit_model(roots, source, {})


def mark_known_observations(posterior: Any, observations: list[tuple[str, float]]) -> None:
    for candidate_id, value in observations:
        condition_noisy_observation(posterior, candidate_id, value)


def condition_noisy_observation(posterior: Any, candidate_id: str, value: float) -> None:
    """Condition a latent candidate delta on a noisy HF observation.

    The source GaussianPosterior treats observations as exact and clears the
    observed arm.  v3b keeps the measurement variance supplied by repeated
    calibration rollouts, so the observed latent value retains uncertainty.
    """
    if candidate_id in posterior.observed:
        raise ValueError(f"candidate already observed: {candidate_id}")
    index = posterior._index[candidate_id]
    covariance_row = posterior.covariance[index, :].copy()
    denominator = float(posterior.covariance[index, index] + posterior.observation_variance[index])
    if denominator <= 0.0:
        raise ValueError("non-positive noisy-observation denominator")
    gain = covariance_row / denominator
    posterior.means = posterior.means + gain * (float(value) - posterior.means[index])
    posterior.covariance = posterior.covariance - np.outer(gain, covariance_row)
    posterior.covariance = (posterior.covariance + posterior.covariance.T) / 2
    eig_min = float(np.linalg.eigvalsh(posterior.covariance).min())
    if eig_min < 1e-12:
        posterior.covariance = posterior.covariance + np.eye(len(posterior.means)) * (1e-12 - eig_min)
    posterior.observed[candidate_id] = float(value)


def select_query(
    method: str,
    posterior: Any,
    rows: list[dict[str, Any]],
    already_queried: set[str],
) -> tuple[str, float | None]:
    from experiments.revision.highway_b4_redesign import exact_kg_next

    available = [candidate for candidate in posterior.candidate_ids if candidate not in already_queried]
    if not available:
        raise ValueError("no candidates remain for method-specific query")
    if method in {"PIVOT-KG-Loop-v3b", "Contextual-KG"}:
        candidate_id, score = exact_kg_next(posterior)
        if candidate_id in already_queried:
            candidate_id = max(
                available,
                key=lambda cid: (posterior.value(cid), -posterior._index[cid]),
            )
            score = None
        return candidate_id, float(score) if score is not None else None
    if method == "IVR-Online":
        scores: dict[str, float] = {}
        for candidate_id in available:
            index = posterior._index[candidate_id]
            denom = float(posterior.covariance[index, index] + posterior.observation_variance[index])
            scores[candidate_id] = float(np.dot(posterior.covariance[:, index], posterior.covariance[:, index]) / denom) if denom > 0 else 0.0
        candidate_id = max(available, key=lambda cid: (scores[cid], -posterior._index[cid]))
        return candidate_id, scores[candidate_id]
    if method == "Top-Proxy-Online":
        by_id = {str(row["candidate_id"]): row for row in rows}
        candidate_id = max(
            available,
            key=lambda cid: (float(by_id[cid]["delta_proxy"]), -posterior._index[cid]),
        )
        return candidate_id, None
    raise ValueError(method)


def run(
    *, source: Path, output: Path, split: str, roots_per_regime: int,
    calibration_roots: int, rounds: int, bootstrap_draws: int, ridge: float,
) -> None:
    if split not in {"development", "confirmation"}:
        raise ValueError("split must be development or confirmation")
    if output.exists():
        raise SystemExit(f"output exists; choose a new directory: {output}")
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import CalibrationModel, GaussianPosterior

    samples, calibration_records, _ = make_calibration_samples(
        source, calibration_roots=calibration_roots, rounds=rounds,
    )
    cv = calibration_cv(samples, folds=5, ridge=ridge)
    model = fit_contextual_model(
        samples, ridge=ridge,
        observation_variance=np.asarray(cv["observation_variance"], dtype=float),
        residual_variance=float(cv["model_residual_variance"]),
    )
    static_model, static_record, _ = fit_static_model(source)
    model_record = model.to_record()
    contextual_hash = sha256_bytes(json.dumps(model_record, sort_keys=True).encode())
    static_hash = sha256_bytes(json.dumps(static_record, sort_keys=True).encode())
    root_start = 830000 if split == "development" else 930000
    methods = [
        "Static-PIVOT-KG-v1",
        "Contextual-KG",
        "PIVOT-KG-Loop-v3b",
        "IVR-Online",
        "Top-Proxy-Online",
    ]
    trajectories: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []

    for regime_name, response_strength in RESPONSE_REGIMES.items():
        for root_offset in range(roots_per_regime):
            root = root_start + list(RESPONSE_REGIMES).index(regime_name) * 10000 + root_offset
            initial_pressure = 0.10 + 0.015 * (root_offset % 7)
            for method in methods:
                world = ResponseWorld(pressure=initial_pressure, response_strength=response_strength)
                incumbent = {"intensity": 0.18, "bias": 0.0}
                online_model = model.clone()
                cumulative_isr = 0.0
                cumulative_utility = 0.0
                cumulative_deployment_return = 0.0
                harmful = 0
                retained = 0
                for round_id in range(rounds):
                    pressure_before = float(world.pressure)
                    candidates = propose(incumbent, root=root, round_id=round_id)
                    rows = [make_row(world, incumbent, candidate, root=root, round_id=round_id, index=i) for i, candidate in enumerate(candidates)]
                    query_details: list[dict[str, Any]] = []
                    queried: list[tuple[str, float]] = []

                    if method == "Static-PIVOT-KG-v1":
                        posterior = GaussianPosterior(rows, static_model)
                        from experiments.revision.highway_b4_redesign import exact_kg_next
                        for query_order in range(2):
                            cid, acquisition = exact_kg_next(posterior)
                            index = posterior._index[cid]
                            query_seed = stable_seed("loop-v3-hf", root, round_id, cid)
                            value = deployment_delta(world, incumbent, rows[index]["candidate_parameters"], seed=query_seed)
                            condition_noisy_observation(posterior, cid, value)
                            queried.append((cid, value))
                            query_details.append({"candidate_id": cid, "query_order": query_order, "acquisition": acquisition, "query_seed": query_seed, "observation": value, "posterior_latent_variance": float(posterior.covariance[index, index]), "observation_variance": float(posterior.observation_variance[index])})
                    else:
                        if method == "Contextual-KG":
                            # This ablation has contextual calibration but no cross-round updates.
                            current_model = model.clone()
                        else:
                            current_model = online_model
                        posterior = current_model.predict(
                            rows, pressure=pressure_before, response_strength=response_strength,
                            incumbent=incumbent, round_id=round_id, rounds=rounds,
                            calibration_model_type=CalibrationModel,
                        )
                        online = method != "Contextual-KG"
                        budget_query_count = 2
                        if online:
                            sentinel_index = round_id % len(rows)
                            sentinel = rows[sentinel_index]
                            sentinel_seed = stable_seed("loop-v3-hf", root, round_id, sentinel["candidate_id"])
                            sentinel_value = deployment_delta(world, incumbent, sentinel["candidate_parameters"], seed=sentinel_seed)
                            corr = sentinel_value - float(sentinel["delta_proxy"])
                            update = current_model.observe(
                                sentinel, corr, pressure=pressure_before,
                                response_strength=response_strength, incumbent=incumbent,
                                round_id=round_id, rounds=rounds,
                            )
                            queried.append((str(sentinel["candidate_id"]), sentinel_value))
                            condition_noisy_observation(posterior, str(sentinel["candidate_id"]), sentinel_value)
                            sentinel_position = posterior._index[str(sentinel["candidate_id"])]
                            query_details.append({"candidate_id": sentinel["candidate_id"], "query_order": 0, "kind": "common_sentinel", "sentinel_rule": "candidate_index = round_id mod 4", "query_seed": sentinel_seed, "observation": sentinel_value, "posterior_latent_variance": float(posterior.covariance[sentinel_position, sentinel_position]), "observation_variance": float(posterior.observation_variance[sentinel_position]), **update})
                        else:
                            budget_query_count = 2

                        while len(queried) < budget_query_count:
                            cid, acquisition = select_query(method, posterior, rows, {item[0] for item in queried})
                            index = posterior._index[cid]
                            query_seed = stable_seed("loop-v3-hf", root, round_id, cid)
                            value = deployment_delta(world, incumbent, rows[index]["candidate_parameters"], seed=query_seed)
                            corr = value - float(rows[index]["delta_proxy"])
                            if method != "Contextual-KG":
                                update = current_model.observe(
                                    rows[index], corr, pressure=pressure_before,
                                    response_strength=response_strength, incumbent=incumbent,
                                    round_id=round_id, rounds=rounds,
                                )
                            else:
                                update = {"correction": corr, "online_update": False}
                            condition_noisy_observation(posterior, cid, value)
                            queried.append((cid, value))
                            query_details.append({"candidate_id": cid, "query_order": len(queried) - 1, "kind": "method_specific", "acquisition": acquisition, "query_seed": query_seed, "observation": value, "posterior_latent_variance": float(posterior.covariance[index, index]), "observation_variance": float(posterior.observation_variance[index]), **update})
                            # Keep the within-round latent posterior conditioned on the
                            # observation.  The coefficient model is updated separately
                            # for the next round, avoiding double counting this datum.

                    selected = posterior.select()
                    eval_seed = stable_seed("loop-v3-independent-eval", root, round_id)
                    eval_deltas = [
                        deployment_delta(world, incumbent, candidate, seed=eval_seed + i)
                        for i, candidate in enumerate(candidates)
                    ]
                    best = max([0.0, *eval_deltas])
                    selected_delta = 0.0 if selected == "incumbent" else eval_deltas[int(selected[1:])]
                    isr = float(best - selected_delta)
                    retained_flag = selected == "incumbent"
                    harmful_flag = selected_delta < 0.0
                    chosen_policy = incumbent if retained_flag else candidates[int(selected[1:])]
                    state_before = world.state_hash()
                    deploy_seed = stable_seed("loop-v3-deployment-response", root, round_id)
                    deployment_return = world.value(chosen_policy, seed=deploy_seed, deployment=True)
                    world.deploy(chosen_policy, seed=deploy_seed)
                    state_after = world.state_hash()
                    ledger.append({
                        "split": split, "regime": regime_name, "response_strength": response_strength,
                        "root_id": root, "method": method, "round_id": round_id,
                        "pressure_before": pressure_before, "pressure_after": float(world.pressure),
                        "candidate_panel": rows, "proposer_seed": stable_seed("proposer", root, round_id),
                        "evaluation_seed": eval_seed, "deployment_seed": deploy_seed,
                        "calibration_hash": static_hash if method == "Static-PIVOT-KG-v1" else contextual_hash,
                        "sentinel_rule": "candidate_index = round_id mod 4" if method in {"PIVOT-KG-Loop-v3b", "IVR-Online", "Top-Proxy-Online"} else None,
                        "query_ids": [cid for cid, _ in queried], "query_count": len(queried),
                        "query_details": query_details, "selected_policy": selected,
                        "terminal_selection_rule": "posterior mean over incumbent plus four candidates",
                        "retained_incumbent": retained_flag,
                        "deployment_outcome": {"independent_deltas": eval_deltas, "selected_delta": selected_delta, "best_delta": best, "ISR": isr, "deployment_return": float(deployment_return)},
                        "initial_state_hash": state_before, "next_state_hash": state_after,
                        "next_pressure": float(world.pressure),
                    })
                    cumulative_isr += isr
                    cumulative_utility += selected_delta
                    cumulative_deployment_return += float(deployment_return)
                    harmful += int(harmful_flag)
                    retained += int(retained_flag)
                    incumbent = {"intensity": float(chosen_policy["intensity"]), "bias": float(chosen_policy["bias"])}
                trajectories.append({
                    "split": split, "regime": regime_name, "root_id": root, "method": method,
                    "CISR": cumulative_isr, "cumulative_utility": cumulative_utility,
                    "cumulative_deployment_return": cumulative_deployment_return,
                    "harmful_adoption_rate": harmful / rounds, "retention_rate": retained / rounds,
                    "rounds": rounds,
                })

    output.mkdir(parents=True)
    ledger_path = output / "trajectory_ledger.json"
    ledger_blob = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    ledger_path.write_bytes(ledger_blob)
    (output / "trajectory_ledger.sha256").write_text(sha256_bytes(ledger_blob) + "\n")
    (output / "trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n")
    (output / "calibration_model.json").write_text(json.dumps(model_record, indent=2, sort_keys=True) + "\n")
    (output / "static_calibration_model.json").write_text(json.dumps(static_record, indent=2, sort_keys=True) + "\n")
    (output / "calibration_rows.json").write_text(json.dumps(calibration_records, indent=2, sort_keys=True) + "\n")
    (output / "calibration_training_samples.json").write_text(json.dumps(samples, indent=2, sort_keys=True) + "\n")
    protocol = {
        "protocol_id": "P1-LOOP-V3B-2026-11-10",
        "split": split,
        "roots_per_regime": roots_per_regime,
        "root_start": root_start,
        "calibration_root_start": 510000,
        "rounds": rounds,
        "candidate_count": 4,
        "calibration_roots": calibration_roots,
        "calibration_repeats_per_candidate": 8,
        "hf_budget_per_round": 2,
        "online_methods": ["PIVOT-KG-Loop-v3b", "IVR-Online", "Top-Proxy-Online"],
        "sentinel_rule": "candidate_index = round_id mod 4; one common-rule sentinel plus one method-specific query",
        "response_regimes": RESPONSE_REGIMES,
        "response_regime_order": list(RESPONSE_REGIMES),
        "initial_test_pressure": "0.10 + 0.015 * (root_offset mod 7)",
        "shared_random_streams": {"evaluation": "root, round, candidate index", "deployment": "root, round", "queries": "root, round, candidate id"},
        "terminal_rule": "posterior mean over incumbent plus four candidates",
        "acquisition": {"PIVOT-KG-Loop-v3b": "exact one-step Gaussian KG", "IVR-Online": "integrated variance reduction", "Top-Proxy-Online": "descending proxy delta"},
        "method_definitions": {
            "PIVOT-KG-Loop-v3b": "common sentinel plus one KG-selected query; correction regression updates persist across rounds",
            "IVR-Online": "common sentinel plus one IVR-selected query; correction regression updates persist across rounds",
            "Top-Proxy-Online": "common sentinel plus one proxy-ranked query; correction regression updates persist across rounds",
            "Contextual-KG": "two KG-selected queries; posterior conditions within each round, but correction regression resets to the calibration fit at each round",
            "Static-PIVOT-KG-v1": "two KG-selected queries under the original narrow-state static calibration prior; legacy reference",
        },
        "contextual_model": {"type": "Bayesian linear correction model", "within_round_update": "noisy Gaussian conditioning with positive observation variance; coefficient update is kept separate", "cross_round_update": "Gaussian rank-one correction-regression update for the three online methods", "observation_variance": "mean within-context variance over eight independent calibration rollouts, by response strength and candidate slot", "model_residual_variance": "five-fold held-out MSE minus variance of the calibration mean, floored at 1e-6", "drift_diagnostic": "standardized residuals logged; no test-tuned threshold"},
        "physical_simulator_called": False,
    }
    checks = {
        "complete_method_root_round_panel": len(ledger) == roots_per_regime * len(RESPONSE_REGIMES) * len(methods) * rounds,
        "two_queries_each_online_round": all(row["query_count"] == 2 for row in ledger if row["method"] in {"PIVOT-KG-Loop-v3b", "IVR-Online", "Top-Proxy-Online"}),
        "unique_query_ids_within_round": all(len(row["query_ids"]) == len(set(row["query_ids"])) for row in ledger),
        "common_deployment_seed_across_methods": len({(row["root_id"], row["round_id"], row["deployment_seed"]) for row in ledger}) == roots_per_regime * len(RESPONSE_REGIMES) * rounds,
        "terminal_rule_includes_incumbent": all("incumbent" in row["terminal_selection_rule"] for row in ledger),
        "state_transitions_recorded": all(row["initial_state_hash"] != row["next_state_hash"] for row in ledger),
    }
    summary: dict[str, Any] = {
        "analysis_kind": "contextual_online_multiround_selector_comparison",
        "protocol": protocol,
        "protocol_sha256": sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "analyzer": sha256_file(Path(__file__).with_name("analyze_multiround_loop_v3b.py")),
            "preregistration": sha256_file(Path(__file__).with_name("P1_LOOP_V3B_PROTOCOL_2026-11-10.md")),
            "v1_runner": sha256_file(Path(__file__).with_name("run_multiround_loop.py")),
            "calibration_source": sha256_file(source / "experiments/revision/highway_calibration.py"),
            "kg_source": sha256_file(source / "experiments/revision/highway_b4_redesign.py"),
        },
        "calibration": {"contextual_model_hash": contextual_hash, "static_model_hash": static_hash, "five_fold_cv": cv},
        "methods": methods,
        "method_summary": {},
        "audit_checks": checks,
        "ledger_sha256": sha256_bytes(ledger_blob),
    }
    for regime in RESPONSE_REGIMES:
        summary["method_summary"][regime] = {}
        regime_roots = sorted({int(row["root_id"]) for row in trajectories if row["regime"] == regime})
        for method in methods:
            subset = [row for row in trajectories if row["regime"] == regime and row["method"] == method]
            summary["method_summary"][regime][method] = {
                metric: boot(
                    [float(row[metric]) for row in subset],
                    seed=stable_seed("loop-v3-summary", split, regime, method, metric),
                    draws=bootstrap_draws,
                )
                for metric in ("CISR", "cumulative_utility", "harmful_adoption_rate", "retention_rate")
            }
        summary["method_summary"][regime]["n_roots"] = len(regime_roots)
    summary["audit_checks"]["all_pass"] = all(checks.values())
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--roots-per-regime", type=int, default=50)
    parser.add_argument("--calibration-roots", type=int, default=240)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=0.5)
    args = parser.parse_args()
    run(
        source=args.source.resolve(), output=args.output.resolve(), split=args.split,
        roots_per_regime=args.roots_per_regime, calibration_roots=args.calibration_roots,
        rounds=args.rounds, bootstrap_draws=args.bootstrap_draws, ridge=args.ridge,
    )
    print((args.output / "summary.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
