from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schema import numeric_features
from .statistics import sign, spearman

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float


def global_features(row: Mapping[str, Any], role: str) -> FloatArray:
    parameters = row.get(f"{role}_parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    return np.asarray(
        [
            float(parameters.get("intensity", 0.0)),
            float(parameters.get("bias", 0.0)),
            float(row.get("response_strength", 0.0) or 0.0),
            float(row.get("operator_shift", 0.0) or 0.0),
        ],
        dtype=float,
    )


class LinearRegressor:
    """Dependency-free Bayesian-ridge-like linear predictor."""

    def __init__(self, alpha: float = 1.0, noise_variance: float = 1.0) -> None:
        if alpha <= 0 or noise_variance <= 0:
            raise ValueError("alpha and noise_variance must be positive")
        self.alpha = float(alpha)
        self.noise_variance = float(noise_variance)
        self.coef: FloatArray | None = None
        self.intercept = 0.0
        self.residual_std = 1.0
        self.covariance: FloatArray | None = None

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> LinearRegressor:
        matrix = np.asarray(features, dtype=float)
        target = np.asarray(targets, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != target.shape[0]:
            raise ValueError("features and targets must have equal non-empty rows")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
            raise ValueError("training values must be finite")
        design = np.column_stack([np.ones(matrix.shape[0]), matrix])
        regularizer = self.alpha * np.eye(design.shape[1])
        regularizer[0, 0] = self.alpha * 0.01
        precision = design.T @ design + regularizer
        self.covariance = np.linalg.inv(precision) * max(self.noise_variance, 1e-9)
        coefficients = np.linalg.solve(precision, design.T @ target)
        self.intercept = float(coefficients[0])
        self.coef = np.asarray(coefficients[1:], dtype=float)
        residuals = target - design @ coefficients
        self.residual_std = max(float(np.std(residuals, ddof=1)) if len(target) > 1 else 0.0, 1e-6)
        return self

    def predict(self, features: Sequence[float]) -> Prediction:
        if self.coef is None or self.covariance is None:
            raise RuntimeError("model must be fitted")
        vector = np.asarray(features, dtype=float).reshape(-1)
        if vector.shape != self.coef.shape:
            raise ValueError("feature dimension mismatch")
        design = np.concatenate(([1.0], vector))
        mean = float(self.intercept + vector @ self.coef)
        variance = float(design @ self.covariance @ design) + self.residual_std**2
        return Prediction(mean, math.sqrt(max(variance, 1e-12)))


@dataclass
class BootstrapEnsemble:
    task: str
    members: int = 16
    seed: int = 0
    models: list[LinearRegressor] | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> BootstrapEnsemble:
        if not rows:
            raise ValueError("ensemble requires non-empty rows")
        rng = np.random.default_rng(self.seed)
        if self.task not in {"global", "differential"}:
            raise ValueError("task must be global or differential")
        features, targets = _training_arrays(rows, self.task)
        models: list[LinearRegressor] = []
        for _ in range(self.members):
            indices = rng.integers(0, len(rows), size=len(rows))
            model = LinearRegressor(alpha=0.1, noise_variance=max(float(np.var(targets)), 1e-4))
            model.fit(features[indices].tolist(), targets[indices].tolist())
            models.append(model)
        self.models = models
        return self

    def predict_row(self, row: Mapping[str, Any]) -> Prediction:
        if self.models is None:
            raise RuntimeError("ensemble must be fitted")
        features = _features_for_task(row, self.task)
        predictions = np.asarray([model.predict(features.tolist()).mean for model in self.models], dtype=float)
        return Prediction(float(predictions.mean()), max(float(predictions.std(ddof=1)) if len(predictions) > 1 else 0.0, 1e-6))


def _features_for_task(row: Mapping[str, Any], task: str) -> FloatArray:
    if task == "global":
        return global_features(row, "candidate")
    return np.asarray(numeric_features(row), dtype=float)


def _training_arrays(rows: Sequence[Mapping[str, Any]], task: str) -> tuple[FloatArray, FloatArray]:
    features: list[FloatArray] = []
    targets: list[float] = []
    for row in rows:
        if task == "global":
            features.append(global_features(row, "candidate"))
            target = row.get("actor_candidate_value", row.get("delta_true"))
        else:
            features.append(np.asarray(numeric_features(row), dtype=float))
            target = None if row.get("delta_true") is None or row.get("delta_proxy") is None else float(row["delta_true"]) - float(row["delta_proxy"])
        if target is not None:
            targets.append(float(target))
        else:
            features.pop()
    if not targets:
        raise ValueError("rows have no training targets")
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def evaluate_ood(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    family: str = "bayesian_linear",
) -> dict[str, Any]:
    """Evaluate matched global/differential learners on a predeclared OOD split."""

    if not train_rows or not test_rows:
        raise ValueError("train and test rows must be non-empty")
    train_ids = {_evaluation_identity(row) for row in train_rows}
    test_ids = {_evaluation_identity(row) for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("OOD train and test IDs overlap")
    if family == "bayesian_linear":
        global_learner: Any = _fit_global_linear(train_rows)
        differential_learner: Any = _fit_differential_linear(train_rows)
    elif family == "bootstrap_ensemble":
        global_learner = BootstrapEnsemble("global", seed=4091).fit(train_rows)
        differential_learner = BootstrapEnsemble("differential", seed=4092).fit(train_rows)
    else:
        raise ValueError("unknown evaluator family")
    global_predictions: list[float] = []
    global_true: list[float] = []
    global_delta_rows: list[dict[str, Any]] = []
    differential_rows: list[dict[str, Any]] = []
    posterior_sign: list[float] = []
    observed_sign: list[float] = []
    for row in test_rows:
        if family == "bayesian_linear":
            candidate_prediction = global_learner.predict(global_features(row, "candidate"))
            incumbent_prediction = global_learner.predict(global_features(row, "incumbent"))
            correction_prediction = differential_learner.predict(numeric_features(row))
        else:
            candidate_prediction = global_learner.predict_row(row)
            incumbent_prediction = _ensemble_global_incumbent(global_learner, row)
            correction_prediction = differential_learner.predict_row(row)
        global_predictions.append(candidate_prediction.mean)
        global_true.append(float(row["actor_candidate_value"]))
        predicted_global_delta = candidate_prediction.mean - incumbent_prediction.mean
        global_delta_rows.append({"delta_proxy": predicted_global_delta, "delta_true": row["delta_true"], "round_id": row.get("trajectory_id")})
        predicted_transition_delta = float(row["delta_proxy"]) + correction_prediction.mean
        differential_rows.append({"delta_proxy": predicted_transition_delta, "delta_true": row["delta_true"], "round_id": row.get("trajectory_id")})
        posterior_sign.append(1.0 if predicted_transition_delta > 0 else 0.0)
        observed_sign.append(1.0 if float(row["delta_true"]) > 0 else 0.0)
    global_mae = float(np.mean(np.abs(np.asarray(global_predictions) - np.asarray(global_true))))
    global_rmse = float(math.sqrt(np.mean(np.square(np.asarray(global_predictions) - np.asarray(global_true)))))
    global_isc = _isc(global_delta_rows)
    transition_isc = _isc(differential_rows)
    brier = float(np.mean((np.asarray(posterior_sign) - np.asarray(observed_sign)) ** 2))
    return {
        "family": family,
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "policy_MAE": global_mae,
        "policy_RMSE": global_rmse,
        "policy_Spearman": spearman(global_predictions, global_true),
        "global_IDE": _ide(global_delta_rows),
        "global_ISC": global_isc,
        "global_IRR": _irr(global_delta_rows),
        "transition_IDE": _ide(differential_rows),
        "transition_ISC": transition_isc,
        "transition_IRR": _irr(differential_rows),
        "Brier": brier,
        "coverage_50": _coverage(differential_rows, differential_learner, 0.50, family),
        "coverage_80": _coverage(differential_rows, differential_learner, 0.80, family),
        "coverage_95": _coverage(differential_rows, differential_learner, 0.95, family),
        "sign_calibration": _sign_calibration(posterior_sign, observed_sign),
    }


def _fit_global_linear(rows: Sequence[Mapping[str, Any]]) -> LinearRegressor:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        features.append(global_features(row, "candidate").tolist())
        targets.append(float(row["actor_candidate_value"]))
    return LinearRegressor(noise_variance=max(float(np.var(targets)), 1e-4)).fit(features, targets)


def _evaluation_identity(row: Mapping[str, Any]) -> str:
    """Use contextual identity for OOD leakage checks.

    Candidate templates are intentionally reused across environments and
    response regimes.  Those are distinct observations for OOD evaluation;
    only exact reuse within the same context is leakage.
    """

    return "|".join(
        str(row.get(key))
        for key in (
            "transition_id",
            "environment_id",
            "response_strength",
            "operator_family",
            "operator_shift",
            "seed",
        )
    )


def _fit_differential_linear(rows: Sequence[Mapping[str, Any]]) -> LinearRegressor:
    features = [numeric_features(row) for row in rows]
    targets = [float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows]
    return LinearRegressor(noise_variance=max(float(np.var(targets)), 1e-4)).fit(features, targets)


def _ensemble_global_incumbent(ensemble: BootstrapEnsemble, row: Mapping[str, Any]) -> Prediction:
    assert ensemble.models is not None
    features = global_features(row, "incumbent")
    predictions = np.asarray([model.predict(features.tolist()).mean for model in ensemble.models], dtype=float)
    return Prediction(float(predictions.mean()), max(float(predictions.std(ddof=1)), 1e-6))


def _ide(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([abs(float(row["delta_proxy"]) - float(row["delta_true"])) for row in rows]))


def _isc(rows: Sequence[Mapping[str, Any]]) -> float:
    values = [sign(float(row["delta_proxy"])) == sign(float(row["delta_true"])) for row in rows if sign(float(row["delta_proxy"])) and sign(float(row["delta_true"]))]
    return float(np.mean(values)) if values else 0.0


def _irr(rows: Sequence[Mapping[str, Any]]) -> float:
    positive = [row for row in rows if float(row["delta_proxy"]) > 0]
    return float(np.mean([float(row["delta_true"]) < 0 for row in positive])) if positive else 0.0


def _coverage(rows: Sequence[Mapping[str, Any]], learner: Any, level: float, family: str) -> float:
    covered = 0
    for row in rows:
        prediction = learner.predict(numeric_features(row)) if family == "bayesian_linear" else learner.predict_row(row)
        radius = 1.96 * prediction.std * level
        covered += int(abs(float(row["delta_true"]) - (float(row["delta_proxy"]) + prediction.mean)) <= radius)
    return covered / max(len(rows), 1)


def _sign_calibration(predicted: Sequence[float], observed: Sequence[float]) -> list[dict[str, float | int]]:
    bins = []
    for lower, upper in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        selected = [index for index, value in enumerate(predicted) if lower <= value < upper]
        bins.append({"lower": lower, "upper": upper, "n": len(selected), "predicted": float(np.mean([predicted[index] for index in selected])) if selected else 0.0, "observed": float(np.mean([observed[index] for index in selected])) if selected else 0.0})
    return bins
