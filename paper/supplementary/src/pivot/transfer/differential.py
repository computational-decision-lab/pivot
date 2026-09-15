from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .features import transition_feature_vector

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CorrectionPrediction:
    correction: float
    standard_deviation: float
    predicted_delta: float
    sign_change_probability: float


@dataclass
class DifferentialModel:
    """Ridge model for the paired correction `Delta_H - Delta_proxy`."""

    alpha: float = 1e-3
    include_footprint: bool = True
    coefficients: FloatArray | None = None
    intercept: float = 0.0
    residual_std: float = 0.0
    train_transition_ids: tuple[str, ...] = ()
    hf_budget: int = 0

    def fit(
        self,
        transition_features: Sequence[Any],
        high_fidelity_corrections: Sequence[float],
        transition_ids: Sequence[str] | None = None,
    ) -> None:
        if len(transition_features) != len(high_fidelity_corrections) or not high_fidelity_corrections:
            raise ValueError("features and corrections must have equal non-zero length")
        matrix = _coerce_matrix(transition_features, include_footprint=self.include_footprint)
        target = np.asarray(high_fidelity_corrections, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError("corrections must be finite")
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        target_mean = float(target.mean())
        regularizer = self.alpha * np.eye(centered.shape[1], dtype=np.float64)
        self.coefficients = np.linalg.solve(centered.T @ centered + regularizer, centered.T @ (target - target_mean))
        self.intercept = target_mean - float(matrix.mean(axis=0) @ self.coefficients)
        predicted = self.intercept + matrix @ self.coefficients
        residuals = target - predicted
        self.residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        self.train_transition_ids = tuple(str(value) for value in (transition_ids or ()))
        self.hf_budget = len(high_fidelity_corrections)

    def predict_correction(self, transition: Mapping[str, Any] | Any) -> CorrectionPrediction:
        if self.coefficients is None:
            raise RuntimeError("DifferentialModel must be fitted before prediction")
        vector = transition_feature_vector(transition, include_footprint=self.include_footprint)
        correction = float(self.intercept + vector @ self.coefficients)
        delta_proxy = float(transition.get("delta_proxy", 0.0)) if isinstance(transition, Mapping) else float(getattr(transition, "delta_proxy", 0.0) or 0.0)
        predicted_delta = delta_proxy + correction
        probability = _negative_probability(predicted_delta, self.residual_std)
        return CorrectionPrediction(correction, self.residual_std, predicted_delta, probability)

    def predict(self, transition: Mapping[str, Any] | Any) -> float:
        return self.predict_correction(transition).correction

    def uncertainty(self, transition: Mapping[str, Any] | Any) -> float:
        return self.predict_correction(transition).standard_deviation


@dataclass(frozen=True)
class _RegressionStump:
    feature: int
    threshold: float
    left_value: float
    right_value: float


@dataclass
class GradientBoostedDifferentialModel:
    """Small dependency-free gradient-boosted regression-stump baseline."""

    n_estimators: int = 40
    learning_rate: float = 0.08
    initial_value: float = 0.0
    stumps: tuple[_RegressionStump, ...] = ()
    residual_std: float = 0.0
    hf_budget: int = 0

    def fit(
        self,
        transition_features: Sequence[Any],
        high_fidelity_corrections: Sequence[float],
        transition_ids: Sequence[str] | None = None,
    ) -> None:
        _ = transition_ids
        if len(transition_features) != len(high_fidelity_corrections) or not high_fidelity_corrections:
            raise ValueError("features and corrections must have equal non-zero length")
        matrix = _coerce_matrix(transition_features)
        target = np.asarray(high_fidelity_corrections, dtype=np.float64)
        self.initial_value = float(target.mean())
        prediction = np.full(len(target), self.initial_value, dtype=np.float64)
        fitted: list[_RegressionStump] = []
        for _ in range(self.n_estimators):
            residual = target - prediction
            stump = _fit_stump(matrix, residual)
            if stump is None:
                break
            update = np.where(
                matrix[:, stump.feature] <= stump.threshold,
                stump.left_value,
                stump.right_value,
            )
            prediction += self.learning_rate * update
            fitted.append(stump)
        self.stumps = tuple(fitted)
        residual = target - prediction
        self.residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        self.hf_budget = len(target)

    def predict_correction(self, transition: Mapping[str, Any] | Any) -> CorrectionPrediction:
        vector = transition_feature_vector(transition)
        correction = self.initial_value
        for stump in self.stumps:
            correction += self.learning_rate * (
                stump.left_value if vector[stump.feature] <= stump.threshold else stump.right_value
            )
        delta_proxy = float(transition.get("delta_proxy", 0.0)) if isinstance(transition, Mapping) else float(getattr(transition, "delta_proxy", 0.0) or 0.0)
        predicted_delta = delta_proxy + correction
        return CorrectionPrediction(
            correction,
            self.residual_std,
            predicted_delta,
            _negative_probability(predicted_delta, self.residual_std),
        )

    def uncertainty(self, transition: Mapping[str, Any] | Any) -> float:
        return self.predict_correction(transition).standard_deviation


def _fit_stump(matrix: FloatArray, residual: FloatArray) -> _RegressionStump | None:
    best: tuple[float, _RegressionStump] | None = None
    for feature in range(matrix.shape[1]):
        values = np.unique(matrix[:, feature])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left = matrix[:, feature] <= threshold
            if not left.any() or left.all():
                continue
            left_value = float(residual[left].mean())
            right_value = float(residual[~left].mean())
            predicted = np.where(left, left_value, right_value)
            error = float(np.sum((residual - predicted) ** 2))
            stump = _RegressionStump(feature, float(threshold), left_value, right_value)
            if best is None or error < best[0]:
                best = (error, stump)
    return None if best is None else best[1]


def _coerce_matrix(
    features: Sequence[Any], *, include_footprint: bool = True
) -> FloatArray:
    vectors: list[FloatArray] = []
    for feature in features:
        if _is_numeric_vector(feature):
            vectors.append(np.asarray(feature, dtype=np.float64))
        else:
            vectors.append(
                transition_feature_vector(feature, include_footprint=include_footprint)
            )
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("transition features must form a two-dimensional matrix")
    return matrix


def _is_numeric_vector(value: Any) -> bool:
    if isinstance(value, Mapping) or hasattr(value, "delta_proxy"):
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.ndim == 1


def _negative_probability(predicted_delta: float, standard_deviation: float) -> float:
    if standard_deviation <= 1e-12:
        return 1.0 if predicted_delta < 0 else 0.0
    # Normal approximation; use erf from the standard library to avoid scipy.
    import math

    z = -predicted_delta / (standard_deviation * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z)))
