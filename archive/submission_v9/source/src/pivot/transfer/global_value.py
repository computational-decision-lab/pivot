from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .features import policy_parameter_mapping

FloatArray = NDArray[np.float64]


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-aware rank correlation without a scipy dependency."""

    if len(left) != len(right) or not left:
        raise ValueError("rank inputs must have equal non-zero length")
    left_rank = _average_ranks(np.asarray(left, dtype=float))
    right_rank = _average_ranks(np.asarray(right, dtype=float))
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return 0.0 if denominator == 0.0 else float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


@dataclass
class GlobalValueModel:
    """Ridge policy-value evaluator used as the matched global baseline."""

    alpha: float = 1e-3
    feature_keys: tuple[str, ...] = ()
    coefficients: FloatArray | None = None
    intercept: float = 0.0
    residual_std: float = 0.0
    train_policy_ids: tuple[str, ...] = ()
    hf_budget: int = 0

    def fit(
        self,
        policy_features: Sequence[Mapping[str, float] | Sequence[float]],
        high_fidelity_values: Sequence[float],
        policy_ids: Sequence[str] | None = None,
    ) -> None:
        if len(policy_features) != len(high_fidelity_values) or not high_fidelity_values:
            raise ValueError("features and values must have equal non-zero length")
        matrix, keys = _coerce_feature_matrix(policy_features)
        target = np.asarray(high_fidelity_values, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError("high-fidelity values must be finite")
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        target_mean = float(target.mean())
        regularizer = self.alpha * np.eye(centered.shape[1], dtype=np.float64)
        self.coefficients = np.linalg.solve(centered.T @ centered + regularizer, centered.T @ (target - target_mean))
        self.intercept = target_mean - float(matrix.mean(axis=0) @ self.coefficients)
        residuals = target - self.predict_many(matrix)
        self.residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        self.feature_keys = keys
        self.train_policy_ids = tuple(str(value) for value in (policy_ids or ()))
        self.hf_budget = len(high_fidelity_values)

    def predict(self, policy: Mapping[str, float] | Sequence[float] | Any) -> float:
        if self.coefficients is None:
            raise RuntimeError("GlobalValueModel must be fitted before prediction")
        vector = _coerce_single(policy, self.feature_keys)
        return float(self.intercept + vector @ self.coefficients)

    def predict_many(self, policies: Sequence[Any] | FloatArray) -> FloatArray:
        if self.coefficients is None:
            raise RuntimeError("GlobalValueModel must be fitted before prediction")
        matrix: FloatArray = policies if isinstance(policies, np.ndarray) else np.asarray(
            [_coerce_single(policy, self.feature_keys) for policy in policies], dtype=np.float64
        )
        return np.asarray(self.intercept + matrix @ self.coefficients, dtype=np.float64)

    def predict_row(self, row: Mapping[str, Any], role: str = "candidate") -> float:
        return self.predict(policy_parameter_mapping(row, role=role))

    def uncertainty(self, policy: Any) -> float:
        _ = policy
        return self.residual_std


def _coerce_feature_matrix(
    features: Sequence[Mapping[str, float] | Sequence[float]],
) -> tuple[FloatArray, tuple[str, ...]]:
    if all(isinstance(item, Mapping) for item in features):
        mappings = [item for item in features if isinstance(item, Mapping)]
        keys = tuple(sorted({str(key) for item in mappings for key in item}))
        matrix = np.asarray(
            [[float(item.get(key, 0.0)) for key in keys] for item in mappings],
            dtype=np.float64,
        )
        return matrix, keys
    if any(isinstance(item, Mapping) for item in features):
        raise TypeError("cannot mix mapping and numeric policy features")
    matrix = np.asarray([list(item) for item in features], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("numeric features must be a two-dimensional matrix")
    return matrix, tuple(f"x{index}" for index in range(matrix.shape[1]))


def _coerce_single(policy: Any, keys: Sequence[str]) -> FloatArray:
    if hasattr(policy, "parameters"):
        mapping = policy.parameters
    elif isinstance(policy, Mapping):
        mapping = policy
    else:
        vector = np.asarray(policy, dtype=np.float64)
        if vector.shape != (len(keys),):
            raise ValueError("numeric feature vector has the wrong shape")
        return vector
    if not isinstance(mapping, Mapping):
        raise TypeError("policy features must be a mapping or numeric vector")
    return np.asarray([float(mapping.get(key, 0.0)) for key in keys], dtype=np.float64)
