from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pivot.acquisition.common import candidate_id, validate_budget
from pivot.transfer.features import transition_feature_vector

FloatArray = NDArray[np.float64]


@dataclass
class BayesianLinearDeltaPosterior:
    """Conjugate Gaussian posterior for the correction Delta_* - Delta_V."""

    prior_precision: float = 1.0
    noise_variance: float = 1.0
    mean: FloatArray | None = None
    covariance: FloatArray | None = None
    n_observations: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.prior_precision) or self.prior_precision <= 0.0:
            raise ValueError("prior_precision must be finite and positive")
        if not math.isfinite(self.noise_variance) or self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be finite and positive")

    @property
    def feature_dim(self) -> int:
        return 0 if self.mean is None else int(self.mean.shape[0])

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.covariance is not None

    def fit(self, features: FloatArray, corrections: FloatArray) -> BayesianLinearDeltaPosterior:
        matrix, target = _validate_training_data(features, corrections)
        precision = self.prior_precision * np.eye(matrix.shape[1])
        precision += (matrix.T @ matrix) / self.noise_variance
        covariance = np.linalg.inv(precision)
        mean = covariance @ matrix.T @ target / self.noise_variance
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.n_observations = int(matrix.shape[0])
        return self

    def condition(
        self,
        features: Sequence[float] | FloatArray,
        correction: float,
        *,
        observation_variance: float | None = None,
    ) -> BayesianLinearDeltaPosterior:
        """Return a posterior after one hypothetical or observed HF query."""

        self._require_fitted()
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.shape != (self.feature_dim,) or not np.all(np.isfinite(vector)):
            raise ValueError("conditioning features have the wrong shape or are not finite")
        if not math.isfinite(float(correction)):
            raise ValueError("conditioning correction must be finite")
        variance = self.noise_variance if observation_variance is None else float(observation_variance)
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("observation_variance must be finite and positive")
        assert self.mean is not None and self.covariance is not None
        # Rank-one Bayesian linear update.  Besides being numerically stable,
        # this avoids two dense matrix inversions for every VOI fantasy.
        covariance_vector = self.covariance @ vector
        denominator = variance + float(vector @ covariance_vector)
        covariance = self.covariance - np.outer(covariance_vector, covariance_vector) / denominator
        mean = self.mean + covariance_vector * (float(correction) - float(vector @ self.mean)) / denominator
        return BayesianLinearDeltaPosterior(
            prior_precision=self.prior_precision,
            noise_variance=self.noise_variance,
            mean=np.asarray(mean, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            n_observations=self.n_observations + 1,
        )

    def predict(self, features: FloatArray) -> FloatArray:
        self._require_fitted()
        matrix = _validate_features(features, self.feature_dim)
        assert self.mean is not None
        return np.asarray(matrix @ self.mean, dtype=np.float64)

    def predictive_variance(self, features: FloatArray, *, include_observation: bool = False) -> FloatArray:
        self._require_fitted()
        matrix = _validate_features(features, self.feature_dim)
        assert self.covariance is not None
        variance = np.einsum("ij,jk,ik->i", matrix, self.covariance, matrix)
        if include_observation:
            variance = variance + self.noise_variance
        return np.maximum(np.asarray(variance, dtype=np.float64), 0.0)

    def sample_predictions(self, features: FloatArray, samples: int, rng: np.random.Generator) -> FloatArray:
        self._require_fitted()
        if samples <= 0:
            raise ValueError("samples must be positive")
        matrix = _validate_features(features, self.feature_dim)
        assert self.mean is not None and self.covariance is not None
        coefficients = rng.multivariate_normal(self.mean, self.covariance, size=samples)
        return np.asarray(coefficients @ matrix.T, dtype=np.float64)

    def predict_correction(self, candidate: Mapping[str, Any] | Any) -> Any:
        """Expose the legacy correction-model protocol for round orchestration."""

        features = np.asarray([_features(candidate, self.feature_dim)], dtype=np.float64)
        correction = float(self.predict(features)[0])
        standard_deviation = math.sqrt(float(self.predictive_variance(features, include_observation=True)[0]))
        delta_proxy = _numeric_value(candidate, "delta_proxy")
        predicted_delta = delta_proxy + correction
        return PosteriorCorrectionPrediction(
            correction=correction,
            standard_deviation=standard_deviation,
            predicted_delta=predicted_delta,
            sign_change_probability=1.0 if predicted_delta < 0.0 else 0.0,
        )

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("BayesianLinearDeltaPosterior must be fitted before prediction")


@dataclass(frozen=True)
class PosteriorCorrectionPrediction:
    correction: float
    standard_deviation: float
    predicted_delta: float
    sign_change_probability: float


def expected_simple_regret(samples: FloatArray, selected_index: int) -> float:
    """Estimate E[max_j Delta_j - Delta_selected] from posterior draws."""

    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] == 0:
        raise ValueError("posterior samples must be a non-empty two-dimensional array")
    if not 0 <= selected_index < draws.shape[1]:
        raise ValueError("selected_index is outside the candidate set")
    regret = np.max(draws, axis=1) - draws[:, selected_index]
    return float(np.mean(np.maximum(regret, 0.0)))


def score_pivot_voi(
    candidates: Sequence[Mapping[str, Any] | Any],
    posterior: BayesianLinearDeltaPosterior,
    *,
    seed: int = 0,
    fantasies: int = 64,
    posterior_samples: int = 256,
    cost_key: str = "hf_query_cost",
) -> list[dict[str, float | int | str]]:
    """Score queries by Monte-Carlo EVSI per high-fidelity cost."""

    if not candidates:
        return []
    if fantasies <= 0 or posterior_samples <= 0:
        raise ValueError("fantasies and posterior_samples must be positive")
    posterior._require_fitted()
    rng = np.random.default_rng(seed)
    matrix = np.asarray(
        [_features(candidate, posterior.feature_dim) for candidate in candidates], dtype=np.float64
    )
    proxies = np.asarray(
        [_numeric_value(candidate, "delta_proxy") for candidate in candidates], dtype=np.float64
    )
    current_draws = posterior.sample_predictions(matrix, posterior_samples, rng) + proxies
    current_means = posterior.predict(matrix) + proxies
    current_selected = int(np.argmax(current_means))
    current_regret = expected_simple_regret(current_draws, current_selected)
    selection_probability = float(np.mean(np.argmax(current_draws, axis=1) == current_selected))

    scores: list[dict[str, float | int | str]] = []
    for index, candidate in enumerate(candidates):
        feature = matrix[index]
        mean_correction = float(posterior.predict(feature.reshape(1, -1))[0])
        predictive_variance = float(
            posterior.predictive_variance(feature.reshape(1, -1), include_observation=True)[0]
        )
        fantasy_values = rng.normal(mean_correction, math.sqrt(max(predictive_variance, 1e-12)), fantasies)
        post_regrets: list[float] = []
        post_selection_probabilities: list[float] = []
        for fantasy in fantasy_values:
            updated = posterior.condition(feature, float(fantasy))
            updated_draws = updated.sample_predictions(matrix, posterior_samples, rng) + proxies
            updated_means = updated.predict(matrix) + proxies
            updated_selected = int(np.argmax(updated_means))
            post_regrets.append(expected_simple_regret(updated_draws, updated_selected))
            post_selection_probabilities.append(
                float(np.mean(np.argmax(updated_draws, axis=1) == updated_selected))
            )
        evsi = max(0.0, current_regret - float(np.mean(post_regrets)))
        cost = max(_numeric_value(candidate, cost_key, default=1.0), 1e-12)
        scores.append(
            {
                "transition_id": candidate_id(candidate),
                "candidate_index": index,
                "current_selected": current_selected,
                "current_regret": current_regret,
                "selection_probability": selection_probability,
                "predicted_correction": mean_correction,
                "predictive_variance": predictive_variance,
                "post_query_regret": float(np.mean(post_regrets)),
                "post_query_selection_probability": float(np.mean(post_selection_probabilities)),
                "evsi": evsi,
                "cost": cost,
                "acquisition": evsi / cost,
                "fantasies": fantasies,
                "posterior_samples": posterior_samples,
            }
        )
    scores.sort(key=lambda item: (-float(item["acquisition"]), str(item["transition_id"])))
    return scores


def select_pivot_voi(
    candidates: Sequence[Mapping[str, Any] | Any],
    posterior: BayesianLinearDeltaPosterior,
    budget: int,
    *,
    seed: int = 0,
    fantasies: int = 64,
    posterior_samples: int = 256,
    cost_key: str = "hf_query_cost",
) -> list[str]:
    """Select exactly `budget` candidates by EVSI divided by HF cost."""

    validate_budget(candidates, budget)
    scores = score_pivot_voi(
        candidates,
        posterior,
        seed=seed,
        fantasies=fantasies,
        posterior_samples=posterior_samples,
        cost_key=cost_key,
    )
    return [str(item["transition_id"]) for item in scores[:budget]]


def should_stop(
    *,
    selection_probability: float,
    max_acquisition: float,
    delta: float,
    eta: float,
) -> tuple[bool, str | None]:
    """Apply the prespecified posterior-confidence or EVSI-per-cost stop."""

    if not 0.0 <= selection_probability <= 1.0:
        raise ValueError("selection_probability must lie in [0, 1]")
    if not math.isfinite(max_acquisition) or max_acquisition < 0.0:
        raise ValueError("max_acquisition must be finite and non-negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    if selection_probability >= 1.0 - delta:
        return True, "selection_probability"
    if max_acquisition < eta:
        return True, "evsi_per_cost"
    return False, None


def _features(candidate: Mapping[str, Any] | Any, feature_dim: int) -> FloatArray:
    values: FloatArray
    if isinstance(candidate, Mapping) and "features" in candidate:
        values = np.asarray(candidate["features"], dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(transition_feature_vector(candidate), dtype=np.float64).reshape(-1)
    return cast(FloatArray, _validate_features(values.reshape(1, -1), feature_dim)[0])


def _numeric_value(candidate: Mapping[str, Any] | Any, key: str, default: float = 0.0) -> float:
    if isinstance(candidate, Mapping):
        value = candidate.get(key, default)
    else:
        value = getattr(candidate, key, default)
    if value is None:
        value = default
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    return numeric


def _validate_features(features: FloatArray, feature_dim: int | None = None) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a non-empty finite matrix")
    if feature_dim is not None and matrix.shape[1] != feature_dim:
        raise ValueError("features have the wrong dimension")
    return matrix


def _validate_training_data(features: FloatArray, corrections: FloatArray) -> tuple[FloatArray, FloatArray]:
    matrix = _validate_features(features)
    target = np.asarray(corrections, dtype=np.float64).reshape(-1)
    if target.shape[0] != matrix.shape[0] or not np.all(np.isfinite(target)):
        raise ValueError("corrections must match finite feature rows")
    return matrix, target
