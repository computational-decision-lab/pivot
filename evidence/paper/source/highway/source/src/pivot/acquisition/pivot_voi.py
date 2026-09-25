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
        variance = (
            self.noise_variance if observation_variance is None else float(observation_variance)
        )
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("observation_variance must be finite and positive")
        assert self.mean is not None and self.covariance is not None
        # Rank-one Bayesian linear update.  Besides being numerically stable,
        # this avoids two dense matrix inversions for every VOI fantasy.
        covariance_vector = self.covariance @ vector
        denominator = variance + float(vector @ covariance_vector)
        covariance = self.covariance - np.outer(covariance_vector, covariance_vector) / denominator
        mean = (
            self.mean
            + covariance_vector * (float(correction) - float(vector @ self.mean)) / denominator
        )
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

    def predictive_variance(
        self, features: FloatArray, *, include_observation: bool = False
    ) -> FloatArray:
        self._require_fitted()
        matrix = _validate_features(features, self.feature_dim)
        assert self.covariance is not None
        variance = np.einsum("ij,jk,ik->i", matrix, self.covariance, matrix)
        if include_observation:
            variance = variance + self.noise_variance
        return np.maximum(np.asarray(variance, dtype=np.float64), 0.0)

    def sample_predictions(
        self, features: FloatArray, samples: int, rng: np.random.Generator
    ) -> FloatArray:
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
        standard_deviation = math.sqrt(
            float(self.predictive_variance(features)[0]) + _observation_variance(candidate, self)
        )
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
    decision_candidates: Sequence[Mapping[str, Any] | Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Score queries by EVSI/cost with a known zero-utility incumbent.

    Decision index zero always denotes retaining the incumbent.  Queried
    candidates may remain in ``decision_candidates`` with ``observed_delta``;
    they are decision alternatives, but are never queried again.
    """

    if not candidates:
        return []
    if fantasies <= 0 or posterior_samples <= 0:
        raise ValueError("fantasies and posterior_samples must be positive")
    posterior._require_fitted()
    candidates = sorted(candidates, key=candidate_id)
    decisions = sorted(
        [
            candidate
            for candidate in (candidates if decision_candidates is None else decision_candidates)
            if not (isinstance(candidate, Mapping) and candidate.get("is_incumbent", False))
        ],
        key=candidate_id,
    )
    decision_ids = [candidate_id(candidate) for candidate in decisions]
    if not {candidate_id(candidate) for candidate in candidates} <= set(decision_ids):
        raise ValueError("query candidates must belong to the decision set")
    rng = np.random.default_rng(seed)
    current_draws, current_means = _action_predictions(decisions, posterior, posterior_samples, rng)
    current_selected = int(np.argmax(current_means))
    current_regret = expected_simple_regret(current_draws, current_selected)
    selection_probability = float(np.mean(np.argmax(current_draws, axis=1) == current_selected))
    confidence_lower, confidence_upper, regret_upper = _gaussian_decision_bounds(
        decisions, posterior, current_selected, current_means
    )

    scores: list[dict[str, float | int | str]] = []
    for index, candidate in enumerate(candidates):
        feature = _features(candidate, posterior.feature_dim)
        mean_correction = float(posterior.predict(feature.reshape(1, -1))[0])
        observation_variance = _observation_variance(candidate, posterior)
        predictive_variance = (
            float(posterior.predictive_variance(feature.reshape(1, -1))[0]) + observation_variance
        )
        fantasy_values = rng.normal(
            mean_correction, math.sqrt(max(predictive_variance, 1e-12)), fantasies
        )
        post_regrets: list[float] = []
        post_selection_probabilities: list[float] = []
        for fantasy in fantasy_values:
            updated = posterior.condition(
                feature, float(fantasy), observation_variance=observation_variance
            )
            updated_draws, updated_means = _action_predictions(
                decisions, updated, posterior_samples, rng
            )
            fantasy_index = decision_ids.index(candidate_id(candidate)) + 1
            observed_delta = _numeric_value(candidate, "delta_proxy") + float(fantasy)
            updated_draws[:, fantasy_index] = observed_delta
            updated_means[fantasy_index] = observed_delta
            updated_selected = int(np.argmax(updated_means))
            post_regrets.append(expected_simple_regret(updated_draws, updated_selected))
            post_selection_probabilities.append(
                float(np.mean(np.argmax(updated_draws, axis=1) == updated_selected))
            )
        evsi = max(0.0, current_regret - float(np.mean(post_regrets)))
        cost = _numeric_value(candidate, cost_key, default=1.0)
        if cost <= 0:
            raise ValueError("declared HF query cost must be positive")
        scores.append(
            {
                "transition_id": candidate_id(candidate),
                "candidate_index": decision_ids.index(candidate_id(candidate)) + 1,
                "current_selected": current_selected,
                "current_selected_id": "incumbent"
                if current_selected == 0
                else decision_ids[current_selected - 1],
                "incumbent_utility": 0.0,
                "decision_count": len(decisions) + 1,
                "current_regret": current_regret,
                "selection_probability": selection_probability,
                "selection_probability_lower": confidence_lower,
                "selection_probability_upper": confidence_upper,
                "probability_bound_method": "gaussian_pairwise_union",
                "current_regret_upper": regret_upper,
                "incumbent_probability": float(np.mean(np.argmax(current_draws, axis=1) == 0)),
                "predicted_correction": mean_correction,
                "predictive_variance": predictive_variance,
                "post_query_regret": float(np.mean(post_regrets)),
                "post_query_selection_probability": float(np.mean(post_selection_probabilities)),
                "evsi": evsi,
                "cost": cost,
                "acquisition": evsi / cost,
                "acquisition_upper": regret_upper / cost,
                "fantasies": fantasies,
                "posterior_samples": posterior_samples,
            }
        )
    scores.sort(key=lambda item: (-float(item["acquisition"]), str(item["transition_id"])))
    return scores


def _observation_variance(
    candidate: Mapping[str, Any] | Any, posterior: BayesianLinearDeltaPosterior
) -> float:
    variance = _numeric_value(candidate, "observation_variance", posterior.noise_variance)
    if variance <= 0:
        raise ValueError("observation variances must be finite and positive")
    return variance


def _gaussian_decision_bounds(
    candidates: Sequence[Mapping[str, Any] | Any],
    posterior: BayesianLinearDeltaPosterior,
    selected: int,
    means: FloatArray,
) -> tuple[float, float, float]:
    """Deterministic bounds under the specified Gaussian outcome posterior.

    Shared coefficient uncertainty is correlated; unobserved outcome residuals
    are conditionally independent. Observed actions and the incumbent are fixed.
    A union bound on pairwise losses bounds P(selected is best); a sum of
    Gaussian positive-part expectations bounds E[max_j(Y_j-Y_selected)].
    EVSI cannot exceed this current-regret bound. These are model-conditional
    bounds, not frequentist coverage or a claim that the model is calibrated.
    """
    matrix = np.zeros((len(candidates) + 1, posterior.feature_dim))
    variances = np.zeros(len(candidates) + 1)
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, Mapping) and candidate.get("hf_queried", False):
            continue
        matrix[index] = _features(candidate, posterior.feature_dim)
        variances[index] = _observation_variance(candidate, posterior)
    assert posterior.covariance is not None
    loss_probabilities = []
    regret_upper = 0.0
    for index in range(len(means)):
        if index == selected:
            continue
        difference = matrix[index] - matrix[selected]
        variance = max(0.0, float(difference @ posterior.covariance @ difference))
        variance += variances[index] + variances[selected]
        mean = float(means[index] - means[selected])
        if variance == 0.0:
            # Canonical argmax assigns deterministic ties to the earlier index.
            probability = float(mean > 0.0 or (mean == 0.0 and index < selected))
            regret_upper += max(mean, 0.0)
        else:
            standard_deviation = math.sqrt(variance)
            z = mean / standard_deviation
            probability = 0.5 * math.erfc(-z / math.sqrt(2.0))
            density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            regret_upper += max(0.0, standard_deviation * density + mean * probability)
        loss_probabilities.append(probability)
    return (
        max(0.0, 1.0 - sum(loss_probabilities)),
        1.0 - max(loss_probabilities, default=0.0),
        regret_upper,
    )


def _action_predictions(
    candidates: Sequence[Mapping[str, Any] | Any],
    posterior: BayesianLinearDeltaPosterior,
    samples: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(
        [_features(candidate, posterior.feature_dim) for candidate in candidates], dtype=np.float64
    )
    proxies = np.asarray([_numeric_value(candidate, "delta_proxy") for candidate in candidates])
    draws = posterior.sample_predictions(matrix, samples, rng) + proxies
    # Target is the as-yet unknown finite-seed HF outcome, not a noiseless
    # regression mean. Use the same residual model as observation fantasies;
    # an actual query reveals this outcome exactly and updates shared coefficients.
    residual_variances = np.asarray(
        [_observation_variance(candidate, posterior) for candidate in candidates]
    )
    if np.any(~np.isfinite(residual_variances)) or np.any(residual_variances <= 0):
        raise ValueError("observation variances must be finite and positive")
    draws += rng.normal(size=draws.shape) * np.sqrt(residual_variances)
    means = posterior.predict(matrix) + proxies
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping) and candidate.get("hf_queried", False):
            if "observed_delta" not in candidate:
                raise ValueError("queried decision candidates require observed_delta")
            observed = _numeric_value(candidate, "observed_delta")
            draws[:, index] = observed
            means[index] = observed
    return np.column_stack((np.zeros(samples), draws)), np.concatenate(([0.0], means))


def select_pivot_voi(
    candidates: Sequence[Mapping[str, Any] | Any],
    posterior: BayesianLinearDeltaPosterior,
    budget: int,
    *,
    seed: int = 0,
    fantasies: int = 64,
    posterior_samples: int = 256,
    cost_key: str = "hf_query_cost",
    decision_candidates: Sequence[Mapping[str, Any] | Any] | None = None,
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
        decision_candidates=decision_candidates,
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


def _validate_training_data(
    features: FloatArray, corrections: FloatArray
) -> tuple[FloatArray, FloatArray]:
    matrix = _validate_features(features)
    target = np.asarray(corrections, dtype=np.float64).reshape(-1)
    if target.shape[0] != matrix.shape[0] or not np.all(np.isfinite(target)):
        raise ValueError("corrections must match finite feature rows")
    return matrix, target
