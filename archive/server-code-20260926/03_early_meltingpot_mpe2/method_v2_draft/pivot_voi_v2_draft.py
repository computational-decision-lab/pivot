"""Unfrozen PIVOT v2 method draft; never use as completed benchmark evidence.

The state is a Gaussian posterior over the *latent deployment improvements* of
the complete candidate set.  A high-fidelity query is a noisy observation of
one latent value.  Query fantasies, real observations, final selection, and
stopping therefore use the same decision rule: select the largest posterior
mean.

This file is deliberately separate from ``sequential_pivot_extension.py`` and
from the frozen Melting Pot source bundle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


VERSION = "pivot_candidate_gaussian_v2_draft_unfrozen"


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _psd_matrix(value: Sequence[Sequence[float]], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has the wrong shape or non-finite entries")
    result = 0.5 * (result + result.T)
    values, vectors = np.linalg.eigh(result)
    scale = max(1.0, float(np.max(np.abs(result))))
    if float(values.min()) < -1e-9 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    # Remove only round-off-sized negative eigenvalues.  Material PSD repair
    # belongs in the frozen calibration procedure, not in the decision loop.
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def expected_positive_normal(mean: float, variance: float) -> float:
    """Return E[max(X, 0)] for X ~ Normal(mean, variance)."""

    if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0.0:
        raise ValueError("normal moments must be finite and variance non-negative")
    if variance <= 1e-18:
        return max(mean, 0.0)
    sd = math.sqrt(variance)
    z = mean / sd
    return sd * _normal_pdf(z) + mean * _normal_cdf(z)


def affine_gaussian_max(means: Sequence[float], slopes: Sequence[float]) -> float:
    """Exactly integrate E[max_i(means[i] + slopes[i] * Z)], Z ~ N(0, 1).

    Candidate counts in the benchmark are small, so enumerating all pairwise
    line intersections is simpler and more reliable than nested Monte Carlo.
    """

    means_array = _finite_vector(means, "means")
    slopes_array = _finite_vector(slopes, "slopes")
    if means_array.shape != slopes_array.shape:
        raise ValueError("means and slopes must have the same shape")
    boundaries = {-math.inf, math.inf}
    for i in range(len(means_array)):
        for j in range(i):
            denominator = slopes_array[i] - slopes_array[j]
            if denominator != 0.0:
                boundaries.add(float((means_array[j] - means_array[i]) / denominator))
    ordered = sorted(boundaries)
    answer = 0.0
    for left, right in zip(ordered, ordered[1:]):
        if left == -math.inf and right == math.inf:
            midpoint = 0.0
        elif left == -math.inf:
            midpoint = right - max(1.0, abs(right))
        elif right == math.inf:
            midpoint = left + max(1.0, abs(left))
        else:
            midpoint = 0.5 * (left + right)
        best = int(np.argmax(means_array + slopes_array * midpoint))
        probability = _normal_cdf(right) - _normal_cdf(left)
        first_moment = _normal_pdf(left) - _normal_pdf(right)
        answer += means_array[best] * probability + slopes_array[best] * first_moment
    return float(answer)


@dataclass(frozen=True)
class CandidateValuePosterior:
    """Joint posterior for latent deployment improvements theta.

    ``observation_variance[j]`` is the variance of the HF sample mean returned
    by one query of candidate j.  It must not include latent model/discrepancy
    variance already represented in ``covariance``.
    """

    ids: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    observation_variance: np.ndarray

    def __post_init__(self) -> None:
        ids = tuple(map(str, self.ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be non-empty and unique")
        mean = _finite_vector(self.mean, "mean")
        if len(mean) != len(ids):
            raise ValueError("mean and ids have different sizes")
        covariance = _psd_matrix(self.covariance, len(ids), "covariance")
        observation_variance = _finite_vector(self.observation_variance, "observation_variance")
        if observation_variance.shape != mean.shape or np.any(observation_variance <= 0.0):
            raise ValueError("observation variances must be positive and match candidates")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "observation_variance", observation_variance)

    @property
    def selected_index(self) -> int:
        return int(np.argmax(self.mean))

    @property
    def selected_id(self) -> str:
        return self.ids[self.selected_index]

    def condition(
        self,
        candidate: int | str,
        observed_value: float,
        *,
        observation_variance: float | None = None,
    ) -> "CandidateValuePosterior":
        """Condition on y_j = theta_j + epsilon using the joint Gaussian rule."""

        index = self.ids.index(candidate) if isinstance(candidate, str) else int(candidate)
        if not 0 <= index < len(self.ids) or not math.isfinite(float(observed_value)):
            raise ValueError("invalid candidate or observation")
        noise = (
            float(self.observation_variance[index])
            if observation_variance is None
            else float(observation_variance)
        )
        if not math.isfinite(noise) or noise <= 0.0:
            raise ValueError("observation variance must be finite and positive")
        column = self.covariance[:, index]
        denominator = float(self.covariance[index, index] + noise)
        gain = column / denominator
        mean = self.mean + gain * (float(observed_value) - float(self.mean[index]))
        covariance = self.covariance - np.outer(column, column) / denominator
        covariance = 0.5 * (covariance + covariance.T)
        return CandidateValuePosterior(self.ids, mean, covariance, self.observation_variance)

    def exact_evsi(self, candidate: int | str) -> float:
        """One-step expected reduction in Bayes simple regret."""

        index = self.ids.index(candidate) if isinstance(candidate, str) else int(candidate)
        if not 0 <= index < len(self.ids):
            raise ValueError("candidate index is out of range")
        denominator = float(self.covariance[index, index] + self.observation_variance[index])
        slopes = self.covariance[:, index] / math.sqrt(denominator)
        value = affine_gaussian_max(self.mean, slopes) - float(np.max(self.mean))
        if value < -1e-10:
            raise ArithmeticError("EVSI identity produced a material negative value")
        return max(0.0, float(value))

    def integrated_contrast_variance_reduction(self, candidate: int | str) -> float:
        """Reduction in centered candidate-value variance, ignoring common mode."""

        index = self.ids.index(candidate) if isinstance(candidate, str) else int(candidate)
        denominator = float(self.covariance[index, index] + self.observation_variance[index])
        column = self.covariance[:, index]
        centered = column - float(np.mean(column))
        return float(centered @ centered / denominator)

    def bayes_simple_regret_upper_bound(self) -> float:
        """Analytic union upper bound on posterior expected simple regret.

        For b = argmax mean, max_i(theta_i-theta_b)_+ is bounded by the sum
        of its pairwise positive parts, each of which has a closed form.
        """

        best = self.selected_index
        bound = 0.0
        for index in range(len(self.ids)):
            if index == best:
                continue
            difference_mean = float(self.mean[index] - self.mean[best])
            difference_variance = float(
                self.covariance[index, index]
                + self.covariance[best, best]
                - 2.0 * self.covariance[index, best]
            )
            bound += expected_positive_normal(difference_mean, max(0.0, difference_variance))
        return float(bound)

    def selected_probability_lower_bound(self) -> float:
        """Bonferroni lower bound for P(the posterior-mean winner is truly best)."""

        best = self.selected_index
        error_upper = 0.0
        for index in range(len(self.ids)):
            if index == best:
                continue
            difference_mean = float(self.mean[index] - self.mean[best])
            difference_variance = float(
                self.covariance[index, index]
                + self.covariance[best, best]
                - 2.0 * self.covariance[index, best]
            )
            if difference_variance <= 1e-18:
                error_upper += float(difference_mean > 0.0)
            else:
                error_upper += _normal_cdf(difference_mean / math.sqrt(difference_variance))
        return max(0.0, 1.0 - error_upper)


def score_queries(
    posterior: CandidateValuePosterior,
    costs: Mapping[str, float],
    available_ids: Sequence[str],
) -> list[dict[str, float | str]]:
    """Score feasible queries by exact EVSI/cost with IVR as a declared tie-break."""

    scores = []
    for identifier in available_ids:
        if identifier not in posterior.ids:
            raise ValueError(f"unknown candidate id: {identifier}")
        cost = float(costs[identifier])
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("query costs must be finite and positive")
        evsi = posterior.exact_evsi(identifier)
        contrast_ivr = posterior.integrated_contrast_variance_reduction(identifier)
        scores.append(
            {
                "transition_id": identifier,
                "evsi": evsi,
                "cost": cost,
                "acquisition": evsi / cost,
                "integrated_contrast_variance_reduction": contrast_ivr,
                "contrast_ivr_per_cost": contrast_ivr / cost,
            }
        )
    # IVR is used only when the primary Bayes utility is exactly tied.  This
    # removes an accidental dependence on lexical candidate ids.
    scores.sort(
        key=lambda row: (
            -float(row["acquisition"]),
            -float(row["contrast_ivr_per_cost"]),
            str(row["transition_id"]),
        )
    )
    return scores


def stop_decision(
    posterior: CandidateValuePosterior,
    scores: Sequence[Mapping[str, float | str]],
    *,
    regret_tolerance: float,
    eta: float | None,
) -> tuple[bool, str | None, dict[str, float]]:
    """Use a decision-risk certificate or a prespecified economic threshold."""

    if not math.isfinite(regret_tolerance) or regret_tolerance < 0.0:
        raise ValueError("regret_tolerance must be finite and non-negative")
    if eta is not None and (not math.isfinite(eta) or eta < 0.0):
        raise ValueError("eta must be None or finite and non-negative")
    risk_bound = posterior.bayes_simple_regret_upper_bound()
    max_acquisition = max((float(row["acquisition"]) for row in scores), default=0.0)
    diagnostics = {
        "bayes_simple_regret_upper_bound": risk_bound,
        "selected_probability_lower_bound": posterior.selected_probability_lower_bound(),
        "max_exact_evsi_per_cost": max_acquisition,
    }
    if risk_bound <= regret_tolerance:
        return True, "regret_certificate", diagnostics
    if eta is not None and max_acquisition <= eta:
        return True, "evsi_per_cost", diagnostics
    if not scores:
        return True, "no_feasible_query", diagnostics
    return False, None, diagnostics


def run_v2(
    posterior: CandidateValuePosterior,
    query: Callable[[str], Mapping[str, float | str] | float],
    costs: Mapping[str, float],
    *,
    episode_budget: float,
    queryable_ids: Sequence[str] | None = None,
    initial_setup_cost: float = 0.0,
    adaptive_stop: bool = False,
    regret_tolerance: float = 0.0,
    eta: float | None = None,
) -> dict[str, object]:
    """Sequential v2 draft with actual cost-budget enforcement.

    Fixed-budget confirmation should call this with ``adaptive_stop=False``.
    Adaptive stopping is a separate cost/regret-frontier analysis.
    """

    if not math.isfinite(episode_budget) or episode_budget < 0.0:
        raise ValueError("episode_budget must be finite and non-negative")
    if not math.isfinite(initial_setup_cost) or initial_setup_cost < 0.0:
        raise ValueError("initial_setup_cost must be finite and non-negative")
    available = list(queryable_ids or [identifier for identifier in posterior.ids if identifier != "incumbent"])
    if len(set(available)) != len(available) or any(identifier not in posterior.ids for identifier in available):
        raise ValueError("queryable ids must be unique posterior candidates")
    state = posterior
    spent = 0.0
    observations: list[dict[str, float | str]] = []
    trace: list[dict[str, object]] = []
    stop_reason = "budget_exhausted"
    while available:
        setup = initial_setup_cost if not observations else 0.0
        feasible = [identifier for identifier in available if spent + setup + float(costs[identifier]) <= episode_budget]
        scores = score_queries(state, costs, feasible)
        stop, reason, diagnostics = stop_decision(
            state,
            scores,
            regret_tolerance=regret_tolerance,
            eta=eta,
        )
        trace.append(
            {
                "posterior_mean": state.mean.tolist(),
                "selected_id_before_query": state.selected_id,
                "scores": scores,
                **diagnostics,
            }
        )
        if not feasible:
            stop_reason = "budget_exhausted"
            break
        if adaptive_stop and stop:
            stop_reason = str(reason)
            break
        chosen = str(scores[0]["transition_id"])
        result = query(chosen)
        if isinstance(result, Mapping):
            if str(result.get("split", "selection")) != "selection":
                raise ValueError("query must come from the selection split")
            observed = float(result["delta"])
            reported_cost = float(result.get("hf_query_cost", costs[chosen]))
        else:
            observed = float(result)
            reported_cost = float(costs[chosen])
        if not math.isfinite(observed) or not math.isclose(reported_cost, float(costs[chosen])):
            raise ValueError("invalid observation or query cost mismatch")
        state = state.condition(chosen, observed)
        charged = reported_cost + setup
        spent += charged
        observations.append(
            {
                "transition_id": chosen,
                "observed_delta": observed,
                "charged_cost": charged,
                "posterior_mean_after": float(state.mean[state.ids.index(chosen)]),
            }
        )
        available.remove(chosen)
    return {
        "version": VERSION,
        "selected_id": state.selected_id,
        "selected_estimate": float(state.mean[state.selected_index]),
        "estimates": dict(zip(state.ids, map(float, state.mean))),
        "observations": observations,
        "charged_cost": spent,
        "stop_reason": stop_reason,
        "trace": trace,
        "decision_rule": "largest posterior mean before and after every noisy query",
    }


def estimate_exchangeable_discrepancy(
    cross_fitted_residual_vectors: Sequence[Sequence[float]],
    known_predictive_covariances: Sequence[Sequence[Sequence[float]]],
) -> tuple[np.ndarray, dict[str, float | list[list[float]]]]:
    """Estimate a two-parameter PSD discrepancy covariance.

    Residuals must be out-of-fold candidate-vector errors.  Each known
    covariance must include the fold-specific linear-posterior covariance and
    the HF observation covariance, preventing those terms from being counted
    again as model discrepancy.
    """

    residuals = np.asarray(cross_fitted_residual_vectors, dtype=float)
    known = np.asarray(known_predictive_covariances, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 2 or not np.all(np.isfinite(residuals)):
        raise ValueError("at least two finite cross-fitted residual vectors are required")
    roots, candidates = residuals.shape
    if known.shape != (roots, candidates, candidates) or not np.all(np.isfinite(known)):
        raise ValueError("known predictive covariances do not match residual vectors")
    raw = residuals.T @ residuals / roots - np.mean(known, axis=0)
    raw = 0.5 * (raw + raw.T)
    unit = np.ones(candidates) / math.sqrt(candidates)
    common_projector = np.outer(unit, unit)
    contrast_projector = np.eye(candidates) - common_projector
    common_variance = max(0.0, float(unit @ raw @ unit))
    contrast_variance = max(0.0, float(np.trace(contrast_projector @ raw) / (candidates - 1)))
    scale = max(1.0, float(np.max(np.abs(raw))))
    if common_variance < 1e-12 * scale:
        common_variance = 0.0
    if contrast_variance < 1e-12 * scale:
        contrast_variance = 0.0
    discrepancy = common_variance * common_projector + contrast_variance * contrast_projector
    diagnostics: dict[str, float | list[list[float]]] = {
        "roots": float(roots),
        "candidates": float(candidates),
        "common_direction_variance": common_variance,
        "contrast_direction_variance": contrast_variance,
        "raw_covariance_after_known_terms": raw.tolist(),
    }
    return discrepancy, diagnostics


def from_linear_calibration(
    ids: Sequence[str],
    proxies: Sequence[float],
    features: Sequence[Sequence[float]],
    coefficient_mean: Sequence[float],
    coefficient_covariance: Sequence[Sequence[float]],
    discrepancy_covariance: Sequence[Sequence[float]],
    observation_variance: Sequence[float],
) -> CandidateValuePosterior:
    """Compose the v2 candidate posterior after discrepancy calibration."""

    proxy = _finite_vector(proxies, "proxies")
    matrix = np.asarray(features, dtype=float)
    beta = _finite_vector(coefficient_mean, "coefficient_mean")
    if matrix.shape != (len(proxy), len(beta)) or not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix does not match candidates and coefficients")
    beta_covariance = _psd_matrix(coefficient_covariance, len(beta), "coefficient_covariance")
    discrepancy = _psd_matrix(discrepancy_covariance, len(proxy), "discrepancy_covariance")
    mean = proxy + matrix @ beta
    covariance = matrix @ beta_covariance @ matrix.T + discrepancy
    normalized_ids = tuple(map(str, ids))
    if "incumbent" in normalized_ids:
        incumbent = normalized_ids.index("incumbent")
        if abs(float(proxy[incumbent])) > 1e-12 or np.max(np.abs(matrix[incumbent])) > 1e-12:
            raise ValueError("incumbent must have zero proxy and zero features")
        if np.max(np.abs(discrepancy[incumbent])) > 1e-12 or np.max(np.abs(discrepancy[:, incumbent])) > 1e-12:
            raise ValueError("incumbent is the zero-delta reference and cannot have discrepancy variance")
        mean[incumbent] = 0.0
        covariance[incumbent, :] = 0.0
        covariance[:, incumbent] = 0.0
    return CandidateValuePosterior(normalized_ids, mean, covariance, np.asarray(observation_variance, dtype=float))
