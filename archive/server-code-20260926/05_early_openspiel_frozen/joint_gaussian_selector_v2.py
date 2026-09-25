"""Development-v2 joint-Gaussian sequential selector.

This module is a research candidate, not the author's original PIVOT method and
not a replacement for ``sequential_pivot_extension.py``.  It fixes one narrow
estimator mismatch: acquisition fantasies, real-query conditioning, and the
final decision all use the same joint Gaussian posterior over candidate latent
values.

Inputs are candidate-level latent means and covariance.  Query noise is an
observation *variance*.  A query observes ``latent_value[j] + noise`` and the
whole candidate vector is conditioned by the standard Gaussian formula.  The
VOI acquisition is deterministic Gauss-Hermite quadrature of

    E[max_i E[f_i | y_j]] - max_i E[f_i].

All methods use the same cost budget, posterior update, no-repeat constraint,
and posterior-mean final decision.  ``posterior_lucb`` is a posterior
best/challenger heuristic; it is not claimed to be classical frequentist LUCB.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss


VERSION = "joint_gaussian_selector_development_v2_not_author_pivot"
METHODS = (
    "joint_gaussian_voi",
    "uniform_random",
    "global_ivr",
    "posterior_lucb",
)


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def affine_gaussian_max(means: Sequence[float], slopes: Sequence[float]) -> float:
    """Exactly integrate ``E[max_i(means[i] + slopes[i] Z)]`` for normal Z."""
    intercept = np.asarray(means, dtype=float).reshape(-1)
    gradient = np.asarray(slopes, dtype=float).reshape(-1)
    if (
        intercept.size == 0
        or intercept.shape != gradient.shape
        or not np.all(np.isfinite(intercept))
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError("means and slopes must be matching non-empty finite vectors")
    boundaries = {-math.inf, math.inf}
    for i in range(len(intercept)):
        for j in range(i):
            denominator = float(gradient[i] - gradient[j])
            if denominator != 0.0:
                boundaries.add(float((intercept[j] - intercept[i]) / denominator))
    ordered = sorted(boundaries)
    expectation = 0.0
    for left, right in zip(ordered, ordered[1:]):
        if left == -math.inf and right == math.inf:
            midpoint = 0.0
        elif left == -math.inf:
            midpoint = right - max(1.0, abs(right))
        elif right == math.inf:
            midpoint = left + max(1.0, abs(left))
        else:
            midpoint = 0.5 * (left + right)
        best = int(np.argmax(intercept + gradient * midpoint))
        probability = _normal_cdf(right) - _normal_cdf(left)
        first_moment = _normal_pdf(left) - _normal_pdf(right)
        expectation += intercept[best] * probability + gradient[best] * first_moment
    return float(expectation)


def _as_vector(values: Sequence[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (n,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-{n} vector")
    return array.copy()


def _stabilize_covariance(covariance: Sequence[Sequence[float]], n: int) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"latent_covariance must be a finite {n}x{n} matrix")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10 * scale):
        raise ValueError("latent_covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(eigenvalues[0]) < -1e-9 * scale:
        raise ValueError("latent_covariance must be positive semidefinite")
    if eigenvalues[0] < 0.0:
        matrix = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        matrix = (matrix + matrix.T) / 2.0
    return matrix


@lru_cache(maxsize=None)
def _standard_normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if type(order) is not int or order < 8 or order > 320:
        raise ValueError("quadrature_order must be an integer in [8, 320]")
    nodes, weights = hermgauss(order)
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def gauss_hermite_evsi(
    latent_mean: Sequence[float],
    latent_covariance: Sequence[Sequence[float]],
    observation_noise: Sequence[float],
    query_index: int,
    *,
    quadrature_order: int = 128,
) -> float:
    """Return deterministic one-query EVSI for a posterior-mean decision.

    ``observation_noise`` contains variances, rather than standard deviations.
    The incumbent (and any other candidate) remains in the maximum even when it
    is not queryable.
    """
    mean = np.asarray(latent_mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("latent_mean must be a non-empty finite vector")
    n = len(mean)
    covariance = _stabilize_covariance(latent_covariance, n)
    noise = _as_vector(observation_noise, n, "observation_noise")
    if np.any(noise < 0.0):
        raise ValueError("observation_noise variances must be nonnegative")
    if type(query_index) is not int or not 0 <= query_index < n:
        raise ValueError("query_index is out of range")

    predictive_variance = float(covariance[query_index, query_index] + noise[query_index])
    if predictive_variance <= 1e-15:
        return 0.0
    shifts = covariance[:, query_index] / math.sqrt(predictive_variance)
    if float(np.max(np.abs(shifts))) <= 1e-15:
        return 0.0
    nodes, weights = _standard_normal_quadrature(quadrature_order)
    posterior_means = mean[None, :] + nodes[:, None] * shifts[None, :]
    expected_best = float(weights @ np.max(posterior_means, axis=1))
    return max(0.0, expected_best - float(np.max(mean)))


def exact_affine_evsi(
    latent_mean: Sequence[float],
    latent_covariance: Sequence[Sequence[float]],
    observation_noise: Sequence[float],
    query_index: int,
) -> float:
    """Exact one-query EVSI under the joint-Gaussian posterior-mean rule."""
    mean = np.asarray(latent_mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("latent_mean must be a non-empty finite vector")
    covariance = _stabilize_covariance(latent_covariance, len(mean))
    noise = _as_vector(observation_noise, len(mean), "observation_noise")
    if np.any(noise < 0.0) or type(query_index) is not int or not 0 <= query_index < len(mean):
        raise ValueError("invalid observation noise or query index")
    predictive_variance = float(covariance[query_index, query_index] + noise[query_index])
    if predictive_variance <= 1e-15:
        return 0.0
    slopes = covariance[:, query_index] / math.sqrt(predictive_variance)
    value = affine_gaussian_max(mean, slopes) - float(np.max(mean))
    if value < -1e-10:
        raise ArithmeticError("exact EVSI produced a material negative value")
    return max(0.0, float(value))


def _global_ivr(covariance: np.ndarray, noise: np.ndarray, index: int) -> float:
    denominator = float(covariance[index, index] + noise[index])
    return 0.0 if denominator <= 1e-15 else float(covariance[:, index] @ covariance[:, index] / denominator)


def condition_joint_gaussian(
    latent_mean: Sequence[float],
    latent_covariance: Sequence[Sequence[float]],
    observation_noise: Sequence[float],
    query_index: int,
    observation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition a joint latent Gaussian on one noisy scalar observation."""
    mean = np.asarray(latent_mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("latent_mean must be a non-empty finite vector")
    n = len(mean)
    covariance = _stabilize_covariance(latent_covariance, n)
    noise = _as_vector(observation_noise, n, "observation_noise")
    if np.any(noise < 0.0):
        raise ValueError("observation_noise variances must be nonnegative")
    if type(query_index) is not int or not 0 <= query_index < n:
        raise ValueError("query_index is out of range")
    if not math.isfinite(float(observation)):
        raise ValueError("observation must be finite")

    column = covariance[:, query_index].copy()
    predictive_variance = float(covariance[query_index, query_index] + noise[query_index])
    if predictive_variance <= 1e-15:
        tolerance = 1e-9 * max(1.0, abs(float(mean[query_index])))
        if not math.isclose(float(observation), float(mean[query_index]), abs_tol=tolerance):
            raise ValueError("observation contradicts a deterministic candidate")
        return mean.copy(), covariance.copy()
    gain = column / predictive_variance
    updated_mean = mean + gain * (float(observation) - float(mean[query_index]))
    updated_covariance = covariance - np.outer(column, column) / predictive_variance
    updated_covariance = _stabilize_covariance(updated_covariance, n)
    return updated_mean, updated_covariance


def _callback_observation(result: object, expected_cost: float) -> float:
    if isinstance(result, Mapping):
        if "observation" not in result:
            raise ValueError("query mapping must contain 'observation'")
        observation = float(result["observation"])
        if "cost" in result and not math.isclose(
            float(result["cost"]), expected_cost, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("reported query cost differs from candidate cost")
    else:
        observation = float(result)
    if not math.isfinite(observation):
        raise ValueError("query returned a nonfinite observation")
    return observation


def run_joint_gaussian_selector(
    candidate_ids: Sequence[str],
    latent_mean: Sequence[float],
    latent_covariance: Sequence[Sequence[float]],
    observation_noise: Sequence[float],
    query_costs: Sequence[float],
    query: Callable[[dict], object],
    *,
    method: str,
    budget: float,
    seed: int = 0,
    queryable: Sequence[bool] | None = None,
    incumbent_id: str | None = None,
    quadrature_order: int = 128,
    lucb_beta: float = 1.96,
    stop_on_zero_voi: bool = True,
) -> dict:
    """Run a cost-budgeted, no-repeat sequential selection policy.

    ``query`` receives a fresh dictionary with ``candidate_id``, ``index``,
    ``posterior_mean``, ``posterior_variance``, ``observation_noise``, and
    ``cost``.  It may return a scalar observation or
    ``{"observation": value, "cost": matching_cost}``.
    """
    ids = [str(candidate_id) for candidate_id in candidate_ids]
    n = len(ids)
    if n == 0 or len(set(ids)) != n:
        raise ValueError("candidate_ids must be non-empty and unique")
    mean = _as_vector(latent_mean, n, "latent_mean")
    covariance = _stabilize_covariance(latent_covariance, n)
    noise = _as_vector(observation_noise, n, "observation_noise")
    costs = _as_vector(query_costs, n, "query_costs")
    if np.any(noise < 0.0):
        raise ValueError("observation_noise variances must be nonnegative")
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if not math.isfinite(float(budget)) or float(budget) < 0.0:
        raise ValueError("budget must be finite and nonnegative")
    if not math.isfinite(float(lucb_beta)) or float(lucb_beta) < 0.0:
        raise ValueError("lucb_beta must be finite and nonnegative")
    _standard_normal_quadrature(quadrature_order)

    if queryable is None:
        can_query = np.ones(n, dtype=bool)
    else:
        can_query = np.asarray(queryable, dtype=bool)
        if can_query.shape != (n,):
            raise ValueError(f"queryable must be a length-{n} boolean vector")
        can_query = can_query.copy()
    if incumbent_id is not None:
        if str(incumbent_id) not in ids:
            raise ValueError("incumbent_id is not a candidate")
        can_query[ids.index(str(incumbent_id))] = False
    if np.any(costs[can_query] <= 0.0) or np.any(costs[~can_query] < 0.0):
        raise ValueError("queryable costs must be positive and other costs nonnegative")

    rng = np.random.default_rng(seed)
    spent = 0.0
    queried: set[int] = set()
    steps: list[dict] = []
    stop_reason = "no_affordable_unqueried_candidate"
    tolerance = 1e-12 * max(1.0, float(budget))

    while True:
        remaining = float(budget) - spent
        available = [
            index
            for index in range(n)
            if can_query[index] and index not in queried and costs[index] <= remaining + tolerance
        ]
        if not available:
            break

        details: dict = {}
        if method == "joint_gaussian_voi":
            acquisitions = {
                index: exact_affine_evsi(mean, covariance, noise, index)
                for index in available
            }
            tie_break = {index: _global_ivr(covariance, noise, index) for index in available}
            index = max(
                available,
                key=lambda j: (acquisitions[j] / costs[j], tie_break[j] / costs[j], -j),
            )
            details = {
                "evsi": {ids[j]: float(acquisitions[j]) for j in available},
                "evsi_per_cost": {ids[j]: float(acquisitions[j] / costs[j]) for j in available},
                "global_ivr_tie_break_per_cost": {
                    ids[j]: float(tie_break[j] / costs[j]) for j in available
                },
                "evsi_engine": "exact_affine_normal_upper_envelope",
            }
            if stop_on_zero_voi and acquisitions[index] <= 1e-12:
                stop_reason = "zero_voi"
                break
        elif method == "uniform_random":
            index = int(rng.choice(available))
        elif method == "global_ivr":
            reductions = {}
            for j in available:
                reductions[j] = _global_ivr(covariance, noise, j)
            index = max(available, key=lambda j: (reductions[j] / costs[j], -j))
            details = {
                "integrated_variance_reduction": {ids[j]: float(reductions[j]) for j in available},
                "integrated_variance_reduction_per_cost": {
                    ids[j]: float(reductions[j] / costs[j]) for j in available
                },
            }
        elif method == "posterior_lucb":
            standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            best = int(np.argmax(mean))
            others = [j for j in range(n) if j != best]
            challenger = max(others, key=lambda j: (mean[j] + lucb_beta * standard_deviation[j], -j)) if others else best
            pair = [j for j in (best, challenger) if j in available]
            pool = pair if pair else available
            index = max(pool, key=lambda j: (standard_deviation[j] / costs[j], -j))
            details = {
                "best_id": ids[best],
                "challenger_id": ids[challenger],
                "posterior_sd_per_cost": {
                    ids[j]: float(standard_deviation[j] / costs[j]) for j in pool
                },
                "heuristic": "posterior best/challenger; query available member with largest SD per cost",
            }
        else:  # guarded above
            raise AssertionError(method)

        request = {
            "candidate_id": ids[index],
            "index": index,
            "posterior_mean": float(mean[index]),
            "posterior_variance": float(covariance[index, index]),
            "observation_noise": float(noise[index]),
            "cost": float(costs[index]),
        }
        prior_mean = mean.copy()
        prior_covariance = covariance.copy()
        observation = _callback_observation(query(dict(request)), float(costs[index]))
        mean, covariance = condition_joint_gaussian(
            mean, covariance, noise, index, observation
        )
        queried.add(index)
        spent += float(costs[index])
        steps.append(
            {
                "step": len(steps),
                "query_id": ids[index],
                "query_index": index,
                "observation": observation,
                "charged_cost": float(costs[index]),
                "spent_cost": spent,
                "remaining_budget": max(0.0, float(budget) - spent),
                "prior_mean": prior_mean.tolist(),
                "posterior_mean": mean.tolist(),
                "prior_variance": np.diag(prior_covariance).tolist(),
                "posterior_variance": np.diag(covariance).tolist(),
                **details,
            }
        )
        stop_reason = "candidate_set_exhausted" if len(queried) == int(np.sum(can_query)) else "budget_exhausted"

    selected = int(np.argmax(mean))
    return {
        "version": VERSION,
        "method": method,
        "selected_id": ids[selected],
        "selected_index": selected,
        "selected_posterior_mean": float(mean[selected]),
        "posterior_mean": mean.tolist(),
        "posterior_covariance": covariance.tolist(),
        "queried_ids": [step["query_id"] for step in steps],
        "query_count": len(steps),
        "spent_cost": spent,
        "remaining_budget": max(0.0, float(budget) - spent),
        "stop_reason": stop_reason,
        "steps": steps,
        "incumbent_id": None if incumbent_id is None else str(incumbent_id),
        "final_decision_rule": "argmax_joint_gaussian_posterior_mean",
        "no_repeat_queries": True,
        "author_original": False,
        "development_candidate": True,
    }
