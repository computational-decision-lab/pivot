"""Finite-decision latent Gaussian population VOI; no noisy observation is truth.

Explicit priors and independent-observation likelihoods are input contracts, not
claims of empirical calibration. All matrices retain their full covariance.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .population_stopping import PosteriorStopRule, decision_bounds


def _array(value: Any, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError("expected finite array with correct dimensions")
    array.flags.writeable = False
    return array


def _scaled_covariance(value: Any, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate in correlation units and return a standardized square root.

    Negative marginal variances and nonzero zero-variance rows are always
    invalid. Congruence by marginal standard deviations makes the PSD check
    invariant to units. Only negative eigenvalues within an eigensolver-scale
    backward-error allowance (8*n*eps*||correlation||_inf) may be projected to
    zero; the repaired covariance is returned, never an indefinite input.
    Cholesky preserves every resolvable positive mode without a rank cutoff.
    """
    array = _array(value, 2)
    if array.shape != (size, size) or not np.allclose(array, array.T, atol=0, rtol=1e-12):
        raise ValueError("covariance dimensions or symmetry invalid")
    array = 0.5 * array + 0.5 * array.T
    diagonal = np.diag(array)
    active = diagonal > 0
    if np.any(diagonal < 0) or np.any(array[~active] != 0):
        raise ValueError("covariance must be positive semidefinite")
    scales = np.ones(size)
    scales[active] = np.sqrt(diagonal[active])
    with np.errstate(over="ignore", invalid="ignore"):
        correlation = array[np.ix_(active, active)] / scales[active, None] / scales[None, active]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("covariance must be positive semidefinite")
    correlation = 0.5 * correlation + 0.5 * correlation.T
    np.fill_diagonal(correlation, 1.0)
    if not np.any(active):
        factor = np.zeros((size, 0))
    else:
        try:
            root = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            tolerance = (
                8 * len(correlation) * np.finfo(float).eps * np.linalg.norm(correlation, ord=np.inf)
            )
            if eigenvalues[0] < -tolerance:
                raise ValueError("covariance must be positive semidefinite") from None
            positive = eigenvalues > tolerance
            root = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
            # Canonicalize roundoff-scale singular input; do not retain negatives.
            repaired = root @ root.T
            array[np.ix_(active, active)] = repaired * scales[active, None] * scales[None, active]
        factor = np.zeros((size, root.shape[1]))
        factor[active] = root
    array.flags.writeable = False
    return array, scales, factor


def _psd(value: Any, size: int) -> np.ndarray:
    return _scaled_covariance(value, size)[0]


@dataclass(frozen=True)
class PopulationObservation:
    observation_id: str
    values: tuple[float, ...]
    actual_cost: float

    def __post_init__(self) -> None:
        if not self.observation_id or not math.isfinite(self.actual_cost) or self.actual_cost < 0:
            raise ValueError("invalid observation identity or cost")
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("observation values must be finite")


@dataclass(frozen=True)
class QueryAction:
    candidate_id: str
    design: np.ndarray
    noise_covariance: np.ndarray
    cost: float
    observation_kind: str

    def __post_init__(self) -> None:
        design = _array(self.design, 2)
        if design.shape[0] not in (1, 2):
            raise ValueError("only scalar delta and paired observation dimensions supported")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "noise_covariance", _psd(self.noise_covariance, design.shape[0]))
        if not self.candidate_id or not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("positive finite declared query cost required")


@dataclass(frozen=True)
class GaussianPopulation:
    candidate_ids: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    decision_matrix: np.ndarray
    representation: str
    model_spec: str
    version: int = 0
    observation_ids: tuple[str, ...] = ()
    queried_candidates: tuple[str, ...] = ()
    lineage_id: str = ""

    def __post_init__(self) -> None:
        ids = tuple(self.candidate_ids)
        if not ids or len(set(ids)) != len(ids) or "incumbent" in ids or not all(ids):
            raise ValueError("candidate identifiers must be unique nonempty and exclude incumbent")
        mean = _array(self.mean, 1)
        decision = _array(self.decision_matrix, 2)
        if decision.shape != (len(ids) + 1, mean.size) or np.any(decision[0] != 0):
            raise ValueError("decision map must include exact zero incumbent")
        if not self.model_spec:
            raise ValueError("explicit model specification required")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _psd(self.covariance, mean.size))
        object.__setattr__(self, "decision_matrix", decision)
        if not isinstance(self.lineage_id, str):
            raise TypeError("model lineage identifier must be a string")

    @classmethod
    def delta(
        cls, candidate_ids: Sequence[str], mean: Any, covariance: Any, *, model_spec: str
    ) -> GaussianPopulation:
        size = len(candidate_ids)
        return cls(
            tuple(candidate_ids),
            mean,
            covariance,
            np.vstack([np.zeros(size), np.eye(size)]),
            "delta",
            model_spec,
        )

    @classmethod
    def absolute(
        cls, candidate_ids: Sequence[str], mean: Any, covariance: Any, *, model_spec: str
    ) -> GaussianPopulation:
        size = len(candidate_ids) + 1
        decision = np.eye(size)
        decision[:, 0] -= 1
        return cls(tuple(candidate_ids), mean, covariance, decision, "absolute", model_spec)

    @property
    def decision_means(self) -> np.ndarray:
        return self.decision_matrix @ self.mean

    @property
    def decision_covariance(self) -> np.ndarray:
        return self.decision_matrix @ self.covariance @ self.decision_matrix.T

    @property
    def selected_candidate_id(self) -> str:
        return ("incumbent", *self.candidate_ids)[int(np.argmax(self.decision_means))]

    def delta_query(self, candidate_id: str, *, noise_variance: float, cost: float) -> QueryAction:
        if self.representation != "delta":
            raise ValueError("delta_query requires a pure delta representation")
        index = self.candidate_ids.index(candidate_id)
        return QueryAction(
            candidate_id, np.eye(len(self.mean))[[index]], [[noise_variance]], cost, "delta"
        )

    def absolute_pair_query(
        self, candidate_id: str, *, noise_covariance: Any, cost: float
    ) -> QueryAction:
        if self.representation != "absolute":
            raise ValueError("absolute_pair_query requires absolute coordinates")
        index = self.candidate_ids.index(candidate_id) + 1
        return QueryAction(
            candidate_id,
            np.eye(len(self.mean))[[0, index]],
            noise_covariance,
            cost,
            "absolute_pair",
        )

    def _predictive(self, action: QueryAction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            action.candidate_id not in self.candidate_ids
            or action.design.shape[1] != self.mean.size
        ):
            raise ValueError("query does not belong to this decision set/model")
        cross = self.covariance @ action.design.T
        predictive = action.design @ cross + action.noise_covariance
        predictive, scales, normalized_factor = _scaled_covariance(
            predictive, action.design.shape[0]
        )
        active = np.diag(predictive) > 0
        root = normalized_factor[active]
        scaled_cross = cross[:, active] / scales[active]
        gain = np.zeros_like(cross)
        if root.shape[1] == np.count_nonzero(active):
            # Two solves with the correlation Cholesky factor retain small axes.
            gain[:, active] = (
                np.linalg.solve(root.T, np.linalg.solve(root, scaled_cross.T)).T / scales[active]
            )
        elif root.shape[1]:
            # Eigen square root columns are orthogonal in standardized units.
            inverse = root / np.sum(root * root, axis=0)
            gain[:, active] = (scaled_cross @ inverse) @ inverse.T / scales[active]
        return scales[:, None] * normalized_factor, gain, predictive

    def condition(
        self, action: QueryAction, observation: PopulationObservation
    ) -> GaussianPopulation:
        if observation.observation_id in self.observation_ids:
            raise ValueError("duplicate observation cannot condition twice")
        values = np.asarray(observation.values)
        if values.shape != (action.design.shape[0],):
            raise ValueError("observation dimension differs from likelihood")
        factor, gain, predictive = self._predictive(action)
        residual = values - action.design @ self.mean
        active = np.diag(predictive) > 0
        if np.any(residual[~active] != 0):
            raise ValueError("observation contradicts deterministic predictive support")
        if factor.shape[1] < np.count_nonzero(active):
            scales = np.sqrt(np.diag(predictive)[active])
            standardized_factor = factor[active] / scales[:, None]
            standardized_residual = residual[active] / scales
            support = (
                standardized_factor
                @ np.linalg.lstsq(standardized_factor, standardized_residual, rcond=None)[0]
            )
            if np.linalg.norm(support - standardized_residual) > 1e-10 * max(
                float(np.linalg.norm(standardized_residual)), np.finfo(float).tiny
            ):
                raise ValueError("observation contradicts deterministic predictive support")
        remaining = np.eye(len(self.mean)) - gain @ action.design
        _, prior_scales, prior_root = _scaled_covariance(self.covariance, len(self.mean))
        _, noise_scales, noise_root = _scaled_covariance(
            action.noise_covariance, action.design.shape[0]
        )
        # Joseph update as Gram products avoids negative roundoff variances when
        # an exact observation removes a rank-deficient prior's uncertainty.
        remaining_root = remaining @ (prior_scales[:, None] * prior_root)
        observed_root = gain @ (noise_scales[:, None] * noise_root)
        covariance = remaining_root @ remaining_root.T + observed_root @ observed_root.T
        return GaussianPopulation(
            self.candidate_ids,
            self.mean + gain @ residual,
            covariance,
            self.decision_matrix,
            self.representation,
            self.model_spec,
            self.version + 1,
            (*self.observation_ids, observation.observation_id),
            (*self.queried_candidates, action.candidate_id),
            self.lineage_id,
        )

    def in_delta_coordinates(
        self, actions: Sequence[QueryAction]
    ) -> tuple[GaussianPopulation, list[QueryAction]]:
        """Invertible [V0,V1,...] -> [V0,D1,...], retaining baseline information.

        This is equivalent to the absolute model, unlike dropping the baseline
        and compressing the paired observation to one difference.
        """
        if self.representation != "absolute":
            raise ValueError("coordinate transform requires absolute model")
        transform = self.decision_matrix.copy()
        transform[0, 0] = 1
        inverse = np.linalg.solve(transform, np.eye(len(transform)))
        model = GaussianPopulation(
            self.candidate_ids,
            transform @ self.mean,
            transform @ self.covariance @ transform.T,
            self.decision_matrix @ inverse,
            "delta_with_baseline",
            self.model_spec,
            self.version,
            self.observation_ids,
            self.queried_candidates,
            self.lineage_id,
        )
        mapped = [
            QueryAction(
                a.candidate_id, a.design @ inverse, a.noise_covariance, a.cost, a.observation_kind
            )
            for a in actions
        ]
        return model, mapped


def _quadrature(means: np.ndarray, perturbation: np.ndarray, order: int) -> float:
    rank = perturbation.shape[1]
    if rank == 0:
        return 0.0
    nodes, weights = hermgauss(order)
    indices = np.array(list(itertools.product(range(order), repeat=rank)))
    draws = math.sqrt(2) * nodes[indices]
    probabilities = np.prod(weights[indices] / math.sqrt(math.pi), axis=1)
    values = np.max(means[:, None] + perturbation @ draws.T, axis=0)
    return float(probabilities @ (values - np.max(means)))


def score_queries(
    model: GaussianPopulation,
    actions: Sequence[QueryAction],
    *,
    quadrature_order: int = 24,
    integration: str = "quadrature",
    mc_samples: int = 1024,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Signed KG estimates on expected latent improvement, normalized by cost.

    Gauss-Hermite order n vs 2n differences are diagnostics, not rigorous error
    bounds. MC standard errors are diagnostics too. Neither certifies stopping.
    """
    if quadrature_order < 2 or integration not in {"quadrature", "mc"} or mc_samples < 2:
        raise ValueError("invalid integration configuration")
    if len({a.candidate_id for a in actions}) != len(actions):
        raise ValueError("one query likelihood per candidate is required")
    scores = []
    rng = np.random.default_rng(seed)
    for action in sorted(actions, key=lambda a: a.candidate_id):
        factor, gain, _ = model._predictive(action)
        perturbation = model.decision_matrix @ gain @ factor
        means = model.decision_means
        if integration == "quadrature":
            coarse = _quadrature(means, perturbation, quadrature_order)
            kg = _quadrature(means, perturbation, 2 * quadrature_order)
            error, standard_error = abs(kg - coarse), None
        else:
            draws = rng.normal(size=(perturbation.shape[1], mc_samples))
            increments = np.max(means[:, None] + perturbation @ draws, axis=0) - np.max(means)
            kg = float(np.mean(increments))
            error, standard_error = None, float(np.std(increments, ddof=1) / math.sqrt(mc_samples))
        scores.append(
            {
                "candidate_id": action.candidate_id,
                "kg": kg,
                "acquisition": kg / action.cost,
                "cost": action.cost,
                "integration": integration,
                "quadrature_order": 2 * quadrature_order if integration == "quadrature" else None,
                "integration_error_estimate": error,
                "integration_error_is_bound": False,
                "mc_standard_error": standard_error,
                "mc_samples": mc_samples if integration == "mc" else None,
                "posterior_version": model.version,
                "current_value": float(np.max(means)),
                "representation": model.representation,
                "model_spec": model.model_spec,
            }
        )
    return sorted(scores, key=lambda score: (-score["acquisition"], score["candidate_id"]))


@dataclass(frozen=True)
class PopulationRoundResult:
    selected_candidate_id: str
    selected_delta_estimate: float
    posterior: GaussianPopulation
    query_count: int
    hf_cost: float
    audit_trace: tuple[dict[str, Any], ...]
    stop_reason: str
    confidence_certified: bool = False


def run_population_round(
    model: GaussianPopulation,
    actions: Sequence[QueryAction],
    observe: Callable[[QueryAction, int], PopulationObservation],
    *,
    max_queries: int,
    cost_budget: float,
    quadrature_order: int = 24,
    stopping: PosteriorStopRule | None = None,
) -> PopulationRoundResult:
    """Sequential paid observations; repeated candidate queries remain eligible.

    Action cost is a prespecified per-query spending ceiling. Returned actual
    cost may be lower; exceeding it fails explicitly. This loop does not issue
    a PAC or calibrated-confidence certificate and does not stop on finite KG.
    An optional stop rule uses analytic bounds conditional on the supplied joint
    Gaussian model. Calibration of that model remains a separate readiness gate.
    """
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries < 0:
        raise ValueError("max_queries must be a nonnegative integer")
    if not math.isfinite(cost_budget) or cost_budget < 0:
        raise ValueError("cost_budget must be finite and nonnegative")
    posterior = model
    count, spent = 0, 0.0
    trace = []

    def record(event: str, **details: Any) -> None:
        trace.append(
            {
                "event": event,
                "query_count": count,
                "posterior_version": posterior.version,
                "cumulative_hf_cost": spent,
                "confidence_certified": False,
                **details,
            }
        )

    scores = score_queries(posterior, actions, quadrature_order=quadrature_order)
    bounds = decision_bounds(posterior)
    record("rescore", scores=scores, **bounds)
    stopped_on_model = False
    while count < max_queries:
        if stopping is not None and stopping.satisfied(bounds, actions):
            stopped_on_model = True
            break
        affordable = [score for score in scores if score["cost"] <= cost_budget - spent + 1e-12]
        if not affordable:
            break
        action = next(a for a in actions if a.candidate_id == affordable[0]["candidate_id"])
        replicate = posterior.queried_candidates.count(action.candidate_id)
        record(
            "query",
            candidate_id=action.candidate_id,
            replicate_index=replicate,
            declared_cost=action.cost,
        )
        observation = observe(action, replicate)
        if observation.actual_cost > action.cost + 1e-12:
            raise ValueError("actual query cost exceeds declared spending ceiling")
        count += 1
        spent += observation.actual_cost
        record(
            "observe",
            candidate_id=action.candidate_id,
            observation_id=observation.observation_id,
            values=list(observation.values),
            actual_cost=observation.actual_cost,
        )
        before = posterior.version
        posterior = posterior.condition(action, observation)
        record(
            "condition",
            posterior_version_before=before,
            posterior_version_after=posterior.version,
            observation_id=observation.observation_id,
        )
        scores = score_queries(posterior, actions, quadrature_order=quadrature_order)
        bounds = decision_bounds(posterior)
        record("rescore", scores=scores, **bounds)
    reason = (
        "model_conditional_probability_and_voi"
        if stopped_on_model
        else "query_budget"
        if count == max_queries
        else "cost_budget"
    )
    record(
        "stop",
        stop_reason=reason,
        model_conditional_stop_criteria_met=stopped_on_model,
        max_acquisition_upper=stopping.acquisition_upper(bounds, actions) if stopping else None,
        **bounds,
    )
    selected = posterior.selected_candidate_id
    estimate = float(np.max(posterior.decision_means))
    record("select", candidate_id=selected, selected_delta_estimate=estimate)
    return PopulationRoundResult(selected, estimate, posterior, count, spent, tuple(trace), reason)
