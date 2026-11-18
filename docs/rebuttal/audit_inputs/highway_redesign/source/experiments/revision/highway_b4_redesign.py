"""Frozen candidate and analysis helpers for the HighwayEnv B=4 redesign."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.revision.highway_calibration import CalibrationModel, GaussianPosterior
from pivot.core.policy import Policy

ACTION_PARAMETERS: dict[str, dict[str, float]] = {
    "slower": {"intensity": -0.8, "bias": 0.0},
    "lane_left": {"intensity": 0.0, "bias": -0.8},
    "lane_right": {"intensity": 0.0, "bias": 0.8},
    "faster": {"intensity": 0.8, "bias": 0.0},
}
ACTION_LABELS = {
    "slower": "SLOWER",
    "lane_left": "LANE_LEFT",
    "lane_right": "LANE_RIGHT",
    "faster": "FASTER",
}
ACTION_ORDER = ("slower", "lane_left", "lane_right", "faster")
PHASE_ORDER = ("early", "late")


def initial_policy() -> Policy:
    return Policy.from_mapping({"intensity": 0.18, "bias": 0.0}, metadata={"v9": "initial"})


def discovery_panel_b4() -> dict[str, Policy]:
    """Return eight distinct scheduled candidates in frozen order."""

    panel: dict[str, Policy] = {}
    for phase in PHASE_ORDER:
        for action in ACTION_ORDER:
            candidate_id = f"{phase}_{action}"
            panel[candidate_id] = Policy.from_mapping(
                ACTION_PARAMETERS[action],
                metadata={
                    "highway_action": ACTION_LABELS[action],
                    "intervention_phase": phase,
                    "intervention_start": "0" if phase == "early" else "10",
                    "intervention_end": "10" if phase == "early" else "20",
                },
            )
    return panel


def candidate_panel() -> dict[str, Policy]:
    """Alias used by the runner and tests."""

    return {"incumbent": initial_policy(), **discovery_panel_b4()}


def validate_panel(panel: Mapping[str, Policy]) -> None:
    if list(panel) != ["incumbent", *[f"{phase}_{action}" for phase in PHASE_ORDER for action in ACTION_ORDER]]:
        raise ValueError("B=4 panel order does not match the preregistration")
    candidate_ids = list(panel)[1:]
    if len(set(candidate_ids)) != 8:
        raise ValueError("B=4 panel contains duplicate candidate IDs")
    schedules = {
        (
            tuple(policy.parameters.items()),
            policy.metadata.get("highway_action"),
            policy.metadata.get("intervention_phase"),
        )
        for policy in (panel[candidate] for candidate in candidate_ids)
    }
    if len(schedules) != 8:
        raise ValueError("B=4 panel contains duplicate policy schedules")
    if any(policy.metadata.get("intervention_phase") not in PHASE_ORDER for policy in panel.values() if policy is not panel["incumbent"]):
        raise ValueError("B=4 panel has an unknown intervention phase")


def enumerate_subsets(candidate_ids: Sequence[str], budget: int) -> tuple[tuple[str, ...], ...]:
    ids = tuple(str(candidate) for candidate in candidate_ids)
    if not 1 <= budget <= len(ids):
        raise ValueError("budget must be within the candidate panel")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    return tuple(itertools.combinations(ids, budget))


def _selector(posterior: GaussianPosterior) -> str:
    return posterior.select()


def condition_on_subset(
    rows: Sequence[Mapping[str, Any]],
    model: CalibrationModel,
    truth_values: Mapping[str, float],
    subset: Sequence[str],
) -> GaussianPosterior:
    posterior = GaussianPosterior(rows, model)
    for candidate_id in subset:
        if candidate_id not in truth_values:
            raise KeyError(candidate_id)
        posterior.observe(candidate_id, float(truth_values[candidate_id]))
    return posterior


def uniform_expected_selection(
    rows: Sequence[Mapping[str, Any]],
    model: CalibrationModel,
    truth_values: Mapping[str, float],
    budget: int,
) -> dict[str, Any]:
    """Enumerate all Uniform subsets and average their actor-world ISR."""

    candidate_ids = tuple(model.candidate_ids)
    best = max([0.0, *[float(truth_values[candidate]) for candidate in candidate_ids]])
    subset_rows: list[dict[str, Any]] = []
    for subset_index, subset in enumerate(enumerate_subsets(candidate_ids, budget)):
        posterior = condition_on_subset(rows, model, truth_values, subset)
        selected = _selector(posterior)
        selected_value = 0.0 if selected == "incumbent" else float(truth_values[selected])
        subset_rows.append(
            {
                "subset_index": subset_index,
                "queried_candidates": list(subset),
                "selected_candidate": selected,
                "selected_actor_delta": selected_value,
                "actor_best_delta": best,
                "ISR": best - selected_value,
            }
        )
    return {
        "budget": budget,
        "subset_count": len(subset_rows),
        "mean_ISR": float(np.mean([row["ISR"] for row in subset_rows])),
        "rows": subset_rows,
    }


def common_normal_grid(sample_count: int, dimension: int, *, seed: int = 20260921) -> np.ndarray:
    if sample_count <= 0 or dimension <= 0:
        raise ValueError("sample_count and dimension must be positive")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((sample_count, dimension))


def _deterministic_simple_regret(posterior: GaussianPosterior, normals: np.ndarray) -> float:
    covariance = (posterior.covariance + posterior.covariance.T) / 2.0
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    draws = posterior.means + normals @ root.T
    selected = posterior.select()
    selected_values = np.zeros(len(draws)) if selected == "incumbent" else draws[:, posterior._index[selected]]
    best = np.maximum(0.0, np.max(np.column_stack([np.zeros(len(draws)), draws]), axis=1))
    return float(np.mean(np.maximum(0.0, best - selected_values)))


def _deterministic_evsi(
    posterior: GaussianPosterior,
    candidate_id: str,
    normals: np.ndarray,
    fantasies: np.ndarray,
) -> float:
    current = _deterministic_simple_regret(posterior, normals)
    index = posterior._index[candidate_id]
    scale = float(max(0.0, posterior.covariance[index, index] + posterior.observation_variance[index])) ** 0.5
    if scale <= 0.0:
        return 0.0
    values = posterior.means[index] + scale * fantasies
    after = []
    for value in values:
        updated = posterior.clone()
        updated.observe(candidate_id, float(value))
        after.append(_deterministic_simple_regret(updated, normals))
    return max(0.0, current - float(np.mean(after)))


def _normal_cdf(value: float) -> float:
    if value == float("inf"):
        return 1.0
    if value == float("-inf"):
        return 0.0
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def expected_max_affine_normal(intercepts: Sequence[float], slopes: Sequence[float]) -> float:
    """Return E[max_i(a_i + b_i Z)] for one standard normal Z.

    Sorting by slope reduces the maximum to an upper envelope.  Each surviving
    line dominates on one interval, whose Gaussian integral is analytic.
    """

    if len(intercepts) != len(slopes) or not intercepts:
        raise ValueError("intercepts and slopes must have the same positive length")
    ordered = sorted(
        ((float(slope), float(intercept), index) for index, (intercept, slope) in enumerate(zip(intercepts, slopes, strict=True))),
        key=lambda item: (item[0], item[1], -item[2]),
    )
    unique: list[tuple[float, float, int]] = []
    for line in ordered:
        if unique and math.isclose(line[0], unique[-1][0], rel_tol=0.0, abs_tol=1e-15):
            if line[1] > unique[-1][1]:
                unique[-1] = line
        else:
            unique.append(line)
    hull: list[tuple[float, float, int]] = []
    starts: list[float] = []
    for line in unique:
        start = float("-inf")
        while hull:
            previous = hull[-1]
            start = (previous[1] - line[1]) / (line[0] - previous[0])
            if start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = float("-inf")
        hull.append(line)
        starts.append(start)
    expectation = 0.0
    for index, (slope, intercept, _) in enumerate(hull):
        lower = starts[index]
        upper = starts[index + 1] if index + 1 < len(starts) else float("inf")
        probability = _normal_cdf(upper) - _normal_cdf(lower)
        expectation += intercept * probability + slope * (_normal_pdf(lower) - _normal_pdf(upper))
    return float(expectation)


def exact_gaussian_kg_score(posterior: GaussianPosterior, candidate_id: str) -> float:
    """Exact one-step KG/EVSI for finite correlated Gaussian alternatives."""

    if candidate_id in posterior.observed:
        return 0.0
    if candidate_id not in posterior._index:
        raise KeyError(candidate_id)
    index = posterior._index[candidate_id]
    predictive_variance = float(
        posterior.covariance[index, index] + posterior.observation_variance[index]
    )
    if predictive_variance <= 0.0:
        return 0.0
    scale = math.sqrt(predictive_variance)
    intercepts = [0.0, *[float(value) for value in posterior.means]]
    slopes = [0.0, *[float(value) for value in posterior.covariance[:, index] / scale]]
    current_value = max(intercepts)
    expected_value = expected_max_affine_normal(intercepts, slopes)
    return max(0.0, expected_value - current_value)


def exact_kg_next(posterior: GaussianPosterior) -> tuple[str, float]:
    """Select the exact highest-KG unobserved candidate with fixed tie breaking."""

    available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
    if not available:
        raise ValueError("no candidates remain to query")
    scores = {candidate: exact_gaussian_kg_score(posterior, candidate) for candidate in available}
    selected = max(available, key=lambda candidate: (scores[candidate], -posterior._index[candidate]))
    return selected, float(scores[selected])


def deterministic_kg_choice(
    posterior: GaussianPosterior,
    budget: int,
    sample_count: int,
    *,
    fantasy_count: int | None = None,
    seed: int = 20260921,
) -> tuple[str, ...]:
    if not 1 <= budget <= len(posterior.candidate_ids):
        raise ValueError("budget must be within the candidate panel")
    fantasy_count = sample_count if fantasy_count is None else fantasy_count
    selected: list[str] = []
    current = posterior.clone()
    for _ in range(budget):
        candidate = deterministic_kg_next(
            current,
            sample_count=sample_count,
            fantasy_count=fantasy_count,
            seed=seed,
        )
        selected.append(candidate)
        # Acquisition is pre-decision; use the posterior mean as the common
        # fantasy observation to make the query sequence independent of test truth.
        current.observe(candidate, current.value(candidate))
    return tuple(selected)


def deterministic_kg_next(
    posterior: GaussianPosterior,
    *,
    sample_count: int,
    fantasy_count: int | None = None,
    seed: int = 20260921,
) -> str:
    """Select one query using a fixed common-normal grid."""

    fantasy_count = sample_count if fantasy_count is None else fantasy_count
    normals = common_normal_grid(sample_count, len(posterior.candidate_ids), seed=seed)
    fantasies = common_normal_grid(fantasy_count, 1, seed=seed + 1).ravel()
    available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
    if not available:
        raise ValueError("no candidates remain to query")
    scores = {
        candidate: _deterministic_evsi(posterior, candidate, normals, fantasies)
        for candidate in available
    }
    return max(available, key=lambda value: (scores[value], -posterior._index[value]))


__all__ = [
    "ACTION_ORDER",
    "PHASE_ORDER",
    "candidate_panel",
    "common_normal_grid",
    "condition_on_subset",
    "deterministic_kg_choice",
    "deterministic_kg_next",
    "discovery_panel_b4",
    "enumerate_subsets",
    "exact_gaussian_kg_score",
    "exact_kg_next",
    "expected_max_affine_normal",
    "uniform_expected_selection",
    "validate_panel",
]
