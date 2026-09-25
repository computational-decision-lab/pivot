"""Calibrated paired-posterior acquisition for the Highway redesign.

The original Highway runner used a fixed uncertainty heuristic and selected a
whole query batch before observing any result.  This module keeps the runner
contract small while implementing the pieces required by the paper protocol:
a disjoint correction calibration cohort, a shrinkage joint Gaussian prior,
Gaussian conditioning after each paired observation, and sequential EVSI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id")
    if value is None:
        raise ValueError("candidate row requires candidate_id")
    return str(value)


def _candidate_index(row: Mapping[str, Any]) -> int:
    return int(row.get("candidate_index", 0))


def _project_psd(matrix: np.ndarray, *, floor: float = 1e-9) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


@dataclass(frozen=True)
class CalibrationModel:
    """A frozen candidate-level correction model fit without test outcomes."""

    candidate_ids: tuple[str, ...]
    actions: tuple[str, ...]
    correction_mean: tuple[float, ...]
    correction_covariance: tuple[tuple[float, ...], ...]
    observation_variance: tuple[float, ...]
    calibration_seed_count: int
    covariance_shrinkage: float
    observation_noise: str

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "actions": list(self.actions),
            "correction_mean": list(self.correction_mean),
            "correction_covariance": [list(row) for row in self.correction_covariance],
            "observation_variance": list(self.observation_variance),
            "calibration_seed_count": self.calibration_seed_count,
            "covariance_shrinkage": self.covariance_shrinkage,
            "observation_noise": self.observation_noise,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CalibrationModel:
        candidate_ids = tuple(str(value) for value in record["candidate_ids"])
        actions = tuple(str(value) for value in record["actions"])
        covariance = tuple(
            tuple(float(value) for value in row)
            for row in record["correction_covariance"]
        )
        if len(candidate_ids) != len(actions) or len(covariance) != len(candidate_ids):
            raise ValueError("calibration model dimensions do not agree")
        if any(len(row) != len(candidate_ids) for row in covariance):
            raise ValueError("calibration covariance is not square")
        return cls(
            candidate_ids=candidate_ids,
            actions=actions,
            correction_mean=tuple(float(value) for value in record["correction_mean"]),
            correction_covariance=covariance,
            observation_variance=tuple(float(value) for value in record["observation_variance"]),
            calibration_seed_count=int(record["calibration_seed_count"]),
            covariance_shrinkage=float(record["covariance_shrinkage"]),
            observation_noise=str(record["observation_noise"]),
        )


def fit_calibration_model(
    proxy_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    actor_deltas_by_seed: Mapping[int, Mapping[str, float]],
    *,
    covariance_shrinkage: float = 0.5,
    covariance_floor: float = 1e-8,
) -> CalibrationModel:
    """Fit a shrinkage Gaussian correction model on a sealed cohort.

    ``actor_deltas_by_seed`` is supplied only for the calibration cohort.  A
    correction is actor minus proxy, so the resulting model predicts the
    deployment delta from a fresh proxy row.  Highway rollouts are
    deterministic conditional on a seed; the observation variance is therefore
    a numerical floor rather than an invented stochastic measurement model.
    """

    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1]")
    seeds = sorted(set(proxy_by_seed).intersection(actor_deltas_by_seed))
    if len(seeds) < 2:
        raise ValueError("calibration requires at least two complete seeds")
    first = sorted(proxy_by_seed[seeds[0]], key=_candidate_index)
    candidate_ids = tuple(_candidate_id(row) for row in first)
    actions = tuple(str(row.get("highway_action", "")) for row in first)
    if not all(candidate_ids):
        raise ValueError("calibration candidates must have non-empty ids")
    corrections: list[list[float]] = []
    for seed in seeds:
        rows = sorted(proxy_by_seed[seed], key=_candidate_index)
        ids = tuple(_candidate_id(row) for row in rows)
        if ids != candidate_ids:
            raise ValueError("calibration candidate panel changed across seeds")
        truth = actor_deltas_by_seed[seed]
        if set(truth) != set(candidate_ids):
            raise ValueError("calibration actor deltas do not match candidate panel")
        corrections.append(
            [float(truth[candidate]) - float(row["delta_proxy"]) for candidate, row in zip(candidate_ids, rows, strict=True)]
        )

    matrix = np.asarray(corrections, dtype=float)
    mean = matrix.mean(axis=0)
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - covariance_shrinkage) * covariance + covariance_shrinkage * diagonal
    covariance = _project_psd(covariance, floor=covariance_floor)
    observation_variance = np.full(len(candidate_ids), covariance_floor, dtype=float)
    return CalibrationModel(
        candidate_ids=candidate_ids,
        actions=actions,
        correction_mean=tuple(float(value) for value in mean),
        correction_covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        observation_variance=tuple(float(value) for value in observation_variance),
        calibration_seed_count=len(seeds),
        covariance_shrinkage=float(covariance_shrinkage),
        observation_noise="deterministic_seed_conditioned_rollout_floor",
    )


class GaussianPosterior:
    """Posterior over deployment deltas with an explicit incumbent arm."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        model: CalibrationModel,
    ) -> None:
        ordered = sorted(rows, key=_candidate_index)
        ids = tuple(_candidate_id(row) for row in ordered)
        if ids != model.candidate_ids:
            raise ValueError("proxy panel does not match calibration model")
        self.rows = tuple(dict(row) for row in ordered)
        self.candidate_ids = ids
        self._index = {candidate: index for index, candidate in enumerate(ids)}
        proxy = np.asarray([float(row["delta_proxy"]) for row in ordered], dtype=float)
        self.means = proxy + np.asarray(model.correction_mean, dtype=float)
        self.covariance = _project_psd(
            np.asarray(model.correction_covariance, dtype=float), floor=1e-9
        )
        self.observation_variance = np.asarray(model.observation_variance, dtype=float)
        self.observed: dict[str, float] = {}

    def clone(self) -> GaussianPosterior:
        clone = object.__new__(GaussianPosterior)
        clone.rows = self.rows
        clone.candidate_ids = self.candidate_ids
        clone._index = self._index
        clone.means = self.means.copy()
        clone.covariance = self.covariance.copy()
        clone.observation_variance = self.observation_variance.copy()
        clone.observed = dict(self.observed)
        return clone

    def value(self, candidate_id: str) -> float:
        if candidate_id == "incumbent":
            return 0.0
        return float(self.means[self._index[candidate_id]])

    def uncertainty(self, candidate_id: str) -> float:
        if candidate_id == "incumbent":
            return 0.0
        return float(max(0.0, self.covariance[self._index[candidate_id], self._index[candidate_id]])) ** 0.5

    def observe(self, candidate_id: str, value: float) -> None:
        if candidate_id not in self._index:
            raise KeyError(candidate_id)
        if candidate_id in self.observed:
            raise ValueError(f"candidate already observed: {candidate_id}")
        index = self._index[candidate_id]
        covariance_row = self.covariance[index, :].copy()
        denominator = float(self.covariance[index, index] + self.observation_variance[index])
        if denominator <= 0.0:
            self.means[index] = float(value)
            self.covariance[index, :] = 0.0
            self.covariance[:, index] = 0.0
        else:
            gain = covariance_row / denominator
            self.means += gain * (float(value) - self.means[index])
            self.covariance -= np.outer(gain, covariance_row)
            self.means[index] = float(value)
            self.covariance[index, :] = 0.0
            self.covariance[:, index] = 0.0
        self.covariance = _project_psd(self.covariance, floor=1e-12)
        self.covariance[index, :] = 0.0
        self.covariance[:, index] = 0.0
        self.observed[candidate_id] = float(value)

    def select(self) -> str:
        # Keep the explicit no-update arm as the deterministic tie winner.
        options = [("incumbent", 0.0, 0)]
        options.extend(
            (candidate, self.value(candidate), -1 - _candidate_index(row))
            for row, candidate in zip(self.rows, self.candidate_ids, strict=True)
        )
        return max(options, key=lambda item: (item[1], item[2]))[0]

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        covariance = _project_psd(self.covariance, floor=0.0)
        return rng.multivariate_normal(self.means, covariance, size=count, check_valid="ignore")


def expected_simple_regret(
    posterior: GaussianPosterior,
    rng: np.random.Generator,
    *,
    samples: int = 128,
) -> float:
    draws = posterior.sample(rng, samples)
    selected = posterior.select()
    if selected == "incumbent":
        selected_values = np.zeros(samples, dtype=float)
    else:
        selected_values = draws[:, posterior._index[selected]]
    best = np.maximum(0.0, np.max(np.column_stack([np.zeros(samples), draws]), axis=1))
    return float(np.mean(np.maximum(0.0, best - selected_values)))


def expected_evsi(
    posterior: GaussianPosterior,
    candidate_id: str,
    rng: np.random.Generator,
    *,
    fantasies: int = 32,
    posterior_samples: int = 64,
) -> float:
    if candidate_id in posterior.observed:
        return 0.0
    if candidate_id not in posterior._index:
        raise KeyError(candidate_id)
    current = expected_simple_regret(posterior, rng, samples=posterior_samples)
    index = posterior._index[candidate_id]
    scale = float(max(0.0, posterior.covariance[index, index] + posterior.observation_variance[index])) ** 0.5
    if scale <= 0.0:
        return 0.0
    fantasies_regret: list[float] = []
    fantasy_values = rng.normal(posterior.means[index], scale, size=fantasies)
    for value in fantasy_values:
        updated = posterior.clone()
        updated.observe(candidate_id, float(value))
        fantasies_regret.append(expected_simple_regret(updated, rng, samples=posterior_samples))
    return max(0.0, current - float(np.mean(fantasies_regret)))


def choose_next(
    posterior: GaussianPosterior,
    rng: np.random.Generator,
    *,
    fantasies: int = 32,
    posterior_samples: int = 64,
) -> tuple[str, float]:
    available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
    if not available:
        raise ValueError("no candidates remain to query")
    scores = {
        candidate: expected_evsi(
            posterior,
            candidate,
            rng,
            fantasies=fantasies,
            posterior_samples=posterior_samples,
        )
        for candidate in available
    }
    selected = max(
        available,
        key=lambda candidate: (scores[candidate], -posterior._index[candidate]),
    )
    return selected, float(scores[selected])


__all__ = [
    "CalibrationModel",
    "GaussianPosterior",
    "choose_next",
    "expected_evsi",
    "expected_simple_regret",
    "fit_calibration_model",
]
