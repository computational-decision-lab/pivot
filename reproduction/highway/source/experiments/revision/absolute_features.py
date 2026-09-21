"""DEV joint coefficient/baseline transfer for absolute paired observations.

Candidate improvements are proxy + features @ beta + a persistent arm effect.
The incumbent's absolute value and its covariance with beta survive a round.
New candidate effects are independent of that state and of one another; callers
must supply genuinely new policies rather than reuse discarded arm effects.
These structural Gaussian assumptions are not empirical calibration claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .population import GaussianPopulation, QueryAction, _array, _psd, _scaled_covariance
from .population_features import DeltaFeaturePrior, ObservableTransition


def _identity(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class AbsoluteFeaturePrior:
    """Joint state ordered as [beta..., V_incumbent], with a transfer lineage."""

    mean: Any
    covariance: Any
    model_spec: str
    version: int = 0
    observation_ids: tuple[str, ...] = ()
    lineage_id: str = ""

    def __post_init__(self) -> None:
        mean = _array(self.mean, 1)
        if len(mean) < 2 or not isinstance(self.model_spec, str) or not self.model_spec:
            raise ValueError("coefficient/baseline prior and model specification are required")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise ValueError("prior version must be a nonnegative integer")
        observation_ids = tuple(self.observation_ids)
        if not all(isinstance(value, str) and value for value in observation_ids) or len(
            set(observation_ids)
        ) != len(observation_ids):
            raise ValueError("observation lineage must contain unique nonempty identifiers")
        if not isinstance(self.lineage_id, str):
            raise TypeError("prior lineage identifier must be a string")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _psd(self.covariance, len(mean)))
        object.__setattr__(self, "observation_ids", observation_ids)

    @classmethod
    def from_delta_prior(
        cls,
        delta_prior: DeltaFeaturePrior,
        *,
        baseline_mean: float,
        baseline_variance: float,
    ) -> AbsoluteFeaturePrior:
        """Initialize with the explicit assumption that baseline and beta are independent."""
        if not math.isfinite(baseline_mean):
            raise ValueError("baseline mean must be finite")
        if not math.isfinite(baseline_variance) or baseline_variance < 0:
            raise ValueError("baseline variance must be finite and nonnegative")
        mean = np.append(delta_prior.mean, baseline_mean)
        covariance = np.zeros((len(mean), len(mean)))
        covariance[:-1, :-1] = delta_prior.covariance
        covariance[-1, -1] = baseline_variance
        model_spec = json.dumps(
            {
                "representation": "absolute_feature_transfer",
                "stage": "DEV",
                "calibration_claim": False,
                "initialization_assumption": "independent_initial_baseline",
                "coefficient_model_spec": delta_prior.model_spec,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(
            mean,
            covariance,
            model_spec,
            delta_prior.version,
            delta_prior.observation_ids,
        )

    def for_round(
        self, rows: Sequence[ObservableTransition], *, persistent_variance: float
    ) -> AbsoluteFeatureRound:
        rows = tuple(rows)
        dimension, count = len(self.mean) - 1, len(rows)
        if not rows or any(len(row.features) != dimension for row in rows):
            raise ValueError("candidate feature dimension differs from coefficient prior")
        if not math.isfinite(persistent_variance) or persistent_variance < 0:
            raise ValueError("persistent discrepancy variance must be finite and nonnegative")
        # The exact constant carries observed proxy offsets. Baseline value and
        # beta retain their joint covariance; each new arm gets its own effect,
        # including its fixed proxy mean error rather than fresh query noise.
        mean = np.concatenate(([1.0], self.mean, np.zeros(count)))
        covariance = np.zeros((len(mean), len(mean)))
        covariance[1 : dimension + 2, 1 : dimension + 2] = self.covariance
        covariance[dimension + 2 :, dimension + 2 :] = np.diag(
            [persistent_variance + row.proxy_mean_variance for row in rows]
        )
        decision = np.zeros((count + 1, len(mean)))
        for index, row in enumerate(rows):
            decision[index + 1, 0] = row.delta_proxy
            decision[index + 1, 1 : dimension + 1] = row.features
            decision[index + 1, dimension + 2 + index] = 1
        lineage_id = _identity(
            {
                "representation": "absolute_feature_transfer",
                "prior_mean": self.mean.tolist(),
                "prior_covariance": self.covariance.tolist(),
                "prior_lineage_id": self.lineage_id,
                "model_spec": self.model_spec,
                "version": self.version,
                "observation_ids": self.observation_ids,
                "rows": [
                    {
                        "candidate_id": row.candidate_id,
                        "delta_proxy": row.delta_proxy,
                        "features": row.features,
                        "query_cost": row.query_cost,
                        "proxy_mean_variance": row.proxy_mean_variance,
                    }
                    for row in rows
                ],
                "persistent_variance": persistent_variance,
            }
        )
        model = GaussianPopulation(
            tuple(row.candidate_id for row in rows),
            mean,
            covariance,
            decision,
            "absolute_feature_transfer",
            self.model_spec,
            self.version,
            self.observation_ids,
            (),
            lineage_id,
        )
        return AbsoluteFeatureRound(model, rows, dimension)


@dataclass(frozen=True)
class AbsoluteFeatureRound:
    model: GaussianPopulation
    rows: tuple[ObservableTransition, ...]
    coefficient_count: int

    def query(self, candidate_id: str, *, noise_covariance: Any) -> QueryAction:
        """Observe [V_incumbent, V_candidate] with the supplied paired noise."""
        if candidate_id not in self.model.candidate_ids:
            raise ValueError("query candidate does not belong to this feature round")
        index = self.model.candidate_ids.index(candidate_id)
        design = np.zeros((2, len(self.model.mean)))
        design[:, self.coefficient_count + 1] = 1
        design[1] += self.model.decision_matrix[index + 1]
        return QueryAction(
            candidate_id,
            design,
            noise_covariance,
            self.rows[index].query_cost,
            "absolute_pair",
        )

    def next_prior(
        self,
        posterior: GaussianPopulation,
        selected_candidate_id: str,
        *,
        exact_selected_value: float | None = None,
    ) -> AbsoluteFeaturePrior:
        """Project to [beta, V_selected] without discarding cross covariance.

        ``exact_selected_value`` must come from an actual exact paired purchase
        for the selected policy in this round. The caller retains that evidence.
        It is never inferred from a small posterior variance: legal small modes
        survive unless the caller supplies this explicit deterministic constraint.
        """
        initial_ids, observation_ids = self.model.observation_ids, posterior.observation_ids
        new_observation_count = len(observation_ids) - len(initial_ids)
        if (
            posterior.candidate_ids != self.model.candidate_ids
            or posterior.representation != self.model.representation
            or posterior.model_spec != self.model.model_spec
            or not self.model.lineage_id
            or posterior.lineage_id != self.model.lineage_id
            or not np.array_equal(posterior.decision_matrix, self.model.decision_matrix)
            or observation_ids[: len(initial_ids)] != initial_ids
            or new_observation_count < 0
            or len(set(observation_ids)) != len(observation_ids)
            or not all(isinstance(value, str) and value for value in observation_ids)
            or posterior.version != self.model.version + new_observation_count
            or len(posterior.queried_candidates) != new_observation_count
            or any(value not in self.model.candidate_ids for value in posterior.queried_candidates)
            or posterior.mean[0] != 1
            or np.any(posterior.covariance[0] != 0)
        ):
            raise ValueError("posterior does not belong to this absolute feature round")
        if selected_candidate_id not in ("incumbent", *self.model.candidate_ids):
            raise ValueError("selected candidate does not belong to this feature round")
        projection = np.zeros((self.coefficient_count + 1, len(posterior.mean)))
        projection[:-1, 1 : self.coefficient_count + 1] = np.eye(self.coefficient_count)
        projection[-1, self.coefficient_count + 1] = 1
        if selected_candidate_id != "incumbent":
            index = self.model.candidate_ids.index(selected_candidate_id) + 1
            projection[-1] += self.model.decision_matrix[index]
        mean = projection @ posterior.mean
        # An exact pair can make V_selected deterministic. Project a square root
        # rather than subtract nearly equal covariance entries at that boundary.
        _, scales, root = _scaled_covariance(posterior.covariance, len(posterior.mean))
        projected_root = projection @ (scales[:, None] * root)
        covariance = projected_root @ projected_root.T
        if exact_selected_value is not None:
            exact_selected_value = float(exact_selected_value)
            if (
                not math.isfinite(exact_selected_value)
                or not posterior.queried_candidates
                or (
                    selected_candidate_id != "incumbent"
                    and selected_candidate_id not in posterior.queried_candidates
                )
            ):
                raise ValueError("exact selected value requires a matching finite observation")
            # Only tolerate arithmetic error in the selected-value projection.
            # This check does not decide whether the value is exact; the actual
            # exact-purchase evidence supplied by the caller decides that.
            scale = max(
                float(np.abs(projection[-1]) @ np.abs(posterior.mean)),
                abs(exact_selected_value),
                np.finfo(float).tiny,
            )
            if abs(mean[-1] - exact_selected_value) > 64 * np.finfo(float).eps * scale:
                raise ValueError("exact selected value contradicts the projected posterior mean")
            mean[-1] = exact_selected_value
            covariance[-1, :] = 0
            covariance[:, -1] = 0
        lineage_id = _identity(
            {
                "round_lineage_id": self.model.lineage_id,
                "selected_candidate_id": selected_candidate_id,
                "version": posterior.version,
                "observation_ids": observation_ids,
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
                "exact_selected_value": exact_selected_value,
            }
        )
        return AbsoluteFeaturePrior(
            mean,
            covariance,
            self.model.model_spec,
            posterior.version,
            observation_ids,
            lineage_id,
        )
