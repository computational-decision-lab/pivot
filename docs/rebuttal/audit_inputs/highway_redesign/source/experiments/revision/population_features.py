"""Feature-transfer delta posterior for repeated-sampling E3C/E5C rounds.

Expected improvement is proxy + features @ beta + a persistent per-candidate
discrepancy, including uncertainty in the observed proxy mean. Rollout noise
is supplied separately to each query likelihood.
Only the coefficient marginal transfers to a genuinely new candidate set.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .population import GaussianPopulation, QueryAction, _array, _psd


@dataclass(frozen=True)
class ObservableTransition:
    candidate_id: str
    delta_proxy: float
    features: tuple[float, ...]
    query_cost: float
    proxy_mean_variance: float = 0.0

    def __post_init__(self) -> None:
        features = tuple(float(value) for value in self.features)
        if not self.candidate_id or self.candidate_id == "incumbent":
            raise ValueError("candidate identifier must be nonempty and distinct from incumbent")
        if not features or not all(math.isfinite(value) for value in features):
            raise ValueError("observable features must be finite and nonempty")
        if not math.isfinite(self.delta_proxy):
            raise ValueError("proxy delta must be finite")
        if not math.isfinite(self.query_cost) or self.query_cost <= 0:
            raise ValueError("query spending ceiling must be positive and finite")
        if (
            isinstance(self.proxy_mean_variance, bool)
            or not isinstance(self.proxy_mean_variance, Real)
            or not math.isfinite(self.proxy_mean_variance)
            or self.proxy_mean_variance < 0
        ):
            raise ValueError("proxy mean variance must be numeric, finite and nonnegative")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "proxy_mean_variance", float(self.proxy_mean_variance))


@dataclass(frozen=True)
class DeltaFeaturePrior:
    mean: Any
    covariance: Any
    model_spec: str
    version: int = 0
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mean = _array(self.mean, 1)
        if not mean.size or not self.model_spec:
            raise ValueError("coefficient prior and model specification are required")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _psd(self.covariance, len(mean)))
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))

    def for_round(
        self, rows: Sequence[ObservableTransition], *, persistent_variance: float
    ) -> FeatureRound:
        rows = tuple(rows)
        if not rows or any(len(row.features) != len(self.mean) for row in rows):
            raise ValueError("candidate feature dimension differs from coefficient prior")
        if not math.isfinite(persistent_variance) or persistent_variance < 0:
            raise ValueError("persistent discrepancy variance must be finite and nonnegative")
        dimension, count = len(self.mean), len(rows)
        # The exact constant carries observed proxy means; their uncertainty
        # belongs to the persistent arm effect, shared by all its HF queries.
        mean = np.concatenate(([1.0], self.mean, np.zeros(count)))
        covariance = np.zeros((len(mean), len(mean)))
        covariance[1 : 1 + dimension, 1 : 1 + dimension] = self.covariance
        covariance[1 + dimension :, 1 + dimension :] = np.diag(
            [persistent_variance + row.proxy_mean_variance for row in rows]
        )
        decision = np.zeros((count + 1, len(mean)))
        for index, row in enumerate(rows):
            decision[index + 1, 0] = row.delta_proxy
            decision[index + 1, 1 : 1 + dimension] = row.features
            decision[index + 1, 1 + dimension + index] = 1
        lineage_id = hashlib.sha256(
            json.dumps(
                {
                    "prior_mean": self.mean.tolist(),
                    "prior_covariance": self.covariance.tolist(),
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
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        model = GaussianPopulation(
            tuple(row.candidate_id for row in rows),
            mean,
            covariance,
            decision,
            "delta_feature_transfer",
            self.model_spec,
            self.version,
            self.observation_ids,
            (),
            lineage_id,
        )
        return FeatureRound(model, rows, dimension)


@dataclass(frozen=True)
class FeatureRound:
    model: GaussianPopulation
    rows: tuple[ObservableTransition, ...]
    coefficient_count: int

    def query(self, candidate_id: str, *, noise_variance: float) -> QueryAction:
        index = self.model.candidate_ids.index(candidate_id)
        return QueryAction(
            candidate_id,
            self.model.decision_matrix[[index + 1]],
            [[noise_variance]],
            self.rows[index].query_cost,
            "delta",
        )

    def next_prior(self, posterior: GaussianPopulation) -> DeltaFeaturePrior:
        if (
            posterior.candidate_ids != self.model.candidate_ids
            or posterior.representation != self.model.representation
            or posterior.model_spec != self.model.model_spec
            or not self.model.lineage_id
            or posterior.lineage_id != self.model.lineage_id
            or not np.array_equal(posterior.decision_matrix, self.model.decision_matrix)
            or posterior.observation_ids[: len(self.model.observation_ids)]
            != self.model.observation_ids
        ):
            raise ValueError("posterior does not belong to this feature round")
        indices = slice(1, 1 + self.coefficient_count)
        return DeltaFeaturePrior(
            posterior.mean[indices],
            posterior.covariance[indices, indices],
            self.model.model_spec,
            posterior.version,
            posterior.observation_ids,
        )
