from __future__ import annotations

import numpy as np

from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    score_pivot_voi,
    select_pivot_voi,
)
from pivot.algorithms.pivot import run_pivot_round, run_pivot_voi_round


def test_round_harness_records_voi_scores_and_posterior_metadata() -> None:
    rows = [
        {"transition_id": "a", "features": [1.0, 0.0], "delta_proxy": 1.0, "hf_query_cost": 1.0},
        {"transition_id": "b", "features": [0.0, 1.0], "delta_proxy": 0.9, "hf_query_cost": 1.0},
    ]
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=0.1)
    posterior.fit(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0.0, 0.0]))
    scores = score_pivot_voi(rows, posterior, seed=3, fantasies=8, posterior_samples=16)

    def hf(row):
        return {"delta_true": float(row["delta_proxy"]) - 0.2, "hf_query_cost": 1.0}

    result = run_pivot_round(
        incumbent=None,
        candidates=rows,
        proxy=None,
        hf=hf,
        acquisition=select_pivot_voi,
        budget=1,
        model=posterior,
        acquisition_kwargs={"seed": 3, "fantasies": 8, "posterior_samples": 16},
        acquisition_method="PIVOT-VOI",
        acquisition_scores=scores,
        posterior_version="bayesian-linear-v1",
    )

    assert result.acquisition_method == "PIVOT-VOI"
    assert result.posterior_version == "bayesian-linear-v1"
    assert len(result.acquisition_scores) == 2
    assert result.query_ledger[0]["paired"] is True


def test_voi_round_can_stop_without_consuming_the_remaining_budget() -> None:
    rows = [
        {"transition_id": "a", "features": [1.0, 0.0], "delta_proxy": 3.0, "hf_query_cost": 1.0},
        {"transition_id": "b", "features": [0.0, 1.0], "delta_proxy": 0.0, "hf_query_cost": 1.0},
    ]
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=0.01)
    posterior.fit(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0.0, 0.0]))

    result = run_pivot_voi_round(
        incumbent=None,
        candidates=rows,
        hf=lambda row: {"delta_true": float(row["delta_proxy"])},
        posterior=posterior,
        max_budget=1,
        seed=2,
        delta=0.2,
        eta=100.0,
        fantasies=8,
        posterior_samples=16,
    )

    assert result.hf_budget == 0
    assert result.stop_reason in {"selection_probability", "evsi_per_cost"}
