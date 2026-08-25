from __future__ import annotations

import numpy as np
import pytest

from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    expected_simple_regret,
    score_pivot_voi,
    select_pivot_voi,
    should_stop,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"transition_id": "a", "features": [1.0, 0.0], "delta_proxy": 1.0, "hf_query_cost": 1.0},
        {"transition_id": "b", "features": [0.0, 1.0], "delta_proxy": 0.95, "hf_query_cost": 1.0},
        {"transition_id": "c", "features": [1.0, 1.0], "delta_proxy": 0.2, "hf_query_cost": 2.0},
    ]


def test_posterior_predicts_and_conditions_on_paired_correction() -> None:
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=0.05)
    posterior.fit(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0.2, -0.1]))
    before = posterior.predict(np.asarray([[1.0, 0.0]]))[0]
    updated = posterior.condition(np.asarray([1.0, 0.0]), 0.8, observation_variance=0.01)

    assert updated.predict(np.asarray([[1.0, 0.0]]))[0] > before
    assert updated.predictive_variance(np.asarray([[1.0, 0.0]]))[0] < posterior.predictive_variance(np.asarray([[1.0, 0.0]]))[0]


def test_expected_simple_regret_uses_current_best_posterior_action() -> None:
    samples = np.asarray([[1.0, 0.5], [0.8, 0.7]])
    assert expected_simple_regret(samples, selected_index=0) == pytest.approx(0.0)
    assert expected_simple_regret(samples, selected_index=1) == pytest.approx(0.3)


def test_pivot_voi_returns_auditable_cost_normalized_scores() -> None:
    posterior = BayesianLinearDeltaPosterior(prior_precision=1.0, noise_variance=0.05)
    posterior.fit(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0.0, 0.0]))
    scores = score_pivot_voi(_rows(), posterior, seed=7, fantasies=32, posterior_samples=64)

    assert {item["transition_id"] for item in scores} == {"a", "b", "c"}
    assert all(float(item["evsi"]) >= -1e-12 for item in scores)
    assert all(float(item["acquisition"]) >= 0 for item in scores)
    assert select_pivot_voi(
        _rows(), posterior, budget=1, seed=7, fantasies=32, posterior_samples=64
    ) == [str(scores[0]["transition_id"])]


def test_stopping_rule_handles_confident_selection_and_low_voi() -> None:
    assert should_stop(selection_probability=0.96, max_acquisition=0.5, delta=0.05, eta=0.1) == (True, "selection_probability")
    assert should_stop(selection_probability=0.5, max_acquisition=0.05, delta=0.05, eta=0.1) == (True, "evsi_per_cost")
    assert should_stop(selection_probability=0.5, max_acquisition=0.5, delta=0.05, eta=0.1) == (False, None)
