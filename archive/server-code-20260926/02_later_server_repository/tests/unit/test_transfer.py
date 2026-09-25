from __future__ import annotations

import pytest

from pivot.transfer.differential import DifferentialModel, GradientBoostedDifferentialModel
from pivot.transfer.global_value import GlobalValueModel, spearman_rank_correlation
from pivot.transfer.sampling import stratified_transition_sample


def _row(identifier: str, value: float, proxy: float) -> dict[str, object]:
    return {
        "transition_id": identifier,
        "candidate_parameters": {"intensity": value},
        "incumbent_parameters": {"intensity": 0.2},
        "candidate_policy_id": identifier,
        "delta_proxy": proxy,
        "delta_true": proxy - 0.4 * value,
        "true_candidate_value": 2.0 + value,
        "true_incumbent_value": 2.2,
        "response_strength": 0.5,
        "update_footprint": abs(value - 0.2),
        "candidate_index": 0,
        "footprint_components": {"mean_kl": 0.1, "action_shift": 0.2},
    }


def test_global_value_and_differential_models_fit_without_leaking_true_delta() -> None:
    rows = [_row(f"t{i}", 0.3 + i * 0.1, 0.5 + i * 0.2) for i in range(4)]
    global_model = GlobalValueModel()
    global_model.fit([row["candidate_parameters"] for row in rows], [row["true_candidate_value"] for row in rows])
    assert global_model.predict_row(rows[0]) == pytest.approx(2.3, abs=0.1)
    model = DifferentialModel()
    model.fit(rows[:3], [float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows[:3]])
    prediction = model.predict_correction(rows[3])
    assert prediction.predicted_delta == pytest.approx(float(rows[3]["delta_true"]), abs=0.4)
    assert model.hf_budget == 3
    boosted = GradientBoostedDifferentialModel(n_estimators=10)
    boosted.fit(rows[:3], [float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows[:3]])
    assert boosted.predict_correction(rows[3]).standard_deviation >= 0.0


def test_rank_correlation_handles_ties() -> None:
    assert spearman_rank_correlation([1.0, 2.0, 2.0], [1.0, 3.0, 4.0]) > 0.8


def test_stratified_sample_covers_available_response_levels() -> None:
    rows = [
        {"transition_id": f"{response}-{index}", "response_strength": response, "optimization_strength": 1.0}
        for response in (0.0, 0.5, 1.0)
        for index in range(3)
    ]
    sample = stratified_transition_sample(rows, budget=3)
    assert {row["response_strength"] for row in sample} == {0.0, 0.5, 1.0}
