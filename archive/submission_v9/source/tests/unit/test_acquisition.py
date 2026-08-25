from __future__ import annotations

from dataclasses import dataclass

from pivot.acquisition.footprint import select_largest_footprint
from pivot.acquisition.pivot import select_pivot
from pivot.acquisition.random import select_random
from pivot.acquisition.top_proxy import select_top_proxy
from pivot.acquisition.uncertainty import select_uncertainty


@dataclass
class Prediction:
    predicted_delta: float
    standard_deviation: float
    sign_change_probability: float


class Model:
    def predict_correction(self, candidate):
        return candidate["prediction"]

    def uncertainty(self, candidate):
        return candidate["prediction"].standard_deviation


def _candidates():
    return [
        {"transition_id": "top", "delta_proxy": 2.0, "update_footprint": 0.1, "cost": 1.0, "prediction": Prediction(1.9, 0.01, 0.0)},
        {"transition_id": "ambiguous", "delta_proxy": 1.0, "update_footprint": 0.8, "cost": 1.0, "prediction": Prediction(-0.1, 2.0, 0.8)},
        {"transition_id": "small", "delta_proxy": 0.5, "update_footprint": 0.4, "cost": 1.0, "prediction": Prediction(0.4, 0.1, 0.0)},
    ]


def test_all_selectors_obey_budget() -> None:
    candidates = _candidates()
    assert len(select_random(candidates, budget=2, seed=4)) == 2
    assert len(select_top_proxy(candidates, budget=2)) == 2
    assert len(select_largest_footprint(candidates, budget=2)) == 2
    assert len(select_uncertainty(candidates, Model(), budget=2)) == 2
    assert len(select_pivot(candidates, Model(), budget=2, cost_key="cost")) == 2


def test_pivot_prioritizes_decision_change_candidate() -> None:
    assert select_pivot(_candidates(), Model(), budget=1, cost_key="cost") == ["ambiguous"]
