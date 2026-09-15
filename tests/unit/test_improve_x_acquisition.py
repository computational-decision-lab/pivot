from __future__ import annotations

from dataclasses import dataclass

from improve_x.acquisition import score_decision_preservation, select_pivot_x


@dataclass(frozen=True)
class Prediction:
    predicted_delta: float
    standard_deviation: float
    sign_change_probability: float = 0.0


class Model:
    def __init__(self, values: dict[str, Prediction]) -> None:
        self.values = values

    def predict_correction(self, row: dict[str, object]) -> Prediction:
        return self.values[str(row["transition_id"])]


def rows() -> list[dict[str, object]]:
    return [
        {"transition_id": "stable", "delta_proxy": 1.0, "hf_query_cost": 1.0},
        {"transition_id": "ambiguous", "delta_proxy": 0.9, "hf_query_cost": 1.0},
        {"transition_id": "expensive", "delta_proxy": 0.8, "hf_query_cost": 5.0},
    ]


def test_pivot_x_queries_candidates_likely_to_change_the_update_decision() -> None:
    model = Model(
        {
            "stable": Prediction(1.0, 0.01),
            "ambiguous": Prediction(0.95, 0.4),
            "expensive": Prediction(0.9, 0.8),
        }
    )

    selected = select_pivot_x(rows(), model, budget=1)

    assert selected == ["ambiguous"]


def test_pivot_x_scores_are_auditable_and_cost_normalized() -> None:
    model = Model(
        {
            "stable": Prediction(1.0, 0.01),
            "ambiguous": Prediction(0.95, 0.4),
            "expensive": Prediction(0.9, 0.8),
        }
    )

    scores = score_decision_preservation(rows(), model)

    assert [item["transition_id"] for item in scores] == ["ambiguous", "expensive", "stable"]
    assert all(0.0 <= float(item["decision_change_probability"]) <= 1.0 for item in scores)
    assert all(float(item["cost"]) > 0 for item in scores)
