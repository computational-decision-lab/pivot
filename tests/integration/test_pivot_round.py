from __future__ import annotations

from improve_x.acquisition import select_pivot_x
from pivot.acquisition.pivot import select_pivot
from pivot.algorithms.pivot import run_pivot_round


class Model:
    def predict_correction(self, row):
        class Prediction:
            predicted_delta = float(row["delta_proxy"]) - (1.5 if row["transition_id"] == "ambiguous" else 0.0)
            standard_deviation = 1.0 if row["transition_id"] == "ambiguous" else 0.01
            sign_change_probability = 0.8 if row["transition_id"] == "ambiguous" else 0.0

        return Prediction()


def test_pivot_round_queries_exact_budget_and_records_selection_regret() -> None:
    rows = [
        {"transition_id": "top", "delta_proxy": 2.0, "hf_query_cost": 1.0},
        {"transition_id": "ambiguous", "delta_proxy": 1.0, "hf_query_cost": 1.0},
    ]

    def hf(row):
        return {"delta_true": -0.5 if row["transition_id"] == "ambiguous" else 0.5, "hf_query_cost": 1.0}

    result = run_pivot_round(
        incumbent=None,
        candidates=rows,
        proxy=None,
        hf=hf,
        acquisition=select_pivot,
        budget=1,
        model=Model(),
    )
    assert result.hf_budget == 1
    assert result.queried_ids == ("ambiguous",)
    assert result.selected_candidate_id == "top"
    assert result.selected_delta_estimate == 2.0
    # The unqueried incumbent candidate remains an estimate; regret is only
    # computable once an oracle/all-HF evaluation is available.
    assert result.update_selection_regret is None


def test_pivot_x_is_compatible_with_the_existing_round_harness() -> None:
    rows = [
        {"transition_id": "top", "delta_proxy": 2.0, "hf_query_cost": 1.0},
        {"transition_id": "ambiguous", "delta_proxy": 1.0, "hf_query_cost": 1.0},
    ]

    def hf(row):
        return {"delta_true": -0.5 if row["transition_id"] == "ambiguous" else 0.5, "hf_query_cost": 1.0}

    result = run_pivot_round(
        incumbent=None,
        candidates=rows,
        proxy=None,
        hf=hf,
        acquisition=select_pivot_x,
        budget=1,
        model=Model(),
    )

    assert result.hf_budget == 1
    assert result.queried_ids == ("ambiguous",)
