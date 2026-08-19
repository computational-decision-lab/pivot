from __future__ import annotations

from experiments.e5_budget_frontier import _evaluate_method
from pivot.transfer.differential import DifferentialModel


def test_frontier_method_obeys_query_budget() -> None:
    rows = [
        {
            "transition_id": f"t{index}",
            "delta_proxy": float(index),
            "delta_true": float(index - 1),
            "update_footprint": float(3 - index),
            "response_strength": 0.5,
            "competition_strength": 0.0,
            "candidate_index": index,
            "footprint_components": {"mean_kl": 0.1},
        }
        for index in range(4)
    ]
    model = DifferentialModel()
    model.fit(rows, [float(row["delta_true"]) - float(row["delta_proxy"]) for row in rows])
    result = _evaluate_method("pivot", rows, 2, model, seed=1)
    assert result["queries"] == 2
