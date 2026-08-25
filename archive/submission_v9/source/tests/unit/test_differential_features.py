from __future__ import annotations

from pivot.transfer.differential import DifferentialModel
from pivot.transfer.features import transition_feature_vector


def _row() -> dict[str, object]:
    return {
        "transition_id": "t-1",
        "delta_proxy": 0.2,
        "update_footprint": 0.9,
        "response_strength": 0.7,
        "competition_strength": 0.1,
        "candidate_index": 2,
        "footprint_components": {
            "mean_kl": 0.4,
            "action_shift": 0.3,
            "entropy_change": 0.2,
            "support_expansion": 0.1,
        },
    }


def test_transition_feature_vector_can_exclude_footprint_features() -> None:
    with_footprint = transition_feature_vector(_row(), include_footprint=True)
    without_footprint = transition_feature_vector(_row(), include_footprint=False)
    assert with_footprint[0] == without_footprint[0]
    assert with_footprint[2:5].tolist() == without_footprint[2:5].tolist()
    assert without_footprint[1] == 0.0
    assert without_footprint[5:].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_differential_model_records_footprint_ablation() -> None:
    rows = [_row(), {**_row(), "transition_id": "t-2", "candidate_index": 3, "delta_proxy": 0.3}]
    target = [0.1, -0.2]
    model = DifferentialModel(include_footprint=False)
    model.fit(rows, target)
    assert model.include_footprint is False
    assert model.predict_correction(rows[0]).predicted_delta == model.predict_correction(
        {**rows[0], "update_footprint": 100.0}
    ).predicted_delta
