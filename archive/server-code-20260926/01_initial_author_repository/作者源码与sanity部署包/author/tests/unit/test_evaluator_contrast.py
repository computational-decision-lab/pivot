from __future__ import annotations

from pivot.transfer.evaluator_contrast import EvaluatorContrastConfig, run_evaluator_contrast


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (true_delta, response, footprint) in enumerate(
        ((2.0, 0.0, 0.1), (-0.5, 0.9, 0.5), (1.0, 0.7, 0.3), (0.2, 0.3, 0.2))
    ):
        rows.append(
            {
                "transition_id": f"t{index}",
                "config_id": "toy",
                "seed": 1,
                "response_strength": response,
                "update_footprint": footprint,
                "optimization_strength": 1.0,
                "candidate_index": index,
                "true_incumbent_value": 10.0,
                "true_candidate_value": 10.0 + true_delta,
                "delta_true": true_delta,
            }
        )
    return rows


def test_common_offset_preserves_transition_but_asymmetric_bias_can_reverse() -> None:
    result = run_evaluator_contrast(
        _rows(),
        {"t0"},
        EvaluatorContrastConfig(differential_bias_scale=4.0, common_value_offset=0.6),
    )
    value = result["metrics"]["value_fidelity"]
    transition = result["metrics"]["transition_fidelity"]
    assert value["policy_value_mae"] < transition["policy_value_mae"]
    assert value["improvement_differential_error"] > transition["improvement_differential_error"]
    assert transition["improvement_sign_consistency"] == 1.0
    assert transition["improvement_reversal_rate"] == 0.0
    assert value["n_reversals"] >= 1
    assert value["cumulative_true_improvement"] is not None
    assert transition["cumulative_true_improvement"] is not None


def test_contrast_is_deterministic_and_keeps_train_ids_out_of_test() -> None:
    first = run_evaluator_contrast(_rows(), {"t0"})
    second = run_evaluator_contrast(_rows(), {"t0"})
    assert first == second
    assert "t0" not in first["test_transition_ids"]
    assert first["train_row_count"] == 1
