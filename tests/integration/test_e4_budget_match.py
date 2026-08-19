from __future__ import annotations

from pivot.transfer.reversal import compare_global_vs_local


def _row(index: int, split: str) -> dict[str, object]:
    return {
        "transition_id": f"{split}-{index}",
        "candidate_policy_id": f"p-{split}-{index}",
        "candidate_parameters": {"intensity": 0.2 + index * 0.1},
        "incumbent_parameters": {"intensity": 0.2},
        "true_candidate_value": 1.0 + index,
        "true_incumbent_value": 1.0,
        "delta_proxy": 0.8 - index * 0.1,
        "delta_true": 0.6 - index * 0.2,
        "response_strength": 0.5,
        "competition_strength": 0.0,
        "candidate_index": index,
        "update_footprint": 0.1 + index * 0.1,
        "footprint_components": {"mean_kl": 0.01 * index, "action_shift": 0.1},
        "round_id": 0,
    }


def test_global_and_local_models_use_identical_disjoint_budget() -> None:
    train = [_row(index, "train") for index in range(3)]
    test = [_row(index, "test") for index in range(3)]
    result = compare_global_vs_local(train, test, budget=2)
    assert result["global_hf_budget"] == result["local_hf_budget"] == 2
    assert not set(result["train_transition_ids"]) & set(result["test_transition_ids"])
    assert "policy_value_mae" in result
    assert "improvement_sign_consistency" in result
    assert result["global_update_selection_regret"] is not None
    assert result["update_selection_regret"] is not None
    assert result["boosted_update_selection_regret"] is not None
    assert sum(bool(row["local_selected"]) for row in result["local_rows"]) == 1
    assert sum(bool(row["global_selected"]) for row in result["local_rows"]) == 1
