from __future__ import annotations

import pytest

from pivot.benchmark.improvementbench_v7 import (
    ImprovementBenchV7Row,
    assign_group_split,
    assign_leakage_safe_splits,
    validate_group_splits,
)


def _record() -> dict[str, object]:
    return {
        "transition_id": "t1",
        "trajectory_id": "traj-a",
        "round_id": 2,
        "operator_id": "op-a",
        "environment_id": "env-a",
        "incumbent_policy_id": "p0",
        "candidate_policy_id": "p1",
        "delta_proxy": 0.3,
        "delta_direct": 0.2,
        "delta_actor": -0.1,
        "delta_strategic": None,
        "update_footprint": 0.4,
        "operator_shift": 0.2,
        "response_strength": 0.5,
        "competition_strength": 0.0,
        "paired_rollout_ids": ["seed-2"],
        "proxy_rank": 1,
        "true_rank": 2,
        "failure_type": "mechanical_reversal",
        "hf_queried": True,
        "hf_cost": 12.0,
        "seed": 2,
    }


def test_v7_row_preserves_transition_and_split_contract() -> None:
    row = ImprovementBenchV7Row.from_record(_record(), split="test")
    payload = row.to_record()

    assert payload["split"] == "test"
    assert payload["paired_rollout_ids"] == ["seed-2"]
    assert payload["failure_type"] == "mechanical_reversal"


def test_group_split_is_deterministic_and_split_validator_rejects_leakage() -> None:
    assert assign_group_split("trajectory-a", seed=1) == assign_group_split("trajectory-a", seed=1)
    rows = [
        ImprovementBenchV7Row.from_record(_record(), split="development"),
        ImprovementBenchV7Row.from_record(
            {**_record(), "transition_id": "t2", "trajectory_id": "traj-b"}, split="test"
        ),
    ]
    validate_group_splits(rows)
    leaked = [
        ImprovementBenchV7Row.from_record(_record(), split="development"),
        ImprovementBenchV7Row.from_record({**_record(), "transition_id": "t2"}, split="test"),
    ]
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_splits(leaked, group_key="environment_id")


def test_connected_group_split_keeps_shared_environment_together() -> None:
    records = [_record(), {**_record(), "transition_id": "t2", "trajectory_id": "traj-b"}]
    splits = assign_leakage_safe_splits(records, seed=4)
    assert splits[0] == splits[1]
