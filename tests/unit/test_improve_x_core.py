from __future__ import annotations

from collections.abc import Mapping

import pytest

from improve_x.core.operator import CandidateBatch
from improve_x.core.trajectory import ImprovementTrajectory
from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


def make_batch() -> CandidateBatch:
    incumbent = Policy.from_mapping({"intensity": 0.1})
    transitions = tuple(
        PolicyTransition(
            incumbent=incumbent,
            candidate=Policy.from_mapping({"intensity": value}),
            round_id=0,
            candidate_index=index,
            improvement_operator="test",
            seed=7,
        )
        for index, value in enumerate((0.2, 0.3))
    )
    return CandidateBatch(incumbent, transitions, "test", 0, 7)


def test_candidate_batch_requires_same_incumbent_and_nonempty_candidates() -> None:
    batch = make_batch()
    assert batch.operator == "test"
    assert batch.candidate_ids == tuple(transition.candidate.policy_id for transition in batch.candidates)
    with pytest.raises(ValueError, match="at least one"):
        CandidateBatch(batch.incumbent, (), "test", 0, 7)
    other = Policy.from_mapping({"intensity": 0.9})
    mismatched = PolicyTransition(
        incumbent=other,
        candidate=batch.candidates[0].candidate,
        round_id=0,
        candidate_index=0,
        improvement_operator="test",
    )
    with pytest.raises(ValueError, match="incumbent"):
        CandidateBatch(batch.incumbent, (mismatched,), "test", 0, 7)


def test_candidate_batch_metadata_is_immutable() -> None:
    batch = CandidateBatch(
        make_batch().incumbent,
        make_batch().candidates,
        "test",
        0,
        7,
        metadata={"source": "test"},
    )

    with pytest.raises(TypeError):
        batch.metadata["source"] = "changed"  # type: ignore[index]


def test_trajectory_promotes_selected_candidate_and_preserves_all_rows() -> None:
    batch = make_batch()
    trajectory = ImprovementTrajectory(batch.incumbent)
    evaluations: tuple[Mapping[str, object], ...] = (
        {"delta_proxy": 1.0, "true_delta": -0.5},
        {"delta_proxy": 0.5, "true_delta": 0.25},
    )
    promoted = trajectory.append_round(batch, selected_index=1, evaluations=evaluations, query_cost=2.0)
    assert promoted == batch.candidates[1].candidate
    assert trajectory.current_policy == promoted
    assert trajectory.cumulative_true_improvement == pytest.approx(0.25)
    records = trajectory.to_records()
    assert len(records) == 2
    assert records[0]["selected"] is False
    assert records[1]["selected"] is True
    assert records[0]["true_delta"] == -0.5
    assert trajectory.proxy_curve == (0.0, 0.5)
    assert trajectory.true_curve == (0.0, 0.25)


def test_trajectory_rejects_wrong_evaluation_count() -> None:
    batch = make_batch()
    trajectory = ImprovementTrajectory(batch.incumbent)
    with pytest.raises(ValueError, match="one evaluation"):
        trajectory.append_round(batch, 0, ({"delta_proxy": 1.0},))


def test_trajectory_uses_pivot_delta_true_for_cumulative_curves() -> None:
    batch = make_batch()
    trajectory = ImprovementTrajectory(batch.incumbent)
    evaluations: tuple[Mapping[str, object], ...] = (
        {"delta_proxy": 1.0, "delta_true": -0.5},
        {"delta_proxy": 0.5, "delta_true": -0.25},
    )

    trajectory.append_round(batch, selected_index=1, evaluations=evaluations)

    assert trajectory.cumulative_true_improvement == pytest.approx(-0.25)
    assert trajectory.true_curve == (0.0, -0.25)


def test_trajectory_exposes_actor_and_strategic_curves() -> None:
    batch = make_batch()
    trajectory = ImprovementTrajectory(batch.incumbent)
    evaluations: tuple[Mapping[str, object], ...] = (
        {"delta_proxy": 1.0, "delta_true": 0.5, "delta_actor": 0.5, "delta_strategic": -0.2},
        {"delta_proxy": 0.5, "delta_true": 0.25, "delta_actor": 0.25, "delta_strategic": -0.1},
    )

    trajectory.append_round(batch, selected_index=0, evaluations=evaluations)

    assert trajectory.cumulative_actor_improvement == pytest.approx(0.5)
    assert trajectory.cumulative_strategic_improvement == pytest.approx(-0.2)
    assert trajectory.actor_curve == (0.0, 0.5)
    assert trajectory.strategic_curve == (0.0, -0.2)
