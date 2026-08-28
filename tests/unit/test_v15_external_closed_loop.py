from __future__ import annotations

from pathlib import Path

import pytest


def test_closed_loop_phase_output_is_separate(tmp_path: Path) -> None:
    from experiments.v15.external_closed_loop import phase_output

    assert phase_output(tmp_path, confirmatory=False).name == "dev-external-closed-loop"
    assert phase_output(tmp_path, confirmatory=True).name == "external-closed-loop"


def test_method_selection_never_reads_assessment_outcomes() -> None:
    from experiments.v15.external_closed_loop import select_from_gate_queries

    rows = [
        {"candidate_id": "a", "candidate_index": 0, "proxy_delta": 0.2, "footprint": {}},
        {"candidate_id": "b", "candidate_index": 1, "proxy_delta": 0.1, "footprint": {}},
    ]
    selected, queries = select_from_gate_queries(rows, {}, method="Proxy Only", budget=1, seed=1)

    assert selected["candidate_id"] == "a"
    assert queries == []


def test_closed_loop_rejects_non_terminal_assessment_access() -> None:
    from experiments.v15.external_closed_loop import assessment_tasks_for_terminal

    with pytest.raises(PermissionError, match="terminal"):
        assessment_tasks_for_terminal("promotion")

    assert assessment_tasks_for_terminal("terminal_assessor") == "assessment"


def test_closed_loop_query_accounting_separates_logical_budget_from_truth_audit() -> None:
    from experiments.v15.external_closed_loop import closed_loop_query_accounting

    accounting = closed_loop_query_accounting(
        query_count=2,
        candidate_count=4,
        observed_count=2,
        query_pair_count=6,
        truth_pair_count=6,
    )

    assert accounting == {
        "logical_hf_queries": 2,
        "pre_decision_pair_evaluations": 6,
        "post_decision_truth_evaluations": 2,
        "post_decision_truth_pair_evaluations": 6,
        "total_pair_evaluations": 12,
    }


def test_closed_loop_query_accounting_rejects_inconsistent_counts() -> None:
    from experiments.v15.external_closed_loop import closed_loop_query_accounting

    with pytest.raises(ValueError, match="observed_count"):
        closed_loop_query_accounting(
            query_count=2,
            candidate_count=1,
            observed_count=2,
            query_pair_count=2,
            truth_pair_count=0,
        )


def test_registered_closed_loop_budget_comes_from_protocol() -> None:
    from experiments.v15.external_closed_loop import registered_closed_loop_budget

    assert registered_closed_loop_budget({"hf_budgets": [1, 2, 4]}, 8) == 4
    assert registered_closed_loop_budget({"hf_budgets": [1, 2, 4]}, 2) == 2


def test_registered_closed_loop_budget_rejects_unregistered_shape() -> None:
    from experiments.v15.external_closed_loop import registered_closed_loop_budget

    with pytest.raises(TypeError, match="hf_budgets"):
        registered_closed_loop_budget({"hf_budgets": "4"}, 4)
