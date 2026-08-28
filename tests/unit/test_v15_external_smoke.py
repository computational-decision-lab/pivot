from __future__ import annotations

from experiments.v15.external_runtime import ExecutionRecord


def _record(status: str, success: float) -> ExecutionRecord:
    return ExecutionRecord(
        status=status,
        terminal_state=None,
        task_id="task",
        task_hash="task-hash",
        policy_hash="policy-hash",
        seed=1,
        success=success,
        inspect_log="log",
        trajectory="trajectory",
        initial_tree_hash="initial",
        final_tree_hash="final",
        test_returncode=0 if success else 1,
        exit_status="Submitted" if success else "Failed",
        resource_metrics={"model_calls": 2.0, "tool_calls": 3.0},
        trace=("ls",),
    )


def test_external_smoke_summary_counts_completed_records() -> None:
    from experiments.v15.dev.external_smoke import summarize_records

    summary = summarize_records([_record("COMPLETED", 1.0), _record("COMPLETED", 0.0)])

    assert summary["status"] == "COMPLETED"
    assert summary["container_executions"] == 2
    assert summary["model_calls_performed"] == 4
    assert summary["success_rate"] == 0.5


def test_external_smoke_summary_preserves_failure_state() -> None:
    from experiments.v15.dev.external_smoke import summarize_records

    summary = summarize_records([_record("IMPLEMENTATION_FAILURE", 0.0)])

    assert summary["status"] == "IMPLEMENTATION_FAILURE"
    assert summary["terminal_state"] == "IMPLEMENTATION_FAILURE"
