from __future__ import annotations

from pathlib import Path


def test_family_success_and_pair_deltas_are_aggregated_by_task_family() -> None:
    from experiments.v15.external_runtime import ExecutionRecord, PairedExecutionRecord
    from experiments.v15.external_study import family_pair_deltas, family_success
    from experiments.v15.planes import TaskSpec

    tasks = (
        TaskSpec("a", "bug_fixing", {"a.py": "x = 1\n"}),
        TaskSpec("b", "tool_context", {"b.py": "x = 1\n"}),
    )
    records = [
        ExecutionRecord("COMPLETED", None, "a", tasks[0].task_hash, "p", 1, 1.0, None, None, "i", "f", 0, "ok", {}, ()),
        ExecutionRecord("COMPLETED", None, "b", tasks[1].task_hash, "p", 1, 0.0, None, None, "i", "f", 1, "ok", {}, ()),
    ]
    pairs = [
        PairedExecutionRecord("a", tasks[0].task_hash, 1, "i", "p", 0.0, 1.0, None, None, None),
        PairedExecutionRecord("b", tasks[1].task_hash, 1, "i", "p", 1.0, 0.0, None, None, None),
    ]

    assert family_success(tasks, records) == {"bug_fixing": 1.0, "tool_context": 0.0}
    assert family_pair_deltas(tasks, pairs) == {"bug_fixing": 1.0, "tool_context": -1.0}


def test_external_study_output_path_is_explicitly_phase_scoped(tmp_path: Path) -> None:
    from experiments.v15.external_study import phase_output

    assert phase_output(tmp_path, confirmatory=False).name == "dev-external-transition-audit"
    assert phase_output(tmp_path, confirmatory=True).name == "external-transition-audit"
