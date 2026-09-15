from __future__ import annotations

from pathlib import Path


def test_mutation_response_is_identity_blind_and_restores_source(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.strategic_response import run_mutation_response

    source = tmp_path / "src.py"
    source.write_text("def score(x):\n    return x + 1\n", encoding="utf-8")
    task = TaskSpec(
        task_id="mutation-task",
        family="bug_fixing",
        files={"src.py": source.read_text(encoding="utf-8")},
        metadata={"test_command": "python -m unittest"},
    )
    calls: list[str] = []

    def runner(command: str) -> int:
        calls.append(command)
        return 1

    audit = run_mutation_response(
        tmp_path,
        task,
        changed=("src.py",),
        trace=("python -m unittest",),
        run_command=runner,
    )

    assert audit.status == "COMPLETED"
    assert audit.attempted == 1
    assert audit.killed == 1
    assert source.read_text(encoding="utf-8") == "def score(x):\n    return x + 1\n"
    assert calls == ["python -m unittest"]


def test_mutation_response_does_not_turn_missing_source_into_a_delta(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.strategic_response import run_mutation_response

    task = TaskSpec(
        task_id="docs-task",
        family="tool_context",
        files={"README.md": "read\n"},
        metadata={"test_command": "true"},
    )
    audit = run_mutation_response(
        tmp_path,
        task,
        changed=("README.md",),
        trace=(),
        run_command=lambda _: 0,
    )

    assert audit.status == "NO_MUTABLE_SOURCE"
    assert audit.mutation_score is None


def test_paired_mutation_response_uses_same_response_contract_for_both_trees(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.strategic_response import run_paired_mutation_response

    task = TaskSpec(
        task_id="response-task",
        family="bug_fixing",
        files={"src/math.py": "def add(a, b):\n    return a + b\n"},
        metadata={"test_command": "python -m unittest -v"},
    )
    incumbent = tmp_path / "incumbent"
    candidate = tmp_path / "candidate"
    (incumbent / "src").mkdir(parents=True)
    (candidate / "src").mkdir(parents=True)
    (incumbent / "src/math.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (candidate / "src/math.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    def run_command(_command: str, *, cwd: Path) -> int:
        source = (cwd / "src/math.py").read_text(encoding="utf-8")
        return int(" - " in source)

    audit = run_paired_mutation_response(
        incumbent,
        candidate,
        task,
        changed=("src/math.py",),
        incumbent_trace=("python -m unittest",),
        candidate_trace=("python -m unittest", "sed"),
        run_command=run_command,
    )

    assert audit.status == "COMPLETED"
    assert audit.incumbent_attempted == audit.candidate_attempted == 1
    assert audit.incumbent_score == 1.0
    assert audit.candidate_score == 0.0
    assert audit.delta_strategic == -1.0
    assert " + " in (incumbent / "src/math.py").read_text(encoding="utf-8")
    assert " - " in (candidate / "src/math.py").read_text(encoding="utf-8")


def test_ablation_dev_task_bound_is_explicit_and_never_changes_confirmatory() -> None:
    from experiments.v15.run_ablations import bounded_tasks

    tasks = ["task-a", "task-b", "task-c"]
    assert bounded_tasks(tasks, confirmatory=False, task_limit=1) == ["task-a"]
    assert bounded_tasks(tasks, confirmatory=False, task_limit=None) == tasks
