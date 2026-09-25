from __future__ import annotations

import shutil
from pathlib import Path


def test_pi_confirmatory_plan_matches_registered_protocol() -> None:
    import yaml

    from experiments.v15.run_pi_replication import build_pi_confirmatory_plan

    config = yaml.safe_load(Path("configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    plan = build_pi_confirmatory_plan(
        config,
        {
            "status": "COMPLETED",
            "trajectory_count": 60,
            "round_count": 30,
            "candidate_count": 7200,
        },
    )

    assert plan["operator_count"] == 2
    assert plan["trajectory_count"] == 60
    assert plan["round_count"] == 30
    assert plan["candidates_per_round"] == 4
    assert plan["transition_count"] == 7200


def test_pi_confirmatory_plan_rejects_incomplete_primary_archive() -> None:
    import pytest
    import yaml

    from experiments.v15.run_pi_replication import build_pi_confirmatory_plan

    config = yaml.safe_load(Path("configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="primary candidate count"):
        build_pi_confirmatory_plan(
            config,
            {
                "status": "COMPLETED",
                "trajectory_count": 60,
                "round_count": 30,
                "candidate_count": 7199,
            },
        )


def test_pi_pair_rows_require_same_task_hash_and_compute_difference() -> None:
    from experiments.v15.run_pi_replication import pair_pi_execution_rows

    incumbent = {
        "task_id": "task-1",
        "task_hash": "hash-1",
        "success": 0.0,
        "trajectory": "inc.jsonl",
        "final_tree_path": "inc.tree",
        "resource_metrics": {"tool_calls": 2.0},
    }
    candidate = {
        "task_id": "task-1",
        "task_hash": "hash-1",
        "success": 1.0,
        "trajectory": "cand.jsonl",
        "final_tree_path": "cand.tree",
        "resource_metrics": {"tool_calls": 3.0},
    }

    pair = pair_pi_execution_rows(incumbent, candidate, seed=17)

    assert pair["delta_actor"] == 1.0
    assert pair["task_hash"] == "hash-1"
    assert pair["incumbent_execution"] == "inc.jsonl"
    assert pair["candidate_execution"] == "cand.jsonl"


def test_pi_pair_rows_reject_initial_sandbox_mismatch() -> None:
    import pytest

    from experiments.v15.run_pi_replication import pair_pi_execution_rows

    incumbent = {
        "task_id": "task-1",
        "task_hash": "hash-1",
        "seed": 17,
        "success": 0.0,
        "initial_tree_hash": "tree-a",
        "status": "COMPLETED",
    }
    candidate = {
        "task_id": "task-1",
        "task_hash": "hash-1",
        "seed": 17,
        "success": 1.0,
        "initial_tree_hash": "tree-b",
        "status": "COMPLETED",
    }
    with pytest.raises(ValueError, match="initial sandbox"):
        pair_pi_execution_rows(incumbent, candidate, seed=17)


def test_pi_budget_check_marks_over_limit_rollouts() -> None:
    from experiments.v15.run_pi_replication import pi_budget_violation

    assert pi_budget_violation(tool_calls=4, elapsed_seconds=2.0, tool_limit=4, wall_limit=3) is None
    assert "tool-call" in str(
        pi_budget_violation(tool_calls=5, elapsed_seconds=2.0, tool_limit=4, wall_limit=3)
    )
    assert "wall-clock" in str(
        pi_budget_violation(tool_calls=1, elapsed_seconds=4.0, tool_limit=4, wall_limit=3)
    )
    assert "token" in str(
        pi_budget_violation(
            tool_calls=1,
            elapsed_seconds=1.0,
            tool_limit=4,
            wall_limit=3,
            tokens=11,
            token_limit=10,
        )
    )


def test_pi_token_usage_parser_accepts_common_provider_shapes() -> None:
    from experiments.v15.run_pi_replication import _pi_token_count

    trace = (
        '{"type":"message_end","usage":{"input_tokens":3,"output_tokens":2}}\n'
        '{"type":"message_end","message":{"usage":{"total_tokens":7}}}'
    )
    assert _pi_token_count(trace) == 12.0


def test_pi_sandbox_command_binds_only_the_workspace_when_bwrap_is_available(tmp_path: Path) -> None:
    from experiments.v15.run_pi_replication import pi_sandbox_command

    runtime = tmp_path / "pi-runtime"
    cli = runtime / "dist" / "cli.js"
    extension = tmp_path / "project" / "extension.ts"
    workspace = tmp_path / "workspace"
    cli.parent.mkdir(parents=True)
    extension.parent.mkdir(parents=True)
    workspace.mkdir()
    cli.write_text("", encoding="utf-8")
    extension.write_text("", encoding="utf-8")
    command = pi_sandbox_command(
        cli=cli,
        extension=extension,
        workspace=workspace,
        prompt="repair",
        use_bwrap=True,
        pi_root=runtime,
        node_executable=Path(shutil.which("node") or "node"),
    )

    assert command[0] == "bwrap"
    assert "--ro-bind" in command
    assert str(workspace) in command
    assert "--bind" in command
    assert "/workspace" in command
    assert "repair" in command
    assert "--no-extensions" in command
    # The sandbox must not expose the complete host filesystem.  Runtime
    # inputs are mounted explicitly by the command builder instead.
    assert not any(
        command[index : index + 3] == ["--ro-bind", "/", "/"]
        for index in range(len(command) - 2)
    )


def test_pi_sandbox_command_mounts_runtime_inputs_without_host_root(tmp_path: Path) -> None:
    from experiments.v15.run_pi_replication import pi_sandbox_command

    cli = tmp_path / "pi" / "cli.js"
    extension = tmp_path / "project" / "extension.ts"
    cli.parent.mkdir(parents=True)
    extension.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    extension.write_text("", encoding="utf-8")

    command = pi_sandbox_command(
        cli=cli,
        extension=extension,
        workspace=tmp_path / "task",
        prompt="repair",
        use_bwrap=True,
        pi_root=tmp_path / "pi",
        node_executable=Path(shutil.which("node") or "node"),
    )

    assert command.count("--ro-bind") >= 2
    assert "/runtime/pi-root" in command
    assert "/runtime/pivot-extension.ts" in command
    assert not any(
        command[index : index + 3] == ["--ro-bind", "/", "/"]
        for index in range(len(command) - 2)
    )


def test_pi_sandbox_command_rejects_missing_runtime_root(tmp_path: Path) -> None:
    import pytest

    from experiments.v15.run_pi_replication import pi_sandbox_command

    with pytest.raises(FileNotFoundError, match="Pi runtime root"):
        pi_sandbox_command(
            cli=tmp_path / "missing" / "cli.js",
            extension=tmp_path / "extension.ts",
            workspace=tmp_path,
            prompt="repair",
            use_bwrap=True,
            pi_root=tmp_path / "missing",
            node_executable=Path(shutil.which("node") or "node"),
        )


def test_pi_environment_does_not_inherit_unrelated_host_variables(tmp_path: Path, monkeypatch) -> None:
    from experiments.v15.run_pi_replication import _pi_environment

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret-for-child")
    monkeypatch.setenv("PIVOT_PRIVATE_MARKER", "must-not-cross")
    environment = _pi_environment(
        sandbox=tmp_path,
        session_dir=tmp_path / "session",
        use_bwrap=True,
    )

    assert environment["ANTHROPIC_AUTH_TOKEN"] == "secret-for-child"
    assert "PIVOT_PRIVATE_MARKER" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["HOME"] == "/tmp/home"


def test_pi_registered_tests_use_network_isolated_sandbox(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.run_pi_replication import (
        _portable_test_command,
        _test_sandbox_command,
    )

    task = TaskSpec("task-1", "bug_fixing", {"test.py": ""}, metadata={"test_command": "python -m unittest"})
    command = _portable_test_command(task, inside_sandbox=True)
    sandbox_command = _test_sandbox_command(workspace=tmp_path, command=command)

    assert command.startswith("/usr/bin/python3 ")
    assert "--unshare-all" in sandbox_command
    assert "--network" not in sandbox_command
    assert "/workspace" in sandbox_command


def test_pi_sandbox_uses_explicit_pinned_python_runtime(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.run_pi_replication import _portable_test_command, _test_sandbox_command

    python_root = tmp_path / "python-runtime"
    python_executable = python_root / "bin" / "python3.11"
    python_site = tmp_path / "python-site"
    python_executable.parent.mkdir(parents=True)
    python_site.mkdir()
    python_executable.write_text("", encoding="utf-8")
    task = TaskSpec("task-1", "bug_fixing", {"test.py": ""}, metadata={"test_command": "python -m unittest"})

    test_command = _portable_test_command(
        task,
        inside_sandbox=True,
        python_executable=python_executable,
        python_root=python_root,
    )
    sandbox_command = _test_sandbox_command(
        workspace=tmp_path / "workspace",
        command=test_command,
        python_executable=python_executable,
        python_root=python_root,
        python_site=python_site,
    )

    assert test_command == "/runtime/python/bin/python3.11 -m unittest"
    assert "/runtime/python" in sandbox_command
    assert "/runtime/python-site" in sandbox_command


def test_pi_registered_test_execution_isolated_and_reproducible(tmp_path: Path) -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.run_pi_replication import _run_registered_tests

    (tmp_path / "test_probe.py").write_text("print('ok')\n", encoding="utf-8")
    task = TaskSpec(
        "task-1",
        "bug_fixing",
        {"test_probe.py": "print('ok')\n"},
        metadata={"test_command": "python test_probe.py"},
    )

    returncode, output = _run_registered_tests(task, tmp_path, use_bwrap=True, timeout=10)

    assert returncode == 0
    assert "ok" in output


def test_pi_batch_uses_inspect_control_plane_when_runtime_settings_are_supplied(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.v15.external_runtime import RuntimeSettings
    from experiments.v15.planes import TaskSpec
    from experiments.v15.protocol import AgentPolicy
    from experiments.v15.run_pi_replication import _evaluate_pi_tasks

    task = TaskSpec("task-1", "bug_fixing", {"app.py": "value = 1\n"})
    settings = RuntimeSettings(
        model_name="anthropic/test-model",
        provider="anthropic",
        api_base="https://example.invalid",
        image="python:3.11-slim",
        image_digest="sha256:test",
        dependency_lock="lock.txt",
        artifact_root=tmp_path / "artifacts",
        log_root=tmp_path / "logs",
    )
    expected = {"task_id": "task-1", "task_hash": task.task_hash, "success": 1.0, "status": "COMPLETED"}
    calls: list[dict[str, object]] = []

    def fake_inspect(*args, **kwargs):
        calls.append({"phase": kwargs["phase"], "role": kwargs["role"]})
        return [expected]

    monkeypatch.setattr("experiments.v15.run_pi_replication.evaluate_pi_with_inspect", fake_inspect)
    records, failures = _evaluate_pi_tasks(
        Path("."),
        [task],
        AgentPolicy.minimal().with_updates(metadata={"scaffold": "Pi"}),
        seed=1,
        output=tmp_path / "run",
        agent_steps=4,
        tool_limit=8,
        wall_limit=30,
        settings=settings,
        phase="proxy",
        role="proxy_evaluator",
        run_id="run-1",
    )

    assert records == [expected]
    assert failures == []
    assert calls == [{"phase": "proxy", "role": "proxy_evaluator"}]
