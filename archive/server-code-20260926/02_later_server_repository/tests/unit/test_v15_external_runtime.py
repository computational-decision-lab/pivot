from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest


def test_runtime_settings_resolve_from_pinned_config_without_secret(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import resolve_runtime_settings

    settings = resolve_runtime_settings(Path("."), artifact_root=tmp_path / "artifacts", log_root=tmp_path / "logs")

    assert settings.model_name == "anthropic/claude-haiku-4-5-20251001"
    assert settings.image_digest.startswith("sha256:")
    assert settings.dependency_lock == "configs/v15/external_runtime_requirements.txt"
    manifest = settings.to_manifest()
    assert "api_key" not in json.dumps(manifest).casefold()
    assert "auth_token" not in json.dumps(manifest).casefold()


def test_locked_external_runtime_path_is_resolved_from_provenance() -> None:
    from experiments.v15.external_runtime import locked_runtime_python

    runtime = locked_runtime_python(Path("."))

    # Preserve the manifest's venv entry point.  Resolving its ``python``
    # symlink before ``execve`` would drop the environment's site-packages.
    assert runtime.name == "python"
    assert runtime.is_file()
    assert ".tools/v15/runtime" in runtime.as_posix()


def test_locked_external_runtime_executes_inside_the_pinned_venv() -> None:
    from experiments.v15.external_runtime import locked_runtime_python

    runtime = locked_runtime_python(Path("."))
    prefix = subprocess.check_output(
        [str(runtime), "-c", "import sys; print(sys.prefix)"], text=True
    ).strip()

    assert ".tools/v15/runtime" in prefix


def test_runtime_settings_reject_unresolved_protocol_inputs(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import RuntimeSettings

    with pytest.raises(ValueError, match="image"):
        RuntimeSettings(
            model_name="m",
            provider="anthropic",
            api_base="https://example.invalid",
            image="",
            image_digest="",
            dependency_lock="lock.txt",
            artifact_root=tmp_path / "a",
            log_root=tmp_path / "l",
        )


def test_runtime_settings_rejects_mutable_or_escaping_inputs(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import RuntimeSettings

    kwargs = {
        "model_name": "m",
        "provider": "anthropic",
        "api_base": "https://example.invalid",
        "image": "python:3.11-slim",
        "image_digest": "sha256:test",
        "dependency_lock": "lock.txt",
        "artifact_root": tmp_path / "a",
        "log_root": tmp_path / "l",
    }
    with pytest.raises(ValueError, match="sha256"):
        RuntimeSettings(**{**kwargs, "image_digest": "latest"})
    with pytest.raises(ValueError, match="repository-relative"):
        RuntimeSettings(**{**kwargs, "dependency_lock": "../lock.txt"})


def test_budget_contract_reports_all_overruns(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import (
        RuntimeSettings,
        annotate_budget_metrics,
        budget_violation,
    )

    settings = RuntimeSettings(
        model_name="m",
        provider="anthropic",
        api_base="https://example.invalid",
        image="python:3.11-slim",
        image_digest="sha256:test",
        dependency_lock="lock.txt",
        artifact_root=tmp_path / "a",
        log_root=tmp_path / "l",
        token_limit=10,
        tool_calls=2,
        wall_clock_seconds=3,
    )
    metrics, reason = annotate_budget_metrics(
        {"tokens": 11.0, "tool_calls": 3.0, "wall_clock_seconds": 4.0}, settings
    )

    assert reason is not None
    assert "token budget exceeded" in reason
    assert "tool-call budget exceeded" in reason
    assert "wall-clock budget exceeded" in reason
    assert budget_violation(metrics, settings) == reason
    assert metrics["budget_within_limits"] == 0.0
    assert metrics["budget_violation_count"] == 1.0


def test_budget_contract_treats_missing_counters_as_zero(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import RuntimeSettings, annotate_budget_metrics

    settings = RuntimeSettings(
        model_name="m",
        provider="anthropic",
        api_base="https://example.invalid",
        image="python:3.11-slim",
        image_digest="sha256:test",
        dependency_lock="lock.txt",
        artifact_root=tmp_path / "a",
        log_root=tmp_path / "l",
        token_limit=10,
        tool_calls=2,
        wall_clock_seconds=3,
    )
    metrics, reason = annotate_budget_metrics({"wall_clock_seconds": 1.0}, settings)

    assert reason is None
    assert metrics["budget_within_limits"] == 1.0
    assert metrics["tokens"] == 0.0
    assert metrics["tool_calls"] == 0.0


def test_token_count_falls_back_to_prompt_and_completion_usage() -> None:
    from experiments.v15.external_runtime import _token_count

    class Agent:
        messages: ClassVar[list[dict[str, object]]] = [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {"prompt_tokens": 17, "completion_tokens": 5}
                    }
                },
            }
        ]

    assert _token_count(Agent()) == 22.0


def test_evaluator_path_classifier_covers_existing_and_new_test_files() -> None:
    from experiments.v15.external_runtime import _is_evaluator_path

    protected = (
        "tests/test_contract.py",
        "test.py",
        "src/test.py",
        "src/foo_test.py",
        "src/foo.test.js",
        "__tests__/smoke.ts",
        "test-fixtures/test-data.py",
    )
    unprotected = ("src/testing_helpers.py", "src/contest.py", "README.md")
    assert all(_is_evaluator_path(path) for path in protected)
    assert not any(_is_evaluator_path(path) for path in unprotected)


def test_restore_evaluator_files_removes_new_evaluator_files(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import _restore_evaluator_files

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_existing.py").write_text("original\n", encoding="utf-8")
    snapshot = {"tests/test_existing.py": b"original\n"}
    (tmp_path / "tests/test_existing.py").write_text("tampered\n", encoding="utf-8")
    (tmp_path / "test.py").write_text("new evaluator\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/foo_test.py").write_text("new nested evaluator\n", encoding="utf-8")
    (tmp_path / "src/contest.py").write_text("allowed source\n", encoding="utf-8")

    changed = _restore_evaluator_files(tmp_path, snapshot)

    assert changed == ("src/foo_test.py", "test.py", "tests/test_existing.py")
    assert (tmp_path / "tests/test_existing.py").read_text(encoding="utf-8") == "original\n"
    assert not (tmp_path / "test.py").exists()
    assert not (tmp_path / "src/foo_test.py").exists()
    assert (tmp_path / "src/contest.py").exists()


def test_task_manifest_has_two_families_per_plane() -> None:
    from experiments.v15.planes import load_task_planes

    planes = load_task_planes(Path("configs/v15/task_manifest.json"))
    for plane in (planes.proxy, planes.gate, planes.assessment):
        assert {task.family for task in plane} == {"bug_fixing", "tool_context"}
        assert len(plane) == 4


def test_secret_sanitizer_redacts_nested_credentials() -> None:
    from experiments.v15.external_runtime import _sanitize

    clean = _sanitize(
        {"api_key": "secret", "nested": {"authorization": "Bearer secret", "text": "secret"}},
        secret_values=("secret",),
    )
    assert clean == {
        "api_key": "<redacted>",
        "nested": {"authorization": "<redacted>", "text": "<redacted>"},
    }


def test_execution_record_is_persisted_before_sandbox_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.v15.external_runtime import RuntimeSettings, execute_mini_task
    from experiments.v15.planes import TaskSpec
    from experiments.v15.protocol import AgentPolicy

    class FakeModel:
        def __init__(self, **_: object) -> None:
            pass

    class FakeEnvironment:
        captured: ClassVar[dict[str, object]] = {}

        def __init__(self, **_: object) -> None:
            FakeEnvironment.captured = dict(_)

        def execute(self, _action: dict[str, str], cwd: str = "") -> dict[str, object]:
            return {"returncode": 0, "output": "ok", "exception_info": "", "cwd": cwd}

        def cleanup(self) -> None:
            pass

    class FakeAgent:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.messages: list[dict[str, object]] = []

        def run(self, _task: str) -> dict[str, str]:
            return {"exit_status": "Submitted"}

        def serialize(self) -> dict[str, object]:
            return {"messages": self.messages}

    modules = {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.agents": types.ModuleType("minisweagent.agents"),
        "minisweagent.agents.default": types.ModuleType("minisweagent.agents.default"),
        "minisweagent.environments": types.ModuleType("minisweagent.environments"),
        "minisweagent.environments.docker": types.ModuleType("minisweagent.environments.docker"),
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": types.ModuleType("minisweagent.models.litellm_model"),
    }
    modules["minisweagent.agents.default"].DefaultAgent = FakeAgent  # type: ignore[attr-defined]
    modules["minisweagent.environments.docker"].DockerEnvironment = FakeEnvironment  # type: ignore[attr-defined]
    modules["minisweagent.models.litellm_model"].LitellmModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "minisweagent", modules["minisweagent"])
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")

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
    task = TaskSpec(
        task_id="persist-task",
        family="bug_fixing",
        files={"app.py": "value = 1\n"},
        metadata={"instruction": "Run the task", "test_command": "true"},
    )

    record = execute_mini_task(task, AgentPolicy.minimal(), settings, seed=4, artifact_dir=tmp_path / "run")

    assert record.status == "COMPLETED"
    assert record.final_tree_hash == record.initial_tree_hash
    assert "--user" in list(FakeEnvironment.captured["run_args"])  # type: ignore[arg-type]
    assert FakeEnvironment.captured["image"] == "python:3.11-slim@sha256:test"
    assert list((tmp_path / "run").glob("*.execution.json"))


def test_paired_record_computes_candidate_minus_incumbent_delta() -> None:
    from experiments.v15.external_runtime import PairedExecutionRecord

    pair = PairedExecutionRecord(
        task_id="task",
        task_hash="task-hash",
        seed=2,
        incumbent_policy_hash="incumbent",
        candidate_policy_hash="candidate",
        incumbent_success=0.0,
        candidate_success=1.0,
        incumbent_execution="inc.json",
        candidate_execution="cand.json",
        inspect_log="log.eval",
    )

    assert pair.delta == 1.0
    assert pair.to_record()["delta"] == 1.0


def test_paired_record_failure_uses_side_status() -> None:
    from experiments.v15.external_runtime import PairedExecutionRecord, paired_execution_failed

    pair = PairedExecutionRecord(
        task_id="task",
        task_hash="task-hash",
        seed=2,
        incumbent_policy_hash="incumbent",
        candidate_policy_hash="candidate",
        incumbent_success=0.0,
        candidate_success=1.0,
        incumbent_execution="inc.json",
        candidate_execution="cand.json",
        inspect_log="log.eval",
        candidate_status="IMPLEMENTATION_FAILURE",
        candidate_error="runner failed",
    )

    assert paired_execution_failed(pair) is True
    assert pair.to_record()["candidate_error"] == "runner failed"


def test_paired_record_persists_initial_sandbox_contract() -> None:
    from experiments.v15.external_runtime import PairedExecutionRecord

    pair = PairedExecutionRecord(
        task_id="task",
        task_hash="task-hash",
        seed=2,
        incumbent_policy_hash="incumbent",
        candidate_policy_hash="candidate",
        incumbent_success=0.0,
        candidate_success=1.0,
        incumbent_execution="inc.json",
        candidate_execution="cand.json",
        inspect_log="log.eval",
        incumbent_initial_tree_hash="same-tree",
        candidate_initial_tree_hash="same-tree",
        pairing_contract_hash="contract",
    )

    record = pair.to_record()
    assert record["incumbent_initial_tree_hash"] == "same-tree"
    assert record["candidate_initial_tree_hash"] == "same-tree"
    assert record["pairing_contract_hash"] == "contract"


def test_pairing_contract_hash_is_deterministic(tmp_path: Path) -> None:
    from experiments.v15.external_runtime import RuntimeSettings, pairing_contract_hash
    from experiments.v15.planes import TaskSpec

    settings = RuntimeSettings(
        model_name="anthropic/test",
        provider="anthropic",
        api_base="https://example.invalid",
        image="python:3.11-slim",
        image_digest="sha256:test",
        dependency_lock="lock.txt",
        artifact_root=tmp_path / "artifacts",
        log_root=tmp_path / "logs",
    )
    task = TaskSpec("task", "bug_fixing", {"app.py": "x"})
    first = pairing_contract_hash(task, settings, seed=1, phase="gate", role="promotion")
    second = pairing_contract_hash(task, settings, seed=1, phase="gate", role="promotion")
    assert first == second
    assert len(first) == 64


def test_strategic_registered_tests_use_pinned_isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strategic response tests run in the locked runtime, not the host shell."""

    import importlib

    from experiments.v15.planes import TaskSpec

    run_strategic = importlib.import_module("experiments.v15.run_strategic")

    task = TaskSpec(
        "strategic-task",
        "bug_fixing",
        {"test_probe.py": "print('ok')\n"},
        metadata={"test_command": "python test_probe.py"},
    )
    runtime = tmp_path / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_strategic, "locked_runtime_python", lambda _root: runtime)
    monkeypatch.setattr(run_strategic.shutil, "which", lambda command: "/usr/bin/bwrap" if command == "bwrap" else None)

    def fake_bwrap(**kwargs: object) -> list[str]:
        captured["sandbox"] = kwargs
        return ["/usr/bin/bwrap", "--", "/bin/bash", "-lc", str(kwargs["command"])]

    monkeypatch.setattr(run_strategic, "_test_sandbox_command", fake_bwrap)

    class Completed:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(invocation, **kwargs):
        captured["invocation"] = invocation
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(run_strategic.subprocess, "run", fake_run)
    assert run_strategic._run_registered_tests(task, workspace, root=tmp_path) == 0
    assert captured["sandbox"]
    assert "/runtime/python" in str(captured["sandbox"])
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "ANTHROPIC_AUTH_TOKEN" not in environment
