from __future__ import annotations

from pathlib import Path


def test_pi_runtime_command_uses_project_build(tmp_path: Path) -> None:
    from experiments.v15.pi_runtime import pi_runtime_status

    status = pi_runtime_status(Path("."))
    assert status["source_commit"]
    assert status["cli_path"]
    assert status["node_available"] is True


def test_pi_runtime_policy_prompt_is_stable() -> None:
    from experiments.v15.pi_runtime import pi_policy_prompt
    from experiments.v15.protocol import AgentPolicy

    prompt = pi_policy_prompt(AgentPolicy.minimal())
    assert "registered tests" in prompt
    assert "evaluator files" in prompt


def test_pi_runtime_reports_filesystem_sandbox_capability() -> None:
    from experiments.v15.pi_runtime import pi_runtime_status

    status = pi_runtime_status(Path("."))

    assert "bwrap_available" in status
    assert status["bwrap_available"] is True
