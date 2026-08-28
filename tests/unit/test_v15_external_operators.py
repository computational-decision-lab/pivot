from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _settings(tmp_path: Path):
    from experiments.v15.external_runtime import RuntimeSettings

    return RuntimeSettings(
        model_name="anthropic/test-model",
        provider="anthropic",
        api_base="https://example.invalid",
        image="python:3.11-slim",
        image_digest="sha256:test",
        dependency_lock="lock.txt",
        artifact_root=tmp_path / "a",
        log_root=tmp_path / "l",
        model_output_tokens=256,
    )


def test_external_operator_parses_restricted_candidate_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.v15.external_operators import ExternalProposalOperator
    from experiments.v15.operators import ProposalContext
    from experiments.v15.protocol import AgentPolicy

    captured: dict[str, str] = {}

    class Message:
        content = '[{"system_prompt": "Inspect callers before editing.", "search_policy": {"depth": 3}}, {"test_policy": {"repair": true}}]'

    class Choice:
        message = Message()

    class Response:
        def __init__(self) -> None:
            self.choices = [Choice()]

    def fake_completion(**kwargs: object) -> Response:
        captured["prompt"] = str(kwargs["messages"])
        return Response()

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=fake_completion))
    operator = ExternalProposalOperator(
        name="harness_skill_evolution",
        focus="Modify instructions and search workflow from proxy diagnostics.",
        allowed_fields=("system_prompt", "search_policy", "test_policy"),
        settings=_settings(tmp_path),
    )
    candidates = operator.propose(
        AgentPolicy.minimal(),
        ProposalContext(proxy_score=0.5, proxy_feedback={"failed_tests": 1}, round_index=2, seed=9),
        count=2,
    )

    assert len(candidates) == 2
    assert all(candidate.policy_hash != AgentPolicy.minimal().policy_hash for candidate in candidates)
    assert all(candidate.metadata["operator"] == operator.name for candidate in candidates)
    assert "gate" not in captured["prompt"].casefold()
    assert "assessment" not in captured["prompt"].casefold()
    assert "strategic" not in captured["prompt"].casefold()


def test_external_operator_rejects_edits_outside_registered_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.v15.external_operators import ExternalProposalOperator
    from experiments.v15.operators import ProposalContext
    from experiments.v15.protocol import AgentPolicy

    class Response:
        def __init__(self) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": '[{"tool_policy": {"shell": false}}]'})()},
                )()
            ]

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **_: Response()))
    operator = ExternalProposalOperator(
        name="harness_skill_evolution",
        focus="Modify instructions from proxy diagnostics.",
        allowed_fields=("system_prompt",),
        settings=_settings(tmp_path),
    )

    with pytest.raises(ValueError, match="not allowed"):
        operator.propose(
            AgentPolicy.minimal(),
            ProposalContext(proxy_score=0.5, proxy_feedback={}, round_index=0, seed=1),
            count=1,
        )


def test_external_operator_merges_partial_mapping_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.v15.external_operators import ExternalProposalOperator
    from experiments.v15.operators import ProposalContext
    from experiments.v15.protocol import AgentPolicy

    class Response:
        def __init__(self) -> None:
            self.choices = [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": '[{"search_policy": {"depth": 5}}]'})()},
                )()
            ]

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **_: Response()))
    baseline = AgentPolicy.minimal()
    operator = ExternalProposalOperator(
        name="test-operator",
        focus="Modify search workflow from proxy diagnostics.",
        allowed_fields=("search_policy",),
        settings=_settings(tmp_path),
    )

    candidate = operator.propose(
        baseline,
        ProposalContext(proxy_score=0.5, proxy_feedback={}, round_index=0, seed=1),
        count=1,
    )[0]

    assert candidate.search_policy["depth"] == 5
    assert candidate.search_policy["max_files"] == baseline.search_policy["max_files"]


def test_operator_input_audit_rejects_hidden_task_identity() -> None:
    from experiments.v15.external_operators import assert_public_operator_input
    from experiments.v15.operators import ProposalContext
    from experiments.v15.protocol import AgentPolicy

    context = ProposalContext(
        proxy_score=0.5,
        proxy_feedback={"failed_task_ids": ["gate-bug-001"]},
        round_index=0,
        seed=1,
    )
    with pytest.raises(RuntimeError, match="sealed task_id"):
        assert_public_operator_input(
            AgentPolicy.minimal(),
            context,
            [{"task_id": "gate-bug-001", "task_hash": "hidden-hash"}],
        )


def test_operator_input_audit_allows_proxy_only_payload() -> None:
    from experiments.v15.external_operators import assert_public_operator_input
    from experiments.v15.operators import ProposalContext
    from experiments.v15.protocol import AgentPolicy

    context = ProposalContext(
        proxy_score=0.5,
        proxy_feedback={"failed_task_ids": ["proxy-bug-001"]},
        round_index=0,
        seed=1,
    )
    assert_public_operator_input(
        AgentPolicy.minimal(),
        context,
        [{"task_id": "assessment-bug-001", "task_hash": "hidden-hash"}],
    )
