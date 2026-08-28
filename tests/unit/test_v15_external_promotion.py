from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest


def _row(candidate_id: str = "candidate-a") -> dict[str, object]:
    policy = {
        "system_prompt": "Inspect the repository and run tests.",
        "agent_loop_config": {"max_steps": 8, "stop_on_failure": True},
        "tool_policy": {"shell": True, "read": True, "write": True},
        "search_policy": {"depth": 2, "max_files": 12},
        "test_policy": {"run_tests": True, "repair": False},
        "context_policy": {"max_tokens": 2048, "summarize": True},
        "metadata": {"candidate": candidate_id},
    }
    from experiments.v15.protocol import AgentPolicy

    incumbent = AgentPolicy.minimal().to_record()
    candidate = AgentPolicy.from_record(policy).to_record()
    return {
        "run_id": "run-1",
        "round": 0,
        "candidate_index": 0,
        "candidate_id": candidate_id,
        "candidate_hash": candidate["policy_hash"],
        "incumbent_hash": incumbent["policy_hash"],
        "proxy_delta": 0.2,
        "operator": "harness_skill_evolution",
        "scaffold": "mini-SWE-agent",
        "task_family": "bug_fixing",
        "seed": 10001,
        "incumbent_policy": incumbent,
        "candidate_policy": candidate,
        "footprint": {"prompt_semantic_distance": 0.1},
    }


def test_external_archive_requires_replayable_policy_pairs() -> None:
    from experiments.v15.external_promotion import prepare_candidate_rows

    rows = prepare_candidate_rows([_row()])

    assert rows[0]["incumbent_policy"]["policy_hash"] == rows[0]["incumbent_hash"]
    assert rows[0]["candidate_policy"]["policy_hash"] == rows[0]["candidate_hash"]


def test_external_archive_rejects_hidden_outcome_fields() -> None:
    from experiments.v15.external_promotion import prepare_candidate_rows

    row = _row()
    row["delta_actor"] = 0.4

    with pytest.raises(ValueError, match="hidden outcome"):
        prepare_candidate_rows([row])


def test_external_query_plan_uses_registered_budget_and_candidates() -> None:
    from experiments.v15.external_promotion import build_query_plan

    rows = [_row("a"), _row("b")]
    rows[1]["candidate_index"] = 1
    rows[1]["proxy_delta"] = 0.1
    plan = build_query_plan(rows, method="PIVOT-VOI", budget=1, seed=3)

    assert len(plan) == 1
    assert plan[0]["candidate_id"] in {"a", "b"}
    assert plan[0]["hf_cost"] == 1.0


def test_external_promotion_output_is_phase_scoped(tmp_path: Path) -> None:
    from experiments.v15.external_promotion import phase_output

    assert phase_output(tmp_path, confirmatory=False).name == "dev-external-promotion"
    assert phase_output(tmp_path, confirmatory=True).name == "external-promotion"


def test_external_promotion_counts_each_method_budget_as_logical_hf_query(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    from experiments.v15.external_promotion import run_external_promotion
    from experiments.v15.external_runtime import PairedExecutionRecord, RuntimeSettings
    from experiments.v15.planes import TaskSpec

    rows = [_row("a"), _row("b")]
    rows[1]["candidate_index"] = 1
    archive = tmp_path / "results/v15/dev-external-transition-audit/promotion_candidates.jsonl"
    archive.parent.mkdir(parents=True)
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    gate = TaskSpec("gate-1", "bug_fixing", {"test.py": ""})

    class FakePlanes:
        access_log: ClassVar[list[dict[str, object]]] = []

        def tasks(self, plane: str, role: str):
            return (gate,) if plane == "gate" else ()

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
    monkeypatch.setattr("experiments.v15.external_promotion.load_task_planes", lambda _: FakePlanes())
    monkeypatch.setattr("experiments.v15.external_promotion._settings", lambda *args, **kwargs: settings)

    def fake_pair(tasks, incumbent, candidate, settings, *, seed, phase, role, run_id):
        return [
            PairedExecutionRecord(
                task_id=gate.task_id,
                task_hash=gate.task_hash,
                seed=seed,
                incumbent_policy_hash=incumbent.policy_hash,
                candidate_policy_hash=candidate.policy_hash,
                incumbent_success=0.0,
                candidate_success=1.0 if candidate.metadata.get("candidate") == "a" else 0.0,
                incumbent_execution="inc.jsonl",
                candidate_execution="cand.jsonl",
                inspect_log=None,
            )
        ]

    monkeypatch.setattr("experiments.v15.external_promotion.evaluate_paired_with_inspect", fake_pair)
    result = run_external_promotion(tmp_path, budgets=(1, 2, 4))

    # Two candidates: the registered methods request 31 logical observations
    # across the three budgets; physical execution may reuse immutable paired
    # truth for the post-decision audit but never reduces logical HF cost.
    assert result["logical_hf_queries"] == 31
    assert result["candidate_count"] == 2
    assert result["physical_pair_evaluations"] == 2
    promotion_rows = [
        json.loads(line)
        for line in (tmp_path / "results/v15/dev-external-promotion/promotion_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len({row["candidate_batch_hash"] for row in promotion_rows}) == 1


def test_external_promotion_query_ledger_marks_logical_and_physical_evidence() -> None:
    """A cached truth label must remain a logical query for each method."""

    from experiments.v15.external_promotion import _query_ledger_record

    row = _query_ledger_record(
        phase="DEV",
        method="PIVOT-VOI",
        candidate_row={"run_id": "run-1", "round": 0, "candidate_id": "a", "candidate_hash": "hash-a"},
        query_index=0,
        paired_delta=0.25,
        posterior_before=0.10,
        posterior_after=0.25,
        candidate_batch_hash="batch",
        cache_hit=True,
        evsi=0.03,
    )

    assert row["logical_hf_query"] is True
    assert row["physical_pair_evaluation"] is False
    assert row["cache_hit"] is True
    assert row["EVSI"] == 0.03
    assert row["observed_information_gain"] == pytest.approx(0.15)
