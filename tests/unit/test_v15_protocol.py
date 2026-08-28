from __future__ import annotations

from pathlib import Path

import pytest


def test_agent_policy_hash_is_content_stable_and_diff_is_structured() -> None:
    from experiments.v15.protocol import AgentPolicy

    incumbent = AgentPolicy(
        system_prompt="inspect then edit",
        agent_loop_config={"max_steps": 8},
        tool_policy={"shell": True},
        search_policy={"depth": 2},
        test_policy={"run_tests": True},
        context_policy={"max_tokens": 2048},
    )
    same = AgentPolicy(
        system_prompt="inspect then edit",
        agent_loop_config={"max_steps": 8},
        tool_policy={"shell": True},
        search_policy={"depth": 2},
        test_policy={"run_tests": True},
        context_policy={"max_tokens": 2048},
    )
    candidate = AgentPolicy(
        system_prompt="inspect then edit",
        agent_loop_config={"max_steps": 10},
        tool_policy={"shell": True},
        search_policy={"depth": 3},
        test_policy={"run_tests": True},
        context_policy={"max_tokens": 2048},
    )
    assert incumbent.policy_hash == same.policy_hash
    assert incumbent.policy_hash != candidate.policy_hash
    diff = incumbent.diff(candidate)
    assert diff["loop_parameter_delta"] == 2.0
    assert diff["search_policy_change"] == 1.0
    assert diff["tool_schema_change"] == 0.0
    assert incumbent.to_record()["policy_hash"] == incumbent.policy_hash


def test_transition_record_rejects_mismatched_hashes_and_serializes() -> None:
    from experiments.v15.protocol import AgentPolicy, TransitionRecord

    base = AgentPolicy.minimal()
    candidate = base.with_updates(agent_loop_config={"max_steps": 9})
    record = TransitionRecord(
        run_id="dev-1",
        scaffold="local-reference",
        operator="harness",
        task_family="bug_fixing",
        round_index=0,
        candidate_index=0,
        incumbent=base,
        candidate=candidate,
        delta_proxy=0.25,
        delta_actor=0.1,
        footprint=base.diff(candidate),
        resource_metrics={"tokens": 12},
    )
    payload = record.to_record()
    assert payload["incumbent_hash"] == base.policy_hash
    assert payload["candidate_hash"] == candidate.policy_hash
    restored = TransitionRecord.from_record(payload)
    assert restored.transition_id == record.transition_id
    payload["candidate_hash"] = "bad"
    with pytest.raises(ValueError, match="candidate_hash"):
        TransitionRecord.from_record(payload)


def test_write_table_emits_parquet_and_csv_with_schema(tmp_path: Path) -> None:
    from experiments.v15.protocol import write_table

    rows = [{"run_id": "r1", "delta_proxy": 0.2}, {"run_id": "r2", "delta_proxy": -0.1}]
    outputs = write_table(rows, tmp_path / "transitions", columns=("run_id", "delta_proxy"))
    assert outputs["csv"].is_file()
    assert outputs["parquet"].is_file()
    assert "r1" in outputs["csv"].read_text(encoding="utf-8")
    assert outputs["parquet"].stat().st_size > 0


def test_terminal_states_are_closed() -> None:
    from experiments.v15.protocol import validate_terminal_state

    assert validate_terminal_state("HYPOTHESIS_SUPPORTED")
    assert not validate_terminal_state("NOT_A_STATE")


def test_transition_record_round_trips_optional_level_scores() -> None:
    from experiments.v15.protocol import TERMINAL_STATES, AgentPolicy, TransitionRecord

    incumbent = AgentPolicy.minimal()
    candidate = incumbent.with_updates(system_prompt=incumbent.system_prompt + " Verify.")
    record = TransitionRecord(
        run_id="run",
        scaffold="mini-SWE-agent",
        operator="operator",
        task_family="bug_fixing",
        round_index=0,
        candidate_index=0,
        incumbent=incumbent,
        candidate=candidate,
        delta_proxy=0.2,
        proxy_incumbent_score=0.4,
        proxy_candidate_score=0.6,
        actor_incumbent_score=0.3,
        actor_candidate_score=0.5,
    )

    restored = TransitionRecord.from_record(record.to_record())

    assert restored.transition_id == record.transition_id
    assert restored.proxy_candidate_score == 0.6
    assert restored.actor_incumbent_score == 0.3
    assert "UNDERPOWERED" in TERMINAL_STATES
