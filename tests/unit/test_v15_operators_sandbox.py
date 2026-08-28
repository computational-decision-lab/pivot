from __future__ import annotations


def test_two_operators_generate_distinct_nontrivial_candidates() -> None:
    from experiments.v15.operators import (
        HarnessSkillEvolution,
        MutationSelfEdit,
        ProposalContext,
    )
    from experiments.v15.protocol import AgentPolicy

    policy = AgentPolicy.minimal()
    context = ProposalContext(proxy_score=0.4, proxy_feedback={"failed_tests": 2}, round_index=1, seed=7)
    a = HarnessSkillEvolution().propose(policy, context, count=3)
    b = MutationSelfEdit().propose(policy, context, count=3)
    assert len(a) == len(b) == 3
    assert all(item.policy_hash != policy.policy_hash for item in a + b)
    assert {item.metadata["edit_type"] for item in a} != {item.metadata["edit_type"] for item in b}


def test_paired_sandbox_uses_identical_initial_manifests() -> None:
    from experiments.v15.planes import TaskSpec
    from experiments.v15.protocol import AgentPolicy
    from experiments.v15.sandbox import PairedSandboxRunner

    task = TaskSpec(
        task_id="dev-task",
        family="bug_fixing",
        files={"app.py": "BUG = True\n", "test_app.py": "assert not BUG\n"},
        metadata={"target": "app.py"},
    )
    runner = PairedSandboxRunner()
    result = runner.evaluate_pair(task, AgentPolicy.minimal(), AgentPolicy.minimal().with_updates(
        test_policy={"run_tests": True, "repair": True}
    ), seed=11)
    assert result.incumbent.initial_manifest_hash == result.candidate.initial_manifest_hash
    assert result.incumbent.root_hash != result.candidate.root_hash
    assert result.candidate.resource_metrics["fresh_sandbox"] is True
    assert "action_sequence_distance" in result.behavioral_footprint
