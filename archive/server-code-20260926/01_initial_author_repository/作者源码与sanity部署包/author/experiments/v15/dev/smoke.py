"""Development-only construct and resource smoke for the modern-agent layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..canonical import FOOTPRINT_COLUMNS, RESOURCE_COLUMNS
from ..operators import HarnessSkillEvolution, MutationSelfEdit, ProposalContext, ProposalOperator
from ..planes import AccessDenied, SealedDataPlanes, TaskSpec, load_task_planes
from ..protocol import (
    AgentPolicy,
    TransitionRecord,
    content_hash,
    file_hash,
    write_jsonl,
    write_table,
)
from ..sandbox import PairedSandboxRunner


def default_planes(manifest_path: Path | None = None) -> SealedDataPlanes:
    """Load the frozen task manifest, with a tiny fallback for isolated tests."""

    candidate = manifest_path or Path(__file__).resolve().parents[3] / "configs/v15/task_manifest.json"
    if candidate.is_file():
        return load_task_planes(candidate)

    def task(task_id: str, family: str) -> TaskSpec:
        if family == "bug_fixing":
            return TaskSpec(
                task_id,
                family,
                {"app.py": "BUG = True\n", "test_app.py": "assert not BUG\n"},
                {"target": "app.py"},
            )
        return TaskSpec(
            task_id,
            family,
            {"config.ini": "enabled=false\n", "README.md": "Enable the feature.\n"},
            {"target": "config.ini"},
        )

    return SealedDataPlanes(
        proxy=(task("proxy-bug-001", "bug_fixing"), task("proxy-tools-001", "tool_context")),
        gate=(task("gate-bug-001", "bug_fixing"), task("gate-tools-001", "tool_context")),
        assessment=(
            task("assessment-bug-001", "bug_fixing"),
            task("assessment-tools-001", "tool_context"),
        ),
    )


def run_smoke(output: Path, *, seed: int = 10001, candidates_per_operator: int = 2) -> dict[str, Any]:
    """Run construct checks and emit DEV-only canonical transition artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    planes = default_planes()
    proxy_tasks = planes.tasks("proxy", role="operator")
    gate_tasks = planes.tasks("gate", role="promotion")
    assessment_tasks = planes.tasks("assessment", role="terminal_assessor")
    denied_accesses = 0
    for plane, role in (("gate", "operator"), ("assessment", "pivot")):
        try:
            planes.tasks(plane, role=role)
        except AccessDenied:
            denied_accesses += 1
    operators: list[ProposalOperator] = [
        HarnessSkillEvolution(),
        MutationSelfEdit(),
    ]
    runner = PairedSandboxRunner()
    incumbent = AgentPolicy.minimal()
    transitions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for operator_index, operator in enumerate(operators):
        for task_index, task in enumerate(proxy_tasks):
            context = ProposalContext(
                proxy_score=0.25 + task_index * 0.1,
                proxy_feedback={"failed_tests": task_index + 1},
                round_index=0,
                seed=seed + operator_index,
            )
            proposals = operator.propose(incumbent, context, count=candidates_per_operator)
            for candidate_index, candidate in enumerate(proposals):
                pair = runner.evaluate_pair(task, incumbent, candidate, seed=seed + candidate_index)
                footprint = {**incumbent.diff(candidate), **pair.behavioral_footprint}
                record = TransitionRecord(
                    run_id=f"dev-{operator.name}-{task.task_id}",
                    scaffold="local-reference",
                    operator=operator.name,
                    task_family=task.family,
                    round_index=0,
                    candidate_index=candidate_index,
                    incumbent=incumbent,
                    candidate=candidate,
                    delta_proxy=float(candidate_index + 1) / 10.0,
                    delta_actor=pair.delta,
                    proxy_incumbent_score=0.0,
                    proxy_candidate_score=float(candidate_index + 1) / 10.0,
                    actor_incumbent_score=pair.incumbent.success,
                    actor_candidate_score=pair.candidate.success,
                    footprint=footprint,
                    resource_metrics={
                        "proxy_task_hash": task.task_hash,
                        "gate_task_count": len(gate_tasks),
                        "assessment_task_count": len(assessment_tasks),
                        "candidate_trace": list(pair.candidate.trace),
                        "initial_manifest_hash": pair.incumbent.initial_manifest_hash,
                    },
                    seed=seed + candidate_index,
                    paired_seed_ids=(seed + candidate_index,),
                    source_digest=content_hash(candidate.to_record()),
                    config_hash=content_hash({"phase": "DEV", "operator": operator.name}),
                )
                transitions.append(record.to_record())
                candidates.append(
                    {
                        "run_id": record.run_id,
                        "operator": operator.name,
                        "scaffold": "local-reference",
                        "task_family": task.family,
                        "round": 0,
                        "candidate_index": candidate_index,
                        "candidate_hash": candidate.policy_hash,
                        "incumbent_hash": incumbent.policy_hash,
                        "proxy_delta": record.delta_proxy,
                        "candidate_policy": candidate.to_record(),
                    }
                )
    transition_outputs = write_table(
        transitions,
        output / "autonomous_transitions",
        columns=(
            "transition_id", "run_id", "scaffold", "operator", "task_family", "round",
            "candidate_index", "incumbent_hash", "candidate_hash", "delta_proxy", "delta_actor",
            "delta_strategic", "proxy_positive", "actor_reversal", "strategic_reversal",
            *(f"footprint_{name}" for name in FOOTPRINT_COLUMNS),
            *(f"resource_{name}" for name in RESOURCE_COLUMNS),
            "footprint", "resource_metrics", "seed", "config_hash", "terminal_state",
        ),
    )
    candidate_outputs = write_table(
        candidates,
        output / "promotion_candidates",
        columns=(
            "run_id", "round", "candidate_index", "candidate_hash", "incumbent_hash",
            "proxy_delta", "operator", "scaffold", "task_family", "candidate_policy",
        ),
    )
    write_jsonl(transitions, output / "autonomous_transitions.jsonl")
    write_jsonl(candidates, output / "promotion_candidates.jsonl")
    # The smoke exposes one aggregate transition per operator/task family;
    # individual proposal rows remain in the candidate archive for replay.
    # Confirmatory runs will count independent trajectories, never raw rows.
    manifest = {
        "phase": "DEV",
        # Scientific terminal states are reserved for completed analyses.
        # This smoke has no confirmatory independent-N, so its explicit
        # terminal state is UNDERPOWERED; design validity is tracked
        # separately to avoid inventing a sixth state.
        "terminal_state": "UNDERPOWERED",
        "design_status": "VALIDATED_DEV",
        "confirmatory": False,
        "outcome_chasing": False,
        "planes": planes.manifest(),
        "access_log": list(planes.access_log),
        "denied_access_checks": denied_accesses,
        "task_manifest_sha256": file_hash(
            Path(__file__).resolve().parents[3] / "configs/v15/task_manifest.json"
        )
        if (Path(__file__).resolve().parents[3] / "configs/v15/task_manifest.json").is_file()
        else None,
        "operator_count": len(operators),
        "transition_count": len(operators) * len({task.family for task in proxy_tasks}),
        "candidate_transition_rows": len(transitions),
        "candidate_count": len(candidates),
        "outputs": {name: str(path) for name, path in {**transition_outputs, **candidate_outputs}.items()},
        "note": "Construct-validity smoke only; no confirmatory claim or paper number is derived from this run.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local DEV construct smoke")
    parser.add_argument("--output", type=Path, default=Path("results/v15/dev-smoke"))
    parser.add_argument("--seed", type=int, default=10001)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output.resolve(), seed=args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
