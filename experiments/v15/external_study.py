"""Registered autonomous transition audit for the primary coding scaffold.

The runner deliberately separates proposal generation (proxy-only) from hidden
paired deployment evaluation.  It can be bounded for DEV validation and only
opens confirmatory tasks after the immutable pre-outcome lock is verified by
the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .confirmatory_guards import (
    registered_counts,
    reject_confirmatory_overrides,
    reject_existing_confirmatory_output,
    require_registered_count,
)
from .external_operators import ExternalProposalOperator, assert_public_operator_input
from .external_promotion import _lock_protocol_inputs
from .external_runtime import (
    ExecutionRecord,
    PairedExecutionRecord,
    RuntimeSettings,
    evaluate_paired_with_inspect,
    evaluate_with_inspect,
    paired_execution_failed,
    resolve_runtime_settings,
)
from .operators import ProposalContext
from .planes import TaskSpec, load_task_planes
from .protocol import (
    AgentPolicy,
    TransitionRecord,
    content_hash,
    file_hash,
    write_jsonl,
    write_table,
)


def phase_output(root: Path, *, confirmatory: bool) -> Path:
    """Return the phase-specific output directory without implicit overwrites."""

    return Path(root) / "results/v15" / (
        "external-transition-audit" if confirmatory else "dev-external-transition-audit"
    )


def family_success(tasks: Sequence[TaskSpec], records: Sequence[ExecutionRecord]) -> dict[str, float]:
    """Aggregate task success by task family for diagnostics."""

    family_by_id = {task.task_id: task.family for task in tasks}
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        family = family_by_id.get(record.task_id)
        if family is not None:
            values[family].append(float(record.success))
    return {family: sum(scores) / len(scores) for family, scores in sorted(values.items()) if scores}


def family_pair_deltas(
    tasks: Sequence[TaskSpec], pairs: Sequence[PairedExecutionRecord]
) -> dict[str, float]:
    """Aggregate paired candidate-minus-incumbent differences by family."""

    family_by_id = {task.task_id: task.family for task in tasks}
    values: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        family = family_by_id.get(pair.task_id)
        if family is not None:
            values[family].append(pair.delta)
    return {family: sum(scores) / len(scores) for family, scores in sorted(values.items()) if scores}


def _mean_success(records: Sequence[ExecutionRecord]) -> float:
    return sum(float(record.success) for record in records) / len(records) if records else 0.0


def _execution_metrics(records: Sequence[ExecutionRecord]) -> dict[str, float]:
    keys = (
        "tokens",
        "tool_calls",
        "tests_executed",
        "files_read",
        "files_written",
        "context_peak",
        "wall_clock_seconds",
        "dependency_operations",
    )
    return {
        key: sum(float(record.resource_metrics.get(key, 0.0)) for record in records) / max(len(records), 1)
        for key in keys
    }


def _behavioral_footprint(
    incumbent_records: Sequence[ExecutionRecord], candidate_records: Sequence[ExecutionRecord]
) -> dict[str, float]:
    left = _execution_metrics(incumbent_records)
    right = _execution_metrics(candidate_records)
    return {
        "tool_call_distribution_shift": right["tool_calls"] - left["tool_calls"],
        "shell_command_distribution_shift": right["tool_calls"] - left["tool_calls"],
        "test_execution_shift": right["tests_executed"] - left["tests_executed"],
        "files_read_shift": right["files_read"] - left["files_read"],
        "files_written_shift": right["files_written"] - left["files_written"],
        "dependency_operation_shift": right["dependency_operations"] - left["dependency_operations"],
        "token_usage_shift": right["tokens"] - left["tokens"],
        "context_peak_shift": right["context_peak"] - left["context_peak"],
        "wall_clock_shift": right["wall_clock_seconds"] - left["wall_clock_seconds"],
        "action_sequence_distance": float(
            sum(record.trace != other.trace for record, other in zip(incumbent_records, candidate_records))
        ),
    }


def _paired_behavioral_footprint(pair: PairedExecutionRecord) -> dict[str, float]:
    """Compute behavioral shifts from the already paired execution records."""

    left = dict(pair.incumbent_resource_metrics or {})
    right = dict(pair.candidate_resource_metrics or {})
    keys = {
        "tool_call_distribution_shift": "tool_calls",
        "test_execution_shift": "tests_executed",
        "files_read_shift": "files_read",
        "files_written_shift": "files_written",
        "dependency_operation_shift": "dependency_operations",
        "token_usage_shift": "tokens",
        "context_peak_shift": "context_peak",
        "wall_clock_shift": "wall_clock_seconds",
    }
    output = {
        name: float(right.get(key, 0.0)) - float(left.get(key, 0.0))
        for name, key in keys.items()
    }
    output["shell_command_distribution_shift"] = output["tool_call_distribution_shift"]
    output["action_sequence_distance"] = float(pair.incumbent_trace != pair.candidate_trace)
    return output


def registered_operators(settings: RuntimeSettings) -> tuple[ExternalProposalOperator, ...]:
    """Build the two frozen proposal mechanisms from the protocol contract."""

    return (
        ExternalProposalOperator(
            name="harness_skill_evolution",
            focus="Modify instructions, skills, search, testing, and planning workflow from proxy diagnostics.",
            allowed_fields=("system_prompt", "search_policy", "test_policy"),
            settings=settings,
        ),
        ExternalProposalOperator(
            name="mutation_self_edit",
            focus="Modify prompt/config, loop parameters, tool policy, and context policy from proxy diagnostics.",
            allowed_fields=("system_prompt", "agent_loop_config", "tool_policy", "context_policy"),
            settings=settings,
        ),
    )


def _task_feedback(records: Sequence[ExecutionRecord]) -> dict[str, Any]:
    failed = [record.task_id for record in records if record.success < 0.5]
    return {
        "failed_tests": len(failed),
        "failed_task_ids": failed,
        "mean_success": _mean_success(records),
        "resource_metrics": _execution_metrics(records),
    }


def _config_hash(root: Path) -> str:
    paths = (
        root / "configs/v15/confirmatory.yaml",
        root / "configs/v15/task_manifest.json",
        root / "configs/v15/external_versions.json",
    )
    return content_hash({str(path.relative_to(root)): file_hash(path) for path in paths})


def _write_manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(payload)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_transition_audit(
    root: Path,
    *,
    confirmatory: bool = False,
    trajectory_limit: int | None = None,
    round_limit: int | None = None,
    candidates_per_round: int | None = None,
    task_limit: int | None = None,
    agent_steps: int | None = None,
    verify_image: bool = False,
) -> dict[str, Any]:
    """Run registered autonomous proposals and paired deployment audits.

    The default is a bounded DEV run only when limits are supplied.  A
    confirmatory invocation requires an explicit environment opt-in so a
    normal audit command cannot spend model budget accidentally.
    """

    root = Path(root).resolve()
    if confirmatory and os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
        raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
    reject_confirmatory_overrides(
        confirmatory,
        trajectory_limit=trajectory_limit,
        round_limit=round_limit,
        candidates_per_round=candidates_per_round,
        task_limit=task_limit,
        agent_steps=agent_steps,
    )
    if confirmatory and not verify_image:
        raise ValueError("confirmatory execution requires sandbox image verification")
    lock = _lock_protocol_inputs(root) if confirmatory else None
    config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("confirmatory config must be a mapping")
    planes = load_task_planes(root / "configs/v15/task_manifest.json")
    settings = resolve_runtime_settings(
        root,
        artifact_root=phase_output(root, confirmatory=confirmatory) / "artifacts",
        log_root=phase_output(root, confirmatory=confirmatory) / "inspect-logs",
        verify_image=verify_image,
    )
    if agent_steps is not None:
        if agent_steps <= 0:
            raise ValueError("agent_steps must be positive")
        settings = replace(settings, agent_step_limit=int(agent_steps))
    proxy_tasks = planes.tasks("proxy", role="operator")
    gate_tasks = planes.tasks("gate", role="promotion")
    # Only identifiers/hashes are retained for this audit.  Task contents stay
    # inside the sealed plane and are never serialized into operator input.
    hidden_task_descriptors = tuple(
        descriptor
        for plane_name in ("gate", "assessment")
        for descriptor in planes.manifest().get(plane_name, [])
    )
    if not confirmatory:
        proxy_tasks = proxy_tasks[: task_limit or None]
        gate_tasks = gate_tasks[: task_limit or None]
    operators = registered_operators(settings)
    requested_trajectories = int(config.get("seed_registry", {}).get("trajectory_count_per_operator_family", 30))
    requested_rounds = int(config.get("rounds", 30))
    requested_candidates = int(config.get("candidates_per_round", 4))
    if trajectory_limit is None and round_limit is None and not confirmatory:
        raise ValueError("DEV execution requires an explicit trajectory_limit or round_limit")
    trajectories = min(requested_trajectories, trajectory_limit or requested_trajectories)
    rounds = min(requested_rounds, round_limit or requested_rounds)
    candidates_count = min(requested_candidates, candidates_per_round or requested_candidates)
    counts = registered_counts(config, operator_count=len(operators))
    require_registered_count(confirmatory, trajectories * len(operators), counts["trajectories"], "trajectory")
    require_registered_count(confirmatory, rounds, counts["rounds"], "round")
    require_registered_count(confirmatory, candidates_count, counts["candidates"], "candidate")
    output = phase_output(root, confirmatory=confirmatory)
    reject_existing_confirmatory_output(output, confirmatory)
    if confirmatory:
        lock = _lock_protocol_inputs(root, phase="transition_audit")
    seeds = [int(value) for value in config.get("seed_registry", {}).get("seed_blocks", [10001, 10101])]
    transitions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    model_calls = 0
    container_executions = 0
    execution_failures = 0
    operator_input_checks = 0
    for operator_index, operator in enumerate(operators):
        for trajectory_index in range(trajectories):
            seed = seeds[operator_index % len(seeds)] + trajectory_index
            incumbent = AgentPolicy.minimal().with_updates(
                metadata={"scaffold": "mini-SWE-agent", "operator": operator.name, "trajectory": str(trajectory_index)}
            )
            run_id = f"{operator.name}-trajectory-{trajectory_index:03d}"
            for round_index in range(rounds):
                incumbent_proxy = evaluate_with_inspect(
                    proxy_tasks,
                    incumbent,
                    settings,
                    seed=seed + round_index,
                    phase="transition_proxy",
                    role="proxy_evaluator",
                    run_id=run_id,
                )
                execution_failures += sum(record.status != "COMPLETED" for record in incumbent_proxy)
                context = ProposalContext(
                    proxy_score=_mean_success(incumbent_proxy),
                    proxy_feedback=_task_feedback(incumbent_proxy),
                    round_index=round_index,
                    seed=seed,
                    resource_budget=settings.tool_calls,
                )
                assert_public_operator_input(incumbent, context, hidden_task_descriptors)
                operator_input_checks += 1
                proposed = operator.propose(incumbent, context, count=candidates_count)
                candidate_proxy_scores: list[tuple[AgentPolicy, float, list[ExecutionRecord]]] = []
                for candidate_index, candidate in enumerate(proposed):
                    candidate_proxy = evaluate_with_inspect(
                        proxy_tasks,
                        candidate,
                        settings,
                        seed=seed + round_index,
                        phase="transition_proxy",
                        role="proxy_evaluator",
                        run_id=f"{run_id}-candidate-{candidate_index}",
                    )
                    execution_failures += sum(record.status != "COMPLETED" for record in candidate_proxy)
                    candidate_proxy_score = _mean_success(candidate_proxy)
                    candidate_proxy_scores.append((candidate, candidate_proxy_score, candidate_proxy))
                    paired = evaluate_paired_with_inspect(
                        gate_tasks,
                        incumbent,
                        candidate,
                        settings,
                        seed=seed + round_index,
                        phase="transition_actor",
                        role="promotion",
                        run_id=f"{run_id}-candidate-{candidate_index}",
                    )
                    proxy_incumbent_score = _mean_success(incumbent_proxy)
                    actor_incumbent_score = (
                        sum(pair.incumbent_success for pair in paired) / max(len(paired), 1)
                    )
                    actor_candidate_score = (
                        sum(pair.candidate_success for pair in paired) / max(len(paired), 1)
                    )
                    actor_delta = sum(pair.delta for pair in paired) / max(len(paired), 1)
                    actor_by_family = family_pair_deltas(gate_tasks, paired)
                    proxy_by_family = family_success(proxy_tasks, candidate_proxy)
                    proxy_incumbent_by_family = family_success(proxy_tasks, incumbent_proxy)
                    execution_failures += sum(paired_execution_failed(pair) for pair in paired)
                    footprint = {**incumbent.diff(candidate)}
                    for pair in paired:
                        for name, value in _paired_behavioral_footprint(pair).items():
                            footprint[name] = footprint.get(name, 0.0) + value / max(len(paired), 1)
                    record = TransitionRecord(
                        run_id=run_id,
                        scaffold="mini-SWE-agent",
                        operator=operator.name,
                        task_family="mixed",
                        round_index=round_index,
                        candidate_index=candidate_index,
                        incumbent=incumbent,
                        candidate=candidate,
                        delta_proxy=candidate_proxy_score - _mean_success(incumbent_proxy),
                        delta_actor=actor_delta,
                        proxy_incumbent_score=proxy_incumbent_score,
                        proxy_candidate_score=candidate_proxy_score,
                        actor_incumbent_score=actor_incumbent_score,
                        actor_candidate_score=actor_candidate_score,
                        footprint=footprint,
                        resource_metrics={
                            "proxy_incumbent": _execution_metrics(incumbent_proxy),
                            "proxy_candidate": _execution_metrics(candidate_proxy),
                            "paired_tasks": len(paired),
                            "actor_task_ids": [pair.task_id for pair in paired],
                            "incumbent_final_tree_paths": [pair.incumbent_final_tree_path for pair in paired],
                            "candidate_final_tree_paths": [pair.candidate_final_tree_path for pair in paired],
                            "incumbent_trajectories": [pair.incumbent_execution for pair in paired],
                            "candidate_trajectories": [pair.candidate_execution for pair in paired],
                            "pairing_contract_hashes": [pair.pairing_contract_hash for pair in paired],
                            "initial_sandbox_hashes": [
                                pair.incumbent_initial_tree_hash for pair in paired
                            ],
                            "task_families": sorted({task.family for task in (*proxy_tasks, *gate_tasks)}),
                            "proxy_candidate_by_family": proxy_by_family,
                            "proxy_incumbent_by_family": proxy_incumbent_by_family,
                            "actor_delta_by_family": actor_by_family,
                        },
                        seed=seed + round_index,
                        paired_seed_ids=(seed + round_index,),
                        source_digest=content_hash(candidate.to_record()),
                        config_hash=_config_hash(root),
                    )
                    transitions.append(record.to_record())
                    candidates.append(
                        {
                            "run_id": run_id,
                            "round": round_index,
                            "candidate_index": candidate_index,
                            "candidate_id": record.transition_id,
                            "candidate_hash": candidate.policy_hash,
                            "incumbent_hash": incumbent.policy_hash,
                            "proxy_delta": record.delta_proxy,
                            "operator": operator.name,
                            "scaffold": "mini-SWE-agent",
                            "task_family": "mixed",
                            "candidate_policy": candidate.to_record(),
                            "incumbent_policy": incumbent.to_record(),
                            "footprint": dict(record.footprint),
                            "seed": seed + round_index,
                        }
                    )
                    model_calls += int(sum(record_.resource_metrics.get("model_calls", 0.0) for record_ in incumbent_proxy))
                    model_calls += int(sum(record_.resource_metrics.get("model_calls", 0.0) for record_ in candidate_proxy))
                    model_calls += int(sum(pair.incumbent_resource_metrics.get("model_calls", 0.0) for pair in paired if pair.incumbent_resource_metrics))
                    model_calls += int(sum(pair.candidate_resource_metrics.get("model_calls", 0.0) for pair in paired if pair.candidate_resource_metrics))
                    container_executions += len(incumbent_proxy) + len(candidate_proxy) + 2 * len(paired)
                if candidate_proxy_scores:
                    incumbent = max(candidate_proxy_scores, key=lambda item: item[1])[0]
    write_jsonl(transitions, output / "autonomous_transitions.jsonl")
    write_jsonl(candidates, output / "promotion_candidates.jsonl")
    write_table(
        transitions,
        output / "autonomous_transitions",
        columns=(
            "transition_id", "run_id", "operator", "round", "delta_proxy", "delta_actor",
            "proxy_incumbent_score", "proxy_candidate_score", "actor_incumbent_score",
            "actor_candidate_score", "actor_reversal", "footprint", "resource_metrics", "config_hash",
        ),
    )
    write_table(candidates, output / "promotion_candidates", columns=("run_id", "round", "candidate_id", "candidate_hash", "incumbent_hash", "proxy_delta", "operator", "scaffold", "task_family", "seed", "incumbent_policy", "candidate_policy", "footprint"))
    from .evidence import freeze_candidate_archive

    archive_root = root / "results/v15" / (
        "external-candidate-archive" if confirmatory else "dev-external-candidate-archive"
    )
    archive_manifest = freeze_candidate_archive(
        output / "promotion_candidates.jsonl",
        archive_root,
        phase="CONFIRMATORY" if confirmatory else "DEV",
        confirmatory=confirmatory,
        source_manifest_sha256=None,
    )
    return _write_manifest(
        output,
        {
            "phase": "CONFIRMATORY" if confirmatory else "DEV",
            "confirmatory": confirmatory,
            "status": "COMPLETED" if execution_failures == 0 else "IMPLEMENTATION_FAILURE",
            "terminal_state": (
                "IMPLEMENTATION_FAILURE"
                if execution_failures
                else "UNDERPOWERED"
                if not confirmatory
                else None
            ),
            "execution_attempted": True,
            "design_status": "VALIDATED_DEV" if not confirmatory and execution_failures == 0 else "PENDING_ANALYSIS",
            "leakage_detected": False,
            "operator_input_audit": {
                "checks": operator_input_checks,
                "hidden_descriptor_count": len(hidden_task_descriptors),
                "sealed_outcomes_in_input": False,
            },
            "outcome_chasing": False,
            "trajectory_count": trajectories * len(operators),
            "operator_count": len(operators),
            "round_count": rounds,
            "candidate_count": len(candidates),
            "transition_count": len(transitions),
            "independent_unit": "trajectory_or_task_cluster",
            "model_calls_performed": model_calls,
            "container_executions": container_executions,
            "execution_failure_count": execution_failures,
            "proxy_task_count": len(proxy_tasks),
            "gate_task_count": len(gate_tasks),
            "role_access_log": list(planes.access_log),
            "candidate_archive_frozen": True,
            "candidate_archive_path": str((archive_root / "promotion_candidates.jsonl").relative_to(root)),
            "candidate_archive_sha256": archive_manifest.get("archive_sha256"),
            "candidate_archive_manifest_sha256": file_hash(archive_root / "manifest.json"),
            "task_manifest_sha256": file_hash(root / "configs/v15/task_manifest.json"),
            "config_hash": _config_hash(root),
            "runtime": settings.to_manifest(),
            "lock_hash": lock.get("lock_hash") if lock else None,
            "note": "Actor outcomes are stored after proxy-only proposal generation and never returned to the operator. Any execution failure is terminal and is not relabelled as a scientific result.",
        },
    )


__all__ = [
    "family_pair_deltas",
    "family_success",
    "phase_output",
    "registered_operators",
    "run_transition_audit",
]
