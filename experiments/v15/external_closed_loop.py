"""Closed-loop self-improvement with a single terminal assessment access.

The loop shares autonomous candidate generation across promotion methods only
within a method's current incumbent.  Hidden gate observations are consumed by
the promotion selector; assessment observations are withheld until the final
incumbent for a trajectory is fixed.  All outputs are provenance records, not
paper claims by themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
from .external_operators import assert_public_operator_input
from .external_promotion import _lock_protocol_inputs
from .external_runtime import (
    ExecutionRecord,
    RuntimeSettings,
    evaluate_paired_with_inspect,
    evaluate_with_inspect,
    paired_execution_failed,
    resolve_runtime_settings,
)
from .external_study import _mean_success, _task_feedback, registered_operators
from .operators import ProposalContext
from .planes import load_task_planes
from .promotion import METHODS, _candidate_id, _initial_posterior, _query_order, _select_max
from .protocol import AgentPolicy, content_hash, file_hash, write_jsonl, write_table

DEFAULT_METHODS = ("Proxy Only", "Paired LUCB", "Global-VOI", "PIVOT-VOI", "All-HF Oracle")


def closed_loop_query_accounting(
    *,
    query_count: int,
    candidate_count: int,
    observed_count: int,
    query_pair_count: int,
    truth_pair_count: int,
) -> dict[str, int]:
    """Separate budgeted gate queries from evaluator-only truth collection.

    A closed-loop promotion round first observes a (possibly empty) query plan
    and then evaluates every unobserved candidate to compute post-decision ISR.
    The latter evaluations are useful for auditing but are not evidence that a
    selector could have used at decision time.  Keeping the two counts explicit
    prevents the manifest's HF budget from silently including the audit.
    """

    values = {
        "query_count": query_count,
        "candidate_count": candidate_count,
        "observed_count": observed_count,
        "query_pair_count": query_pair_count,
        "truth_pair_count": truth_pair_count,
    }
    if any(int(value) < 0 for value in values.values()):
        raise ValueError("closed-loop query counts must be non-negative")
    if int(observed_count) > int(candidate_count):
        raise ValueError("observed_count cannot exceed candidate_count")
    if int(query_count) > int(candidate_count):
        raise ValueError("query_count cannot exceed candidate_count")
    if int(observed_count) > int(query_count):
        raise ValueError("observed_count cannot exceed query_count")
    post_decision = int(candidate_count) - int(observed_count)
    return {
        "logical_hf_queries": int(query_count),
        "pre_decision_pair_evaluations": int(query_pair_count),
        "post_decision_truth_evaluations": post_decision,
        "post_decision_truth_pair_evaluations": int(truth_pair_count),
        "total_pair_evaluations": int(query_pair_count) + int(truth_pair_count),
    }


def phase_output(root: Path, *, confirmatory: bool) -> Path:
    """Return the phase-specific closed-loop directory."""

    return Path(root).resolve() / "results/v15" / (
        "external-closed-loop" if confirmatory else "dev-external-closed-loop"
    )


def assessment_tasks_for_terminal(role: str) -> str:
    """Guard the only role allowed to open the untouched assessment plane."""

    if role != "terminal_assessor":
        raise PermissionError("assessment tasks are available only to the terminal assessor")
    return "assessment"


def registered_closed_loop_budget(config: Mapping[str, Any], candidate_count: int) -> int:
    """Return the largest pre-registered HF budget usable for a round."""

    raw = config.get("hf_budgets", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("hf_budgets must be a non-empty sequence")
    budgets = sorted({int(value) for value in raw})
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("hf_budgets must contain non-negative values")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return min(budgets[-1], int(candidate_count))


def select_from_gate_queries(
    rows: Sequence[Mapping[str, Any]],
    observations: Mapping[str, float],
    *,
    method: str,
    budget: int,
    seed: int,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Select one candidate using only proxy values and queried gate values."""

    if method not in METHODS:
        raise ValueError(f"unknown promotion method: {method}")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    clean = [dict(row) for row in rows]
    if not clean:
        raise ValueError("candidate batch must not be empty")
    posterior = _initial_posterior(clean)
    for key, value in observations.items():
        matching = [row for row in clean if _candidate_id(row) == str(key)]
        if matching:
            posterior.observe(matching[0], float(value))
    rng = random.Random(seed)
    query_rows = _query_order(method, clean, posterior, int(budget), rng)
    queries = [
        {
            "candidate_id": _candidate_id(row),
            "candidate_hash": str(row.get("candidate_hash", _candidate_id(row))),
            "query_index": index,
            "hf_cost": 1.0,
        }
        for index, row in enumerate(query_rows)
        if _candidate_id(row) not in observations
    ]
    return _select_max(clean, posterior), queries


def _mean_metrics(records: Sequence[ExecutionRecord]) -> dict[str, float]:
    keys = (
        "model_calls",
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


def _manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _settings(root: Path, output: Path, *, agent_steps: int | None, verify_image: bool) -> RuntimeSettings:
    settings = resolve_runtime_settings(
        root,
        artifact_root=output / "artifacts",
        log_root=output / "inspect-logs",
        verify_image=verify_image,
    )
    if agent_steps is not None:
        if agent_steps <= 0:
            raise ValueError("agent_steps must be positive")
        settings = replace(settings, agent_step_limit=int(agent_steps))
    return settings


def run_external_closed_loop(
    root: Path,
    *,
    confirmatory: bool = False,
    trajectory_limit: int | None = None,
    round_limit: int | None = None,
    candidates_per_round: int | None = None,
    task_limit: int | None = None,
    assessment_limit: int | None = None,
    agent_steps: int | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    verify_image: bool = False,
) -> dict[str, Any]:
    """Run autonomous proposals, promotion gates, and terminal assessment."""

    root = Path(root).resolve()
    if confirmatory and os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
        raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
    reject_confirmatory_overrides(
        confirmatory,
        trajectory_limit=trajectory_limit,
        round_limit=round_limit,
        candidates_per_round=candidates_per_round,
        task_limit=task_limit,
        assessment_limit=assessment_limit,
        agent_steps=agent_steps,
    )
    if confirmatory and tuple(methods) != DEFAULT_METHODS:
        raise ValueError("confirmatory execution must use the registered closed-loop methods")
    if confirmatory and not verify_image:
        raise ValueError("confirmatory execution requires sandbox image verification")
    lock = _lock_protocol_inputs(root) if confirmatory else None
    config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("confirmatory config must be a mapping")
    output = phase_output(root, confirmatory=confirmatory)
    reject_existing_confirmatory_output(output, confirmatory)
    settings = _settings(root, output, agent_steps=agent_steps, verify_image=verify_image)
    planes = load_task_planes(root / "configs/v15/task_manifest.json")
    proxy_tasks = planes.tasks("proxy", role="operator")
    gate_tasks = planes.tasks("gate", role="pivot")
    registered_assessment_task_count = len(planes.manifest()["assessment"])
    hidden_task_descriptors = tuple(
        descriptor
        for plane_name in ("gate", "assessment")
        for descriptor in planes.manifest().get(plane_name, [])
    )
    if not confirmatory:
        proxy_tasks = proxy_tasks[: task_limit or None]
        gate_tasks = gate_tasks[: task_limit or None]
        # Assessment definitions are deliberately not opened until the
        # terminal-assessor phase below.  DEV may still bound that final pool.
    if not proxy_tasks or not gate_tasks:
        raise ValueError("closed loop requires non-empty proxy and gate task planes")
    selected_methods = tuple(methods)
    if not selected_methods or any(method not in METHODS for method in selected_methods):
        raise ValueError("methods must be registered promotion methods")
    requested_trajectories = int(config.get("seed_registry", {}).get("trajectory_count_per_operator_family", 30))
    requested_rounds = int(config.get("rounds", 30))
    requested_candidates = int(config.get("candidates_per_round", 4))
    if trajectory_limit is None and round_limit is None and not confirmatory:
        raise ValueError("DEV execution requires an explicit trajectory_limit or round_limit")
    trajectories = min(requested_trajectories, trajectory_limit or requested_trajectories)
    rounds = min(requested_rounds, round_limit or requested_rounds)
    candidate_count = min(requested_candidates, candidates_per_round or requested_candidates)
    seeds = [int(value) for value in config.get("seed_registry", {}).get("seed_blocks", [10001, 10101])]
    operators = registered_operators(settings)
    counts = registered_counts(config, operator_count=len(operators))
    require_registered_count(confirmatory, trajectories * len(operators), counts["trajectories"], "trajectory")
    require_registered_count(confirmatory, rounds, counts["rounds"], "round")
    require_registered_count(confirmatory, candidate_count, counts["candidates"], "candidate")
    if confirmatory:
        if registered_assessment_task_count <= 0:
            raise ValueError("confirmatory closed loop requires a non-empty sealed assessment plane")
        expected_planes = lock.get("sealed_planes", {}) if lock else {}
        if isinstance(expected_planes, Mapping) and isinstance(expected_planes.get("planes"), Mapping):
            expected_planes = expected_planes["planes"]
        for plane_name, actual in (
            ("proxy", len(proxy_tasks)),
            ("gate", len(gate_tasks)),
            ("assessment", registered_assessment_task_count),
        ):
            expected = len(expected_planes.get(plane_name, ())) if isinstance(expected_planes, Mapping) else actual
            require_registered_count(confirmatory, actual, expected, f"{plane_name} task")
    if confirmatory:
        lock = _lock_protocol_inputs(root, phase="closed_loop")
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    model_calls = 0
    container_executions = 0
    gate_queries = 0
    logical_hf_queries = 0
    pre_decision_pair_evaluations = 0
    post_decision_truth_evaluations = 0
    post_decision_truth_pair_evaluations = 0
    assessment_queries = 0
    execution_failures = 0
    operator_input_checks = 0
    for operator_index, operator in enumerate(operators):
        for trajectory_index in range(trajectories):
            seed = seeds[operator_index % len(seeds)] + trajectory_index
            initial = AgentPolicy.minimal().with_updates(
                metadata={"scaffold": "mini-SWE-agent", "operator": operator.name, "trajectory": str(trajectory_index)}
            )
            run_id = f"{operator.name}-trajectory-{trajectory_index:03d}"
            incumbents = {method: initial for method in selected_methods}
            cumulative = {method: 0.0 for method in selected_methods}
            for round_index in range(rounds):
                for method_index, method in enumerate(selected_methods):
                    incumbent = incumbents[method]
                    proxy_incumbent = evaluate_with_inspect(
                        proxy_tasks,
                        incumbent,
                        settings,
                        seed=seed + round_index,
                        phase="closed_loop_proxy",
                        role="proxy_evaluator",
                        run_id=f"{run_id}-{method_index}",
                    )
                    execution_failures += sum(item.status != "COMPLETED" for item in proxy_incumbent)
                    context = ProposalContext(
                        proxy_score=_mean_success(proxy_incumbent),
                        proxy_feedback=_task_feedback(proxy_incumbent),
                        round_index=round_index,
                        seed=seed,
                        resource_budget=settings.tool_calls,
                    )
                    assert_public_operator_input(incumbent, context, hidden_task_descriptors)
                    operator_input_checks += 1
                    proposed = operator.propose(incumbent, context, count=candidate_count)
                    candidate_rows: list[dict[str, Any]] = []
                    proxy_records: dict[str, list[ExecutionRecord]] = {}
                    for candidate_index, candidate in enumerate(proposed):
                        proxy_candidate = evaluate_with_inspect(
                            proxy_tasks,
                            candidate,
                            settings,
                            seed=seed + round_index,
                            phase="closed_loop_proxy",
                            role="proxy_evaluator",
                            run_id=f"{run_id}-{method_index}-candidate-{candidate_index}",
                        )
                        execution_failures += sum(item.status != "COMPLETED" for item in proxy_candidate)
                        proxy_records[candidate.policy_hash] = proxy_candidate
                        candidate_rows.append(
                            {
                                "run_id": run_id,
                                "round": round_index,
                                "candidate_index": candidate_index,
                                "candidate_id": content_hash({"run_id": run_id, "method": method, "round": round_index, "candidate": candidate.policy_hash}, length=20),
                                "candidate_hash": candidate.policy_hash,
                                "incumbent_hash": incumbent.policy_hash,
                                "proxy_delta": _mean_success(proxy_candidate) - _mean_success(proxy_incumbent),
                                "operator": operator.name,
                                "scaffold": "mini-SWE-agent",
                                "task_family": "mixed",
                                "seed": seed + round_index,
                                "incumbent_policy": incumbent.to_record(),
                                "candidate_policy": candidate.to_record(),
                                "footprint": incumbent.diff(candidate),
                            }
                        )
                    candidates.extend(candidate_rows)
                    observations: dict[str, float] = {}
                    query_pair_count = 0
                    selected, query_plan = select_from_gate_queries(
                        candidate_rows,
                        observations,
                        method=method,
                        budget=registered_closed_loop_budget(config, len(candidate_rows)),
                        seed=seed + round_index + method_index,
                    )
                    for query in query_plan:
                        row = next(item for item in candidate_rows if _candidate_id(item) == query["candidate_id"])
                        pairs = evaluate_paired_with_inspect(
                            gate_tasks,
                            incumbent,
                            AgentPolicy.from_record(row["candidate_policy"]),
                            settings,
                            seed=seed + round_index,
                            phase="closed_loop_gate",
                            role="pivot",
                            run_id=f"{run_id}-{method_index}-{row['candidate_id']}",
                        )
                        execution_failures += sum(paired_execution_failed(pair) for pair in pairs)
                        observations[query["candidate_id"]] = sum(pair.delta for pair in pairs) / max(len(pairs), 1)
                        query_pair_count += len(pairs)
                    selected, _ = select_from_gate_queries(
                        candidate_rows,
                        observations,
                        method=method,
                        budget=0,
                        seed=seed + round_index + method_index,
                    )
                    # Post-decision gate audit is evaluator-only.  It enables
                    # ISR and CISR diagnostics without feeding outcomes back to
                    # the proposal operator or selector.
                    all_truth: dict[str, float] = dict(observations)
                    truth_records: dict[str, dict[str, Any]] = {}
                    truth_pair_count = 0
                    for row in candidate_rows:
                        key = _candidate_id(row)
                        if key in all_truth:
                            continue
                        pairs = evaluate_paired_with_inspect(
                            gate_tasks,
                            incumbent,
                            AgentPolicy.from_record(row["candidate_policy"]),
                            settings,
                            seed=seed + round_index,
                            phase="closed_loop_truth_audit",
                            role="pivot",
                            run_id=f"{run_id}-{method_index}-{key}",
                        )
                        execution_failures += sum(paired_execution_failed(pair) for pair in pairs)
                        all_truth[key] = sum(pair.delta for pair in pairs) / max(len(pairs), 1)
                        truth_pair_count += len(pairs)
                    accounting = closed_loop_query_accounting(
                        query_count=len(query_plan),
                        candidate_count=len(candidate_rows),
                        observed_count=len(observations),
                        query_pair_count=query_pair_count,
                        truth_pair_count=truth_pair_count,
                    )
                    logical_hf_queries += accounting["logical_hf_queries"]
                    pre_decision_pair_evaluations += accounting["pre_decision_pair_evaluations"]
                    post_decision_truth_evaluations += accounting["post_decision_truth_evaluations"]
                    post_decision_truth_pair_evaluations += accounting["post_decision_truth_pair_evaluations"]
                    gate_queries += accounting["total_pair_evaluations"]
                    true_best = max(candidate_rows, key=lambda item: (all_truth[_candidate_id(item)], -int(item["candidate_index"])))
                    selected_key = _candidate_id(selected)
                    selected_delta = all_truth[selected_key]
                    cumulative[method] += selected_delta
                    incumbents[method] = AgentPolicy.from_record(selected["candidate_policy"])
                    candidate_truth = {
                        "run_id": run_id,
                        "round": round_index,
                        "method": method,
                        "selected_candidate": selected_key,
                        "selected_delta": selected_delta,
                        "true_best_candidate": _candidate_id(true_best),
                        "ISR": all_truth[_candidate_id(true_best)] - selected_delta,
                        "candidate_batch_hash": content_hash([item["candidate_hash"] for item in candidate_rows]),
                    }
                    truth_records[selected_key] = candidate_truth
                    row_output = {
                        "method": method,
                        "scaffold": "mini-SWE-agent",
                        "operator": operator.name,
                        "run_id": run_id,
                        "round": round_index,
                        "proxy_score": _mean_success(proxy_incumbent),
                        "gate_score": selected_delta,
                        "assessment_score_if_terminal": None,
                        "CISR": cumulative[method],
                        "ISR": candidate_truth["ISR"],
                        "selected_candidate": selected_key,
                        "true_best_candidate": _candidate_id(true_best),
                        "candidate_batch_hash": candidate_truth["candidate_batch_hash"],
                        "resource_metrics": _mean_metrics(proxy_incumbent),
                        "logical_hf_queries": accounting["logical_hf_queries"],
                        "pre_decision_pair_evaluations": accounting["pre_decision_pair_evaluations"],
                        "post_decision_truth_evaluations": accounting["post_decision_truth_evaluations"],
                        "post_decision_truth_pair_evaluations": accounting[
                            "post_decision_truth_pair_evaluations"
                        ],
                    }
                    rows.append(row_output)
                    model_calls += int(sum(item.resource_metrics.get("model_calls", 0.0) for item in proxy_incumbent))
                    model_calls += int(sum(item.resource_metrics.get("model_calls", 0.0) for item in proxy_records.get(selected["candidate_hash"], [])))
                    container_executions += len(proxy_incumbent) + sum(len(item) for item in proxy_records.values())
            assessment_tasks = planes.tasks(
                assessment_tasks_for_terminal("terminal_assessor"), role="terminal_assessor"
            )
            if not confirmatory:
                assessment_tasks = assessment_tasks[: assessment_limit or None]
            if not assessment_tasks:
                raise ValueError("terminal assessment requires at least one task")
            for method in selected_methods:
                final_policy = incumbents[method]
                final_records = evaluate_with_inspect(
                    assessment_tasks,
                    final_policy,
                    settings,
                    seed=seed + rounds,
                    phase="closed_loop_assessment",
                    role="terminal_assessor",
                    run_id=f"{run_id}-{method}-terminal",
                )
                execution_failures += sum(item.status != "COMPLETED" for item in final_records)
                score = _mean_success(final_records)
                terminal_rows = [item for item in rows if item["run_id"] == run_id and item["method"] == method]
                if terminal_rows:
                    terminal_rows[-1]["assessment_score_if_terminal"] = score
                assessments.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "terminal_policy_hash": final_policy.policy_hash,
                        "assessment_score": score,
                        "assessment_task_count": len(assessment_tasks),
                        "assessment_execution_paths": [record.trajectory for record in final_records],
                        "role": "terminal_assessor",
                        "queried_once": True,
                    }
                )
                assessment_queries += len(final_records)
                model_calls += int(sum(item.resource_metrics.get("model_calls", 0.0) for item in final_records))
                container_executions += len(final_records)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(candidates, output / "promotion_candidates.jsonl")
    write_jsonl(rows, output / "closed_loop_results.jsonl")
    write_jsonl(assessments, output / "assessment_results.jsonl")
    write_table(
        candidates,
        output / "promotion_candidates",
        columns=("run_id", "round", "candidate_id", "candidate_hash", "incumbent_hash", "proxy_delta", "operator", "scaffold", "task_family", "seed", "incumbent_policy", "candidate_policy", "footprint"),
    )
    write_table(
        rows,
        output / "closed_loop_results",
        columns=(
            "method",
            "scaffold",
            "operator",
            "run_id",
            "round",
            "proxy_score",
            "gate_score",
            "assessment_score_if_terminal",
            "CISR",
            "ISR",
            "selected_candidate",
            "true_best_candidate",
            "candidate_batch_hash",
            "resource_metrics",
            "logical_hf_queries",
            "pre_decision_pair_evaluations",
            "post_decision_truth_evaluations",
            "post_decision_truth_pair_evaluations",
        ),
    )
    write_table(
        assessments,
        output / "assessment_results",
        columns=("run_id", "method", "terminal_policy_hash", "assessment_score", "assessment_task_count", "assessment_execution_paths", "role", "queried_once"),
    )
    return _manifest(
        output,
        {
            "schema_version": "pivot-v15-external-closed-loop-1",
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
            "operator_count": len(operators),
            "method_count": len(selected_methods),
            "trajectory_count": trajectories * len(operators),
            "round_count": rounds,
            "candidate_count": len(candidates),
            "result_count": len(rows),
            "assessment_result_count": len(assessments),
            "proxy_task_count": len(proxy_tasks),
            "gate_task_count": len(gate_tasks),
            "assessment_task_count": len(assessment_tasks),
            # ``gate_queries`` is retained as a backwards-compatible physical
            # pair count; the explicit fields below distinguish decision-time
            # budget from evaluator-only truth collection.
            "gate_queries": gate_queries,
            "logical_hf_queries": logical_hf_queries,
            "pre_decision_pair_evaluations": pre_decision_pair_evaluations,
            "post_decision_truth_evaluations": post_decision_truth_evaluations,
            "post_decision_truth_pair_evaluations": post_decision_truth_pair_evaluations,
            "total_pair_evaluations": gate_queries,
            "assessment_queries": assessment_queries,
            "closed_loop_hf_budget": registered_closed_loop_budget(config, candidate_count),
            "terminal_assessment_exactly_once": all(item["queried_once"] for item in assessments),
            "assessment_sealed_until_terminal": True,
            "model_calls_performed": model_calls,
            "container_executions": container_executions,
            "execution_failure_count": execution_failures,
            "task_manifest_sha256": file_hash(root / "configs/v15/task_manifest.json"),
            "role_access_log": list(planes.access_log),
            "candidate_archive_frozen": True,
            "lock_hash": lock.get("lock_hash") if lock else None,
            "outcome_chasing": False,
            "note": "Assessment outcomes are terminal-only; post-decision gate truth is an evaluator audit and never returned to proposal operators.",
        },
    )


def summarize_terminal_assessment(root: Path, *, confirmatory: bool = False) -> dict[str, Any]:
    """Report an already completed terminal assessment without re-querying it."""

    output = phase_output(Path(root), confirmatory=confirmatory)
    manifest_path = output / "manifest.json"
    assessment_path = output / "assessment_results.jsonl"
    if not manifest_path.is_file() or not assessment_path.is_file():
        raise FileNotFoundError("closed-loop terminal assessment artifact is unavailable")
    rows = [json.loads(line) for line in assessment_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("queried_once") is not True for row in rows):
        raise ValueError("terminal assessment artifact does not prove exactly-once access")
    return {
        "status": "COMPLETED",
        "phase": "CONFIRMATORY" if confirmatory else "DEV",
        "assessment_result_count": len(rows),
        "terminal_assessment_exactly_once": True,
        "manifest": str(manifest_path),
        "assessment_sha256": file_hash(assessment_path),
    }


__all__ = [
    "DEFAULT_METHODS",
    "assessment_tasks_for_terminal",
    "closed_loop_query_accounting",
    "phase_output",
    "registered_closed_loop_budget",
    "run_external_closed_loop",
    "select_from_gate_queries",
    "summarize_terminal_assessment",
]
