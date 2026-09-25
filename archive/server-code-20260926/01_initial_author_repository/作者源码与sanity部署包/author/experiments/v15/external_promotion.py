"""External paired promotion replay for the modern-agent study.

Candidate generation and hidden evaluation are deliberately separate.  The
archive contains policies and proxy diagnostics only; paired gate outcomes are
looked up by this module at query time and are never passed to proposal
operators.  The same candidate batches are replayed for every selector so the
comparison is a decision comparison rather than a candidate-generation one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .configuration import ensure_lock, is_confirmatory_open, open_confirmatory_lock
from .confirmatory_guards import (
    registered_counts,
    reject_confirmatory_overrides,
    reject_existing_confirmatory_output,
    require_registered_budgets,
    require_registered_count,
)
from .external_runtime import (
    RuntimeSettings,
    evaluate_paired_with_inspect,
    paired_execution_failed,
    resolve_runtime_settings,
)
from .planes import load_task_planes
from .promotion import (
    METHODS,
    _candidate_id,
    _initial_posterior,
    _query_order,
    _select_max,
    expected_evsi,
)
from .protocol import AgentPolicy, content_hash, file_hash, write_jsonl, write_table

_HIDDEN_FIELDS = frozenset(
    {
        "delta_actor",
        "delta_strategic",
        "actor_reversal",
        "strategic_reversal",
        "true_delta",
        "deployment_score",
        "assessment_score",
    }
)


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Parse JSON/CSV booleans without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on", "y"}:
        return True
    if text in {"false", "0", "no", "off", "n", ""}:
        return False
    return default


def phase_output(root: Path, *, confirmatory: bool) -> Path:
    """Return the phase-specific output directory."""

    return Path(root).resolve() / "results/v15" / (
        "external-promotion" if confirmatory else "dev-external-promotion"
    )


def prepare_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy a proxy-only candidate archive.

    Policy content is required so a promotion replay can reconstruct fresh
    paired sandboxes.  Hidden deployment fields are rejected rather than
    silently ignored, preventing leakage from an earlier audit table.
    """

    if not rows:
        raise ValueError("candidate archive must not be empty")
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        leaked = sorted(_HIDDEN_FIELDS.intersection(row))
        if leaked:
            raise ValueError(f"hidden outcome fields are not allowed in candidate archive: {leaked}")
        candidate_raw = row.get("candidate_policy")
        incumbent_raw = row.get("incumbent_policy")
        if not isinstance(candidate_raw, Mapping) or not isinstance(incumbent_raw, Mapping):
            raise TypeError("candidate archive requires incumbent_policy and candidate_policy")
        candidate = AgentPolicy.from_record(candidate_raw)
        incumbent = AgentPolicy.from_record(incumbent_raw)
        candidate_id = str(row.get("candidate_id") or row.get("candidate_hash") or "")
        if not candidate_id:
            raise ValueError("candidate archive row requires candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if str(row.get("candidate_hash", candidate.policy_hash)) != candidate.policy_hash:
            raise ValueError(f"candidate policy hash mismatch: {candidate_id}")
        if str(row.get("incumbent_hash", incumbent.policy_hash)) != incumbent.policy_hash:
            raise ValueError(f"incumbent policy hash mismatch: {candidate_id}")
        if candidate.policy_hash == incumbent.policy_hash:
            raise ValueError(f"candidate equals incumbent: {candidate_id}")
        proxy = row.get("proxy_delta", row.get("delta_proxy"))
        if proxy is None:
            raise ValueError(f"candidate archive row requires proxy_delta: {candidate_id}")
        row["candidate_id"] = candidate_id
        row["candidate_hash"] = candidate.policy_hash
        row["incumbent_hash"] = incumbent.policy_hash
        row["candidate_policy"] = candidate.to_record()
        row["incumbent_policy"] = incumbent.to_record()
        row["proxy_delta"] = float(proxy)
        row["candidate_index"] = int(row.get("candidate_index", 0))
        row["round"] = int(row.get("round", row.get("round_index", 0)))
        row["seed"] = int(row.get("seed", 0))
        footprint = row.get("footprint", {})
        row["footprint"] = dict(footprint) if isinstance(footprint, Mapping) else {}
        prepared.append(row)
    return prepared


def load_candidate_archive(path: Path) -> list[dict[str, Any]]:
    """Read and validate JSONL candidate rows without opening hidden planes."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("candidate archive rows must be mappings")
    return prepare_candidate_rows(rows)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('run_id')}::{int(row.get('round', 0))}"].append(dict(row))
    return [
        sorted(group, key=lambda item: (int(item.get("candidate_index", 0)), _candidate_id(item)))
        for _, group in sorted(groups.items())
    ]


def build_query_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int = 101,
) -> list[dict[str, Any]]:
    """Build a deterministic registered HF query plan for one candidate batch."""

    if method not in METHODS:
        raise ValueError(f"unknown promotion method: {method}")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    clean = prepare_candidate_rows(rows)
    if len({str(row.get("run_id")) + "::" + str(row.get("round", 0)) for row in clean}) != 1:
        raise ValueError("build_query_plan expects one candidate batch")
    posterior = _initial_posterior(clean)
    rng = random.Random(seed)
    selected = _query_order(method, clean, posterior, int(budget), rng)
    return [
        {
            "method": method,
            "candidate_id": _candidate_id(row),
            "candidate_hash": str(row["candidate_hash"]),
            "hf_cost": 1.0,
            "query_index": index,
        }
        for index, row in enumerate(selected)
    ]


def _lock_protocol_inputs(root: Path, *, phase: str | None = None) -> dict[str, Any]:
    """Validate the immutable lock and optionally record phase opening.

    Validation is side-effect free when ``phase`` is omitted.  A confirmatory
    runner passes its phase only after all cheap preflight checks have passed;
    the resulting structured opening event is then hashed into the lock and
    returned to the caller.
    """

    lock_path = root / "experiments/v15/confirmatory_lock.json"
    ensure_lock(root, lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("confirmatory lock must contain a JSON object")
    status = payload.get("confirmatory_execution")
    if is_confirmatory_open(status) and (
        not isinstance(status, Mapping) or str(status.get("status", "")).upper() != "OPENED"
    ):
        raise PermissionError("confirmatory lock has a terminal execution status")
    expected = payload.get("protocol_input_hashes", {})
    current = {
        "configs/v15/confirmatory.yaml": file_hash(root / "configs/v15/confirmatory.yaml"),
        "configs/v15/task_manifest.json": file_hash(root / "configs/v15/task_manifest.json"),
        "configs/v15/external_versions.json": file_hash(root / "configs/v15/external_versions.json"),
    }
    if expected != current:
        raise ValueError("confirmatory lock input hashes do not match current protocol files")
    if phase is not None and not is_confirmatory_open(status):
        open_confirmatory_lock(root, lock_path, phase=phase)
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("opened confirmatory lock must contain a JSON object")
    return payload


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


def _query_ledger_record(
    *,
    phase: str,
    method: str,
    candidate_row: Mapping[str, Any],
    query_index: int,
    paired_delta: float,
    posterior_before: float,
    posterior_after: float,
    candidate_batch_hash: str,
    cache_hit: bool,
    evsi: float | None = None,
) -> dict[str, Any]:
    """Build an auditable logical/physical query row.

    A physical paired rollout may be reused by selectors during a replay, but
    every selector still consumes its registered logical query budget.
    """

    candidate_id = _candidate_id(candidate_row)
    return {
        "phase": phase,
        "method": method,
        "run_id": str(candidate_row.get("run_id")),
        "round": int(candidate_row.get("round", candidate_row.get("round_index", 0))),
        "candidate_id": candidate_id,
        "candidate_hash": str(candidate_row.get("candidate_hash", candidate_id)),
        "query_index": int(query_index),
        "task_id": f"{candidate_row.get('run_id')}::{candidate_row.get('round', 0)}::{candidate_id}",
        "paired_delta": float(paired_delta),
        "cost": 1.0,
        "posterior_before": float(posterior_before),
        "posterior_after": float(posterior_after),
        "EVSI": None if evsi is None else float(evsi),
        "observed_information_gain": abs(float(paired_delta) - float(posterior_before)),
        "candidate_batch_hash": candidate_batch_hash,
        "logical_hf_query": True,
        "physical_pair_evaluation": not cache_hit,
        "cache_hit": bool(cache_hit),
        "outcome_chasing": False,
    }


def run_external_promotion(
    root: Path,
    *,
    confirmatory: bool = False,
    candidate_archive: Path | None = None,
    task_limit: int | None = None,
    agent_steps: int | None = None,
    budgets: Sequence[int] = (1, 2, 4),
    verify_image: bool = False,
) -> dict[str, Any]:
    """Run registered promotion methods against an immutable candidate archive."""

    root = Path(root).resolve()
    if confirmatory and os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
        raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
    reject_confirmatory_overrides(
        confirmatory,
        candidate_archive=candidate_archive,
        task_limit=task_limit,
        agent_steps=agent_steps,
    )
    if confirmatory and not verify_image:
        raise ValueError("confirmatory execution requires sandbox image verification")
    lock = _lock_protocol_inputs(root) if confirmatory else None
    output = phase_output(root, confirmatory=confirmatory)
    reject_existing_confirmatory_output(output, confirmatory)
    source_archive = root / "results/v15" / (
        "external-transition-audit" if confirmatory else "dev-external-transition-audit"
    ) / "promotion_candidates.jsonl"
    frozen_archive_root = root / "results/v15" / (
        "external-candidate-archive" if confirmatory else "dev-external-candidate-archive"
    )
    if candidate_archive is None:
        # Phase 1 owns generation; freeze its exact rows before any promotion
        # method can inspect them.  This is metadata/content handling only and
        # does not open a hidden task plane.
        from .evidence import freeze_candidate_archive

        if not (frozen_archive_root / "manifest.json").is_file():
            freeze_candidate_archive(
                source_archive,
                frozen_archive_root,
                phase="CONFIRMATORY" if confirmatory else "DEV",
                confirmatory=confirmatory,
            )
        archive = frozen_archive_root / "promotion_candidates.jsonl"
    else:
        archive = Path(candidate_archive).resolve()
    if confirmatory:
        primary_manifest_path = root / "results/v15/external-transition-audit/manifest.json"
        if not primary_manifest_path.is_file():
            raise ValueError("confirmatory promotion requires a completed primary transition archive")
        primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(primary_manifest, Mapping)
            or primary_manifest.get("phase") != "CONFIRMATORY"
            or primary_manifest.get("status") != "COMPLETED"
        ):
            raise ValueError("confirmatory promotion requires a completed CONFIRMATORY transition archive")
        operators_count = int(primary_manifest.get("operator_count", 2))
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise TypeError("confirmatory config must be a mapping")
        counts = registered_counts(config, operator_count=operators_count)
        require_registered_count(confirmatory, int(primary_manifest.get("trajectory_count", 0)), counts["trajectories"], "trajectory")
        require_registered_count(confirmatory, int(primary_manifest.get("round_count", 0)), counts["rounds"], "round")
        expected_candidates = counts["trajectories"] * counts["rounds"] * counts["candidates"]
        require_registered_count(confirmatory, int(primary_manifest.get("candidate_count", 0)), expected_candidates, "candidate")
        archive_manifest_path = archive.parent / "manifest.json"
        if not archive_manifest_path.is_file():
            raise ValueError("confirmatory promotion requires an immutable candidate-archive manifest")
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(archive_manifest, Mapping):
            raise ValueError("confirmatory candidate-archive manifest is invalid")
        if (
            archive_manifest.get("confirmatory") is not True
            or archive_manifest.get("immutable") is not True
            or archive_manifest.get("regeneration_allowed") is not False
            or archive_manifest.get("archive_sha256") != file_hash(archive)
        ):
            raise ValueError("confirmatory candidate archive failed immutability/hash validation")
        if not archive.is_file():
            raise FileNotFoundError(f"confirmatory candidate archive is unavailable: {archive}")
        if int(primary_manifest.get("transition_count", 0)) != expected_candidates:
            raise ValueError("confirmatory transition archive has an incomplete transition count")
    rows = load_candidate_archive(archive)
    planes = load_task_planes(root / "configs/v15/task_manifest.json")
    gate_tasks = planes.tasks("gate", role="promotion")
    if not gate_tasks:
        raise ValueError("promotion requires at least one sealed gate task")
    if confirmatory:
        primary_manifest = json.loads((root / "results/v15/external-transition-audit/manifest.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise TypeError("confirmatory config must be a mapping")
        counts = registered_counts(config, operator_count=int(primary_manifest.get("operator_count", 2)))
        expected_candidates = counts["trajectories"] * counts["rounds"] * counts["candidates"]
        expected_batches = counts["trajectories"] * counts["rounds"]
        require_registered_count(confirmatory, len(rows), expected_candidates, "candidate archive")
        require_registered_count(confirmatory, len(_group_rows(rows)), expected_batches, "candidate batch")
        require_registered_count(confirmatory, len(gate_tasks), int(primary_manifest.get("gate_task_count", 0)), "gate task")
    if confirmatory:
        lock = _lock_protocol_inputs(root, phase="promotion_replay")
    clean_budgets = tuple(sorted({int(value) for value in budgets}))
    if not clean_budgets or any(value < 0 for value in clean_budgets):
        raise ValueError("budgets must contain non-negative integers")
    registered_budgets = tuple(lock.get("hf_budgets", (1, 2, 4))) if lock else clean_budgets
    require_registered_budgets(confirmatory, clean_budgets, registered_budgets)
    settings = _settings(root, output, agent_steps=agent_steps, verify_image=verify_image)
    truth_records: list[dict[str, Any]] = []
    pair_cache: dict[tuple[str, str, str, str, int], float] = {}
    physical_pairs = 0
    evaluation_failures = 0
    logical_hf_queries = 0
    post_decision_truth_evaluations = 0

    def observe(row: Mapping[str, Any]) -> tuple[float, bool]:
        nonlocal evaluation_failures, physical_pairs
        key = (
            str(row.get("run_id")),
            str(row.get("round", 0)),
            _candidate_id(row),
            str(row["incumbent_hash"]),
            int(row.get("seed", 0)),
        )
        cache_hit = key in pair_cache
        if not cache_hit:
            incumbent = AgentPolicy.from_record(row["incumbent_policy"])
            candidate = AgentPolicy.from_record(row["candidate_policy"])
            pairs = evaluate_paired_with_inspect(
                gate_tasks,
                incumbent,
                candidate,
                settings,
                seed=int(row.get("seed", 0)),
                phase="promotion_gate",
                role="promotion",
                run_id=f"{row.get('run_id')}-round-{int(row.get('round', 0))}-{_candidate_id(row)}",
            )
            evaluation_failures += sum(paired_execution_failed(pair) for pair in pairs)
            if len(pairs) != len(gate_tasks):
                evaluation_failures += 1
            value = sum(pair.delta for pair in pairs) / max(len(pairs), 1)
            pair_cache[key] = float(value)
            truth_records.append(
                {
                    "candidate_id": _candidate_id(row),
                    "candidate_hash": str(row["candidate_hash"]),
                    "incumbent_hash": str(row["incumbent_hash"]),
                    "run_id": str(row.get("run_id")),
                    "round": int(row.get("round", 0)),
                    "task_count": len(gate_tasks),
                    "paired_delta": float(value),
                    "role": "promotion",
                    "pair_count": len(pairs),
                    "expected_pair_count": len(gate_tasks),
                    "complete": len(pairs) == len(gate_tasks)
                    and not any(paired_execution_failed(pair) for pair in pairs),
                }
            )
            physical_pairs += len(pairs)
        return pair_cache[key], cache_hit

    results: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for group_index, group in enumerate(_group_rows(rows)):
        batch_hash = content_hash([str(item["candidate_hash"]) for item in group])
        for budget in clean_budgets:
            for method_index, method in enumerate(METHODS):
                posterior = _initial_posterior(group)
                rng = random.Random(101 + group_index * 100_003 + budget * 1_009 + method_index)
                query_rows = _query_order(method, group, posterior, budget, rng)
                observed_values: dict[str, float] = {}
                for query_index, row in enumerate(query_rows):
                    logical_hf_queries += 1
                    before = posterior.value(row)
                    evsi = expected_evsi(
                        group,
                        posterior,
                        row,
                        seed=101 + group_index * 100_003 + budget * 1_009 + method_index * 97 + query_index,
                        fantasies=128,
                        posterior_samples=256,
                    )
                    observed, cache_hit = observe(row)
                    posterior.observe(row, observed)
                    observed_values[_candidate_id(row)] = observed
                    queries.append(
                        _query_ledger_record(
                            phase="CONFIRMATORY" if confirmatory else "DEV",
                            method=method,
                            candidate_row=row,
                            query_index=query_index,
                            paired_delta=observed,
                            posterior_before=before,
                            posterior_after=observed,
                            candidate_batch_hash=batch_hash,
                            cache_hit=cache_hit,
                            evsi=evsi,
                        )
                    )
                selected = _select_max(group, posterior)
                selected_key = _candidate_id(selected)
                # Truth is collected after the decision for evaluation only.
                truth: dict[str, float] = {}
                for row in group:
                    key = _candidate_id(row)
                    if key not in observed_values:
                        post_decision_truth_evaluations += 1
                    truth[key] = observe(row)[0]
                true_best = max(group, key=lambda item: (truth[_candidate_id(item)], -int(item.get("candidate_index", 0))))
                results.append(
                    {
                        "phase": "CONFIRMATORY" if confirmatory else "DEV",
                        "method": method,
                        "run_id": str(selected.get("run_id")),
                        "round": int(selected.get("round", 0)),
                        "hf_budget": int(budget),
                        "selected_candidate": selected_key,
                        "selected_candidate_hash": str(selected["candidate_hash"]),
                        "true_best_candidate": _candidate_id(true_best),
                        "true_best_candidate_hash": str(true_best["candidate_hash"]),
                        "ISR": truth[_candidate_id(true_best)] - truth[selected_key],
                        "hf_cost": float(len(query_rows)),
                        "candidate_count": len(group),
                        "candidate_batch_hash": batch_hash,
                        "outcome_chasing": False,
                    }
                )
    write_jsonl(truth_records, output / "truth_audit.jsonl")
    write_table(
        truth_records,
        output / "truth_audit",
        columns=(
            "candidate_id",
            "candidate_hash",
            "incumbent_hash",
            "run_id",
            "round",
            "task_count",
            "paired_delta",
            "pair_count",
            "expected_pair_count",
            "complete",
            "role",
        ),
    )
    from .promotion import write_promotion_artifacts

    phase = "CONFIRMATORY" if confirmatory else "DEV"
    artifact_manifest = write_promotion_artifacts(
        {"promotion_results": results, "hf_queries": queries},
        output,
        phase=phase,
        candidate_archive=archive,
        note="External paired gate evaluation. Truth audit is collected after each decision and is excluded from query budgets.",
    )
    payload = {
        **artifact_manifest,
        "status": "COMPLETED" if evaluation_failures == 0 else "IMPLEMENTATION_FAILURE",
        "terminal_state": (
            "IMPLEMENTATION_FAILURE"
            if evaluation_failures
            else "UNDERPOWERED"
            if not confirmatory
            else None
        ),
        "execution_attempted": True,
        "design_status": "VALIDATED_DEV" if not confirmatory and evaluation_failures == 0 else "PENDING_ANALYSIS",
        "leakage_detected": False,
        "candidate_count": len(rows),
        "candidate_batch_count": len(_group_rows(rows)),
        "candidate_archive_frozen": bool(
            (archive.parent / "manifest.json").is_file()
            and _as_bool(json.loads((archive.parent / "manifest.json").read_text(encoding="utf-8")).get("immutable"), default=False)
        ),
        "candidate_archive_path": str(archive.relative_to(root)) if archive.is_relative_to(root) else str(archive),
        "candidate_archive_sha256": file_hash(archive) if archive.is_file() else None,
        "candidate_archive_manifest_sha256": file_hash(archive.parent / "manifest.json")
        if (archive.parent / "manifest.json").is_file()
        else None,
        "gate_task_count": len(gate_tasks),
        "physical_pair_evaluations": physical_pairs,
        "logical_hf_queries": logical_hf_queries,
        "logical_query_rows": len(queries),
        "physical_query_rows": sum(_as_bool(row.get("physical_pair_evaluation"), default=False) for row in queries),
        "cache_hit_query_rows": sum(_as_bool(row.get("cache_hit"), default=False) for row in queries),
        "post_decision_truth_evaluations": post_decision_truth_evaluations,
        "execution_failure_count": evaluation_failures,
        "truth_audit_count": len(truth_records),
        "assessment_queries": 0,
        "role_access_log": list(planes.access_log),
        "runtime": settings.to_manifest(),
        "lock_hash": lock.get("lock_hash") if lock else None,
        "outcome_chasing": False,
    }
    return _manifest(output, payload)


__all__ = [
    "_query_ledger_record",
    "build_query_plan",
    "load_candidate_archive",
    "phase_output",
    "prepare_candidate_rows",
    "run_external_promotion",
]
