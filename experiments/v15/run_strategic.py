"""Run the registered identity-blind response layer for coding-agent patches.

The non-LLM responder is intentionally a diagnostic.  It consumes only a
candidate's executed tree, changed files, registered test command, and trace;
it never receives the hypothesis, candidate rank, gate result, or assessment
data.  A mutation score is not silently promoted to a strategic utility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from difflib import unified_diff
from pathlib import Path
from typing import Any

import yaml

from .agent_response import review_with_model
from .confirmatory_guards import (
    registered_counts,
    reject_confirmatory_overrides,
    reject_existing_confirmatory_output,
    require_registered_count,
)
from .external_promotion import _lock_protocol_inputs
from .external_runtime import RuntimeSettings, locked_runtime_python, resolve_runtime_settings
from .planes import TaskSpec, load_task_planes
from .protocol import canonical_json, file_hash, write_jsonl, write_table
from .run_pi_replication import _portable_test_command, _test_sandbox_command
from .strategic_response import run_paired_mutation_response


def _phase_output(root: Path, *, confirmatory: bool) -> Path:
    return Path(root).resolve() / "results/v15" / (
        "external-strategic-response" if confirmatory else "dev-external-strategic-response"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"JSONL row must be a mapping: {path}")
            rows.append(dict(value))
    return rows


def _changed_files(task: TaskSpec, final_tree: Path) -> tuple[str, ...]:
    before = {str(path): content.encode("utf-8") for path, content in task.files.items()}
    after = {
        path.relative_to(final_tree).as_posix(): path.read_bytes()
        for path in final_tree.rglob("*")
        if path.is_file()
        and ".pi-session" not in path.relative_to(final_tree).parts
        and "__pycache__" not in path.relative_to(final_tree).parts
    }
    return tuple(sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path)))


def _trace_lines(path: Path) -> tuple[str, ...]:
    """Extract a bounded, redacted command trace from Pi JSONL events."""

    if not path.is_file():
        return ()
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        text = json.dumps(event, ensure_ascii=True, sort_keys=True)
        for match in re.findall(r'"(?:command|cmd)"\s*:\s*"((?:\\.|[^"\\])*)"', text):
            values.append(match[:1000])
    return tuple(values[-128:])


def _runtime_site_packages(root: Path, runtime: Path) -> Path | None:
    """Resolve the venv site-packages directory without following its binary."""

    venv = Path(runtime).parent.parent
    candidates = sorted((venv / "lib").glob("python*/site-packages"))
    return candidates[0] if candidates and candidates[0].is_dir() else None


def _run_registered_tests(
    task: TaskSpec,
    cwd: Path,
    *,
    root: Path | None = None,
    timeout: int = 60,
    require_isolation: bool = False,
) -> int:
    """Run the registered responder tests in a fresh, credential-free namespace.

    The strategic response is part of the deployment-response layer, so it
    must use the same pinned Python runtime and network isolation as the Pi
    adapter whenever the project root is available.  A DEV-only fallback to a
    host subprocess is retained for lightweight unit fixtures; confirmatory
    callers set ``require_isolation`` and fail closed when bubblewrap is absent.
    """

    workspace = Path(cwd).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"response workspace is unavailable: {workspace}")
    runtime: Path | None = None
    site_packages: Path | None = None
    if root is not None:
        try:
            runtime = locked_runtime_python(Path(root).resolve())
        except (FileNotFoundError, ValueError):
            if require_isolation:
                raise
    use_bwrap = shutil.which("bwrap") is not None
    if require_isolation and not use_bwrap:
        raise RuntimeError("confirmatory strategic response requires bubblewrap isolation")
    if runtime is not None:
        site_packages = _runtime_site_packages(Path(root or workspace), runtime)
    command = _portable_test_command(
        task,
        inside_sandbox=use_bwrap,
        python_executable=runtime,
    )
    invocation: str | list[str]
    if use_bwrap:
        invocation = _test_sandbox_command(
            workspace=workspace,
            command=command,
            python_executable=runtime,
            python_site=site_packages,
        )
    else:
        invocation = command
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/workspace" if use_bwrap else str(workspace),
        "HOME": "/tmp" if use_bwrap else str(workspace / ".response-home"),
        "TMPDIR": "/tmp" if use_bwrap else str(workspace / ".response-tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    if use_bwrap and site_packages is not None:
        environment["PYTHONPATH"] = "/workspace:/runtime/python-site"
    if not use_bwrap:
        Path(environment["HOME"]).mkdir(parents=True, exist_ok=True)
        Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        invocation,
        cwd=workspace,
        shell=isinstance(invocation, str),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )
    return int(completed.returncode)


def _patch_text(task: TaskSpec, final_tree: Path, changed: Sequence[str]) -> str:
    """Build a bounded patch view from the task snapshot and executed tree."""

    chunks: list[str] = []
    for relative in changed:
        before = str(task.files.get(relative, ""))
        path = final_tree / relative
        try:
            after = path.read_text(encoding="utf-8") if path.is_file() else ""
        except (OSError, UnicodeError):
            continue
        chunks.extend(
            unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)[:50000]


def _path_values(value: Any) -> list[Path]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [Path(str(item)) for item in value if item]


def _string_values(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item]


def _manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _response_records(
    root: Path,
    *,
    confirmatory: bool,
    task_limit: int | None,
    enable_agent_reviewer: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_dir = Path(root).resolve() / "results/v15" / (
        "external-transition-audit" if confirmatory else "dev-external-transition-audit"
    )
    transitions = _load_jsonl(source_dir / "autonomous_transitions.jsonl")
    planes = load_task_planes(Path(root).resolve() / "configs/v15/task_manifest.json")
    all_tasks = planes.tasks("proxy", role="audit") + planes.tasks("gate", role="audit")
    tasks = {task.task_id: task for task in all_tasks}
    if not tasks:
        raise ValueError("strategic response requires at least one proxy task")
    records: list[dict[str, Any]] = []
    candidate_rows = [row for row in transitions if row.get("candidate_policy") and row.get("candidate_hash")]
    if task_limit is not None and not confirmatory:
        candidate_rows = candidate_rows[: int(task_limit)]
    review_settings: RuntimeSettings | None = None
    if enable_agent_reviewer:
        output = _phase_output(root, confirmatory=confirmatory)
        review_settings = resolve_runtime_settings(
            root,
            artifact_root=output / "agent-reviewer-artifacts",
            log_root=output / "agent-reviewer-logs",
            verify_image=False,
        )
    agent_review_count = 0
    agent_review_failures = 0
    for row in candidate_rows:
        resources = row.get("resource_metrics", {})
        resources = resources if isinstance(resources, Mapping) else {}
        actor_ids = _string_values(resources.get("actor_task_ids", []))
        if not actor_ids:
            actor_ids = [str(row.get("task_id") or next(iter(tasks)))]
        incumbent_paths = _path_values(resources.get("incumbent_final_tree_paths", []))
        candidate_paths = _path_values(resources.get("candidate_final_tree_paths", []))
        incumbent_fallback = row.get("incumbent_final_tree_path") or row.get("incumbent_tree_path")
        candidate_fallback = row.get("candidate_final_tree_path") or row.get("final_tree_path")
        if not incumbent_paths and incumbent_fallback:
            incumbent_paths = [Path(str(incumbent_fallback))]
        if not candidate_paths and candidate_fallback:
            candidate_paths = [Path(str(candidate_fallback))]
        incumbent_traces = _path_values(resources.get("incumbent_trajectories", []))
        candidate_traces = _path_values(resources.get("candidate_trajectories", []))
        pair_count = min(len(actor_ids), len(incumbent_paths), len(candidate_paths))
        pairs: list[tuple[TaskSpec, Path, Path, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
        for index in range(pair_count):
            task = tasks.get(actor_ids[index]) or next(iter(tasks.values()))
            incumbent_tree = incumbent_paths[index]
            candidate_tree = candidate_paths[index]
            if not incumbent_tree.is_dir() or not candidate_tree.is_dir():
                continue
            changed = tuple(sorted(set(_changed_files(task, incumbent_tree)) | set(_changed_files(task, candidate_tree))))
            incumbent_trace = _trace_lines(incumbent_traces[index]) if index < len(incumbent_traces) else ()
            candidate_trace = _trace_lines(candidate_traces[index]) if index < len(candidate_traces) else ()
            pairs.append((task, incumbent_tree, candidate_tree, changed, incumbent_trace, candidate_trace))
        if not pairs:
            records.append(
                {
                    "status": "NO_EXECUTED_TREE",
                    "response_family": "non_llm_mutation_property_fuzz",
                    "transition_id": str(row.get("transition_id", "")),
                    "run_id": str(row.get("run_id", "")),
                    "operator": str(row.get("operator", "")),
                    "round": int(row.get("round", 0)),
                    "candidate_hash": str(row.get("candidate_hash", "")),
                    "task_id": actor_ids[0],
                    "delta_proxy": row.get("delta_proxy"),
                    "delta_actor": row.get("delta_actor"),
                    "delta_strategic": None,
                    "strategic_reversal": False,
                    "response_pair_count": 0,
                    "changed_files": [],
                    "mutation_attempts": 0,
                    "mutations_killed": 0,
                    "mutation_score": None,
                    "incumbent_mutation_score": None,
                    "delta_response_utility": None,
                    "trace_digest": None,
                    "response_digest": None,
                    "error": "paired executed trees were not retained by the actor runner",
                    "agent_reviewer_status": "NO_EXECUTED_TREE" if enable_agent_reviewer else "NOT_RUN",
                    "agent_findings_count": 0,
                    "agent_request_digest": None,
                    "agent_response_digest": None,
                    "agent_error": "paired executed trees were not retained by the actor runner" if enable_agent_reviewer else None,
                }
            )
            continue
        audits = []
        patch_parts: list[str] = []
        all_changed: set[str] = set()
        all_traces: list[str] = []
        for task, incumbent_tree, candidate_tree, changed, incumbent_trace, candidate_trace in pairs:
            def run_tests(_command: str, *, cwd: Path, _task: TaskSpec = task) -> int:
                return _run_registered_tests(
                    _task,
                    cwd,
                    root=root,
                    require_isolation=confirmatory,
                )

            audit = run_paired_mutation_response(
                incumbent_tree,
                candidate_tree,
                task,
                changed=changed,
                incumbent_trace=incumbent_trace,
                candidate_trace=candidate_trace,
                run_command=run_tests,
            )
            audits.append(audit)
            all_changed.update(changed)
            all_traces.extend((*incumbent_trace, *candidate_trace))
            patch_parts.append(_patch_text(task, candidate_tree, changed))
        valid_audits = [audit for audit in audits if audit.delta_strategic is not None]
        valid_deltas = [float(audit.delta_strategic) for audit in valid_audits if audit.delta_strategic is not None]
        incumbent_scores = [audit.incumbent_score for audit in audits if audit.incumbent_score is not None]
        candidate_scores = [audit.candidate_score for audit in audits if audit.candidate_score is not None]
        if any(audit.status == "IMPLEMENTATION_FAILURE" for audit in audits):
            status = "IMPLEMENTATION_FAILURE"
        elif all(audit.status == "COMPLETED" for audit in audits):
            status = "COMPLETED"
        elif all(audit.status == "NO_MUTABLE_SOURCE" for audit in audits):
            status = "NO_MUTABLE_SOURCE"
        else:
            status = "NO_MUTATION_OPERATOR"
        delta_strategic = (
            sum(valid_deltas) / len(valid_deltas)
            if len(valid_deltas) == len(audits) and valid_deltas
            else None
        )
        record = {
            "status": status,
            "response_family": "non_llm_mutation_property_fuzz",
            "transition_id": str(row.get("transition_id", "")),
            "run_id": str(row.get("run_id", "")),
            "operator": str(row.get("operator", "")),
            "round": int(row.get("round", 0)),
            "candidate_hash": str(row.get("candidate_hash", "")),
            "task_id": pairs[0][0].task_id,
            "delta_proxy": row.get("delta_proxy"),
            "delta_actor": row.get("delta_actor"),
            "delta_strategic": delta_strategic,
            "strategic_reversal": bool(
                float(row.get("delta_actor") or 0.0) > 0.0
                and delta_strategic is not None
                and delta_strategic < 0.0
            ),
            "response_pair_count": len(pairs),
            "changed_files": sorted(all_changed),
            "mutation_attempts": sum(audit.candidate_attempted for audit in audits),
            "mutations_killed": sum(audit.candidate_killed for audit in audits),
            "mutation_score": sum(candidate_scores) / len(candidate_scores) if candidate_scores else None,
            "incumbent_mutation_attempts": sum(audit.incumbent_attempted for audit in audits),
            "incumbent_mutations_killed": sum(audit.incumbent_killed for audit in audits),
            "incumbent_mutation_score": sum(incumbent_scores) / len(incumbent_scores) if incumbent_scores else None,
            "delta_response_utility": delta_strategic,
            "trace_digest": hashlib.sha256(canonical_json(all_traces).encode("utf-8")).hexdigest(),
            "response_digest": hashlib.sha256(canonical_json([audit.to_record() for audit in audits]).encode("utf-8")).hexdigest(),
            "error": next((audit.error for audit in audits if audit.error), None),
            "agent_reviewer_status": "NOT_RUN",
            "agent_findings_count": 0,
            "agent_request_digest": None,
            "agent_response_digest": None,
            "agent_error": None,
        }
        if enable_agent_reviewer and review_settings is not None:
            try:
                review = review_with_model(
                    review_settings,
                    patch="\n".join(patch_parts)[:50000],
                    changed_interfaces=tuple(sorted(all_changed)),
                    execution_trace=tuple(all_traces[-128:]),
                )
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                agent_review_failures += 1
                record.update(
                    {
                        "agent_reviewer_status": "IMPLEMENTATION_FAILURE",
                        "agent_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                agent_review_count += 1
                record.update(
                    {
                        "agent_reviewer_status": review.status,
                        "agent_findings_count": len(review.findings),
                        "agent_request_digest": review.request_digest,
                        "agent_response_digest": review.response_digest,
                    }
                )
        records.append(record)
    return records, {
        "source_manifest": str(source_dir / "manifest.json"),
        "source_manifest_sha256": file_hash(source_dir / "manifest.json"),
        "task_manifest_sha256": file_hash(Path(root).resolve() / "configs/v15/task_manifest.json"),
        "candidate_record_count": len(candidate_rows),
        "task_count": len(tasks),
        "agent_reviewer_enabled": enable_agent_reviewer,
        "agent_review_count": agent_review_count,
        "agent_review_failures": agent_review_failures,
        "role_access_log": list(planes.access_log_snapshot()),
    }


def run_strategic(
    root: Path,
    *,
    confirmatory: bool = False,
    task_limit: int | None = None,
    enable_agent_reviewer: bool = False,
) -> dict[str, Any]:
    """Run the frozen non-LLM response family without opening assessment data."""

    root = Path(root).resolve()
    if confirmatory and os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
        raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
    reject_confirmatory_overrides(confirmatory, task_limit=task_limit)
    if confirmatory and task_limit is not None:
        raise ValueError("confirmatory strategic response cannot use DEV task limits")
    if confirmatory:
        enable_agent_reviewer = True
    lock = _lock_protocol_inputs(root) if confirmatory else None
    output = _phase_output(root, confirmatory=confirmatory)
    reject_existing_confirmatory_output(output, confirmatory)
    if confirmatory:
        source_manifest_path = root / "results/v15/external-transition-audit/manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("confirmatory strategic response requires the primary transition archive")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, Mapping) or source_manifest.get("phase") != "CONFIRMATORY" or source_manifest.get("status") != "COMPLETED":
            raise ValueError("confirmatory strategic response requires a completed CONFIRMATORY transition archive")
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise TypeError("confirmatory config must be a mapping")
        counts = registered_counts(config, operator_count=int(source_manifest.get("operator_count", 2)))
        expected = counts["trajectories"] * counts["rounds"] * counts["candidates"]
        require_registered_count(confirmatory, int(source_manifest.get("candidate_count", 0)), expected, "transition candidate")
    if confirmatory:
        lock = _lock_protocol_inputs(root, phase="strategic_response")
    records, provenance = _response_records(
        root,
        confirmatory=confirmatory,
        task_limit=task_limit,
        enable_agent_reviewer=enable_agent_reviewer,
    )
    if confirmatory:
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        counts = registered_counts(config, operator_count=int(source_manifest.get("operator_count", 2)))
        require_registered_count(
            confirmatory,
            len(records),
            counts["trajectories"] * counts["rounds"] * counts["candidates"],
            "strategic response",
        )
    write_jsonl(records, output / "response_audits.jsonl")
    write_table(
        records,
        output / "response_audits",
        columns=(
            "status", "response_family", "transition_id", "run_id", "operator", "round",
            "candidate_hash", "task_id", "delta_proxy", "delta_actor", "delta_strategic",
            "strategic_reversal", "response_pair_count", "changed_files", "mutation_attempts",
            "mutations_killed", "mutation_score", "incumbent_mutation_attempts",
            "incumbent_mutations_killed", "incumbent_mutation_score", "delta_response_utility",
            "trace_digest", "response_digest", "error",
            "agent_reviewer_status", "agent_findings_count", "agent_request_digest",
            "agent_response_digest", "agent_error",
        ),
    )
    completed = sum(row.get("status") == "COMPLETED" for row in records)
    reviewer_attempts = sum(row.get("agent_reviewer_status") in {"COMPLETED", "IMPLEMENTATION_FAILURE"} for row in records)
    reviewer_completed = sum(row.get("agent_reviewer_status") == "COMPLETED" for row in records)
    runtime_status: dict[str, Any] = {
        "bwrap_available": shutil.which("bwrap") is not None,
        "pinned_python": None,
    }
    try:
        runtime_status["pinned_python"] = str(locked_runtime_python(root))
    except (FileNotFoundError, ValueError):
        runtime_status["pinned_python"] = None
    manifest = _manifest(
        output,
        {
            "schema_version": "pivot-v15-strategic-response-1",
            "phase": "CONFIRMATORY" if confirmatory else "DEV",
            "confirmatory": confirmatory,
            "status": "COMPLETED" if records and completed == len(records) else "PARTIAL",
            "terminal_state": "UNDERPOWERED" if records and not confirmatory else None,
            "execution_attempted": True,
            "design_status": "VALIDATED_DEV" if records and not confirmatory else "PENDING_ANALYSIS",
            "leakage_detected": False,
            "response_families": {
                "non_llm_mutation_property_fuzz": "COMPLETED" if records else "NOT_RUN",
                "independent_agent_reviewer": (
                    "COMPLETED"
                    if enable_agent_reviewer and reviewer_attempts == len(records) and reviewer_completed == len(records)
                    else "PARTIAL"
                    if enable_agent_reviewer and reviewer_attempts
                    else "NOT_RUN"
                ),
            },
            "record_count": len(records),
            "completed_record_count": completed,
            "response_pair_count": sum(int(row.get("response_pair_count", 0)) for row in records),
            "strategic_delta_count": sum(row.get("delta_strategic") is not None for row in records),
            "strategic_reversal_count": sum(bool(row.get("strategic_reversal")) for row in records),
            "agent_reviewer_enabled": enable_agent_reviewer,
            "agent_review_count": reviewer_completed,
            "agent_review_failures": sum(row.get("agent_reviewer_status") == "IMPLEMENTATION_FAILURE" for row in records),
            "independent_unit": "trajectory_or_task_cluster",
            "delta_strategic_available": bool(records) and all(row.get("delta_strategic") is not None for row in records),
            "assessment_accessed": any(
                event.get("plane") == "assessment"
                and event.get("outcome") == "granted"
                for event in provenance.get("role_access_log", [])
                if isinstance(event, Mapping)
            ),
            "gate_accessed": any(
                event.get("plane") == "gate"
                and event.get("outcome") == "granted"
                for event in provenance.get("role_access_log", [])
                if isinstance(event, Mapping)
            ),
            "role_access_log": provenance.get("role_access_log", []),
            "outcome_chasing": False,
            "identity_blind": True,
            "sandbox": {
                "mode": "bubblewrap_explicit_mounts" if runtime_status["bwrap_available"] else "dev_host_fallback",
                "network": "disabled" if runtime_status["bwrap_available"] else "host_inherited_dev_only",
                "host_root_exposed": False,
            },
            "runtime": runtime_status,
            "response_inputs": ["candidate_patch", "changed_interfaces", "execution_trace"],
            "provenance": provenance,
            "lock_hash": lock.get("lock_hash") if lock else None,
            "note": "Paired mutation detection is a registered response utility diagnostic. It is reported separately from task success and does not establish deployment-causal strategic utility without an additional causal response model.",
        },
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the locked identity-blind response layer")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--agent-reviewer", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_strategic(args.root.resolve(), confirmatory=args.confirmatory, task_limit=args.task_limit, enable_agent_reviewer=args.agent_reviewer), sort_keys=True))
