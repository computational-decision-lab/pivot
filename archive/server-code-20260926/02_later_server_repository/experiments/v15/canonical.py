"""Canonical V15 result-table schemas and provenance helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import file_hash, write_table

FOOTPRINT_COLUMNS = (
    "prompt_token_delta",
    "prompt_semantic_distance",
    "skill_diff_size",
    "skills_added",
    "skills_removed",
    "tool_schema_change",
    "tool_count_delta",
    "loop_parameter_delta",
    "context_policy_change",
    "test_policy_change",
    "search_policy_change",
    "tool_call_distribution_shift",
    "shell_command_distribution_shift",
    "test_execution_shift",
    "files_read_shift",
    "files_written_shift",
    "dependency_operation_shift",
    "token_usage_shift",
    "context_peak_shift",
    "wall_clock_shift",
    "action_sequence_distance",
)

RESOURCE_COLUMNS = (
    "tokens",
    "wall_clock_seconds",
    "tool_calls",
    "tests_executed",
    "files_read",
    "files_written",
    "context_peak",
    "timeouts",
    "crashes",
    "dependency_operations",
    "cpu_seconds",
    "memory_mb",
)

AUTONOMOUS_COLUMNS = (
    "run_id", "scaffold", "operator", "task_family", "round", "transition_id",
    "incumbent_hash", "candidate_hash", "delta_proxy", "delta_actor", "delta_strategic",
    "proxy_incumbent_score", "proxy_candidate_score", "actor_incumbent_score", "actor_candidate_score",
    "proxy_positive", "actor_reversal", "strategic_reversal",
    *(f"footprint_{name}" for name in FOOTPRINT_COLUMNS),
    *(f"resource_{name}" for name in RESOURCE_COLUMNS),
    "footprint", "resource_metrics",
)
PROMOTION_CANDIDATE_COLUMNS = (
    "run_id", "round", "candidate_id", "candidate_hash", "proxy_delta", "operator", "scaffold",
)
HF_QUERY_COLUMNS = (
    "method", "run_id", "round", "candidate_id", "query_index", "task_id", "paired_delta",
    "cost", "posterior_before", "posterior_after", "EVSI", "observed_information_gain", "candidate_batch_hash",
    "logical_hf_query", "physical_pair_evaluation", "cache_hit",
)
PROMOTION_RESULT_COLUMNS = (
    "method", "run_id", "round", "selected_candidate", "true_best_candidate", "ISR", "hf_cost",
)
CLOSED_LOOP_COLUMNS = (
    "method", "scaffold", "run_id", "round", "proxy_score", "gate_score",
    "assessment_score_if_terminal", "CISR", "resource_metrics",
    "logical_hf_queries", "pre_decision_pair_evaluations",
    "post_decision_truth_evaluations", "post_decision_truth_pair_evaluations",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"canonical JSONL row must be a mapping: {path}")
        rows.append(dict(value))
    return rows


def _phase_directory(root: Path, name: str, *, confirmatory: bool) -> Path | None:
    # A confirmatory materializer must never silently promote a DEV artifact.
    candidates = (
        (root / "results/v15" / name,)
        if confirmatory
        else (root / "results/v15" / f"dev-{name}",)
    )
    for candidate in candidates:
        manifest = candidate / "manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and payload.get("status") == "COMPLETED":
                return candidate
    return None


def _resource_scalar(resources: Mapping[str, Any], key: str) -> Any:
    direct = resources.get(key)
    if isinstance(direct, (int, float, str)):
        return direct
    values: list[float] = []
    for name in ("proxy_incumbent", "proxy_candidate"):
        nested = resources.get(name)
        if isinstance(nested, Mapping) and isinstance(nested.get(key), (int, float)):
            values.append(float(nested[key]))
    return sum(values) / len(values) if values else None


def _flatten_transition(row: Mapping[str, Any], strategic: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output = dict(row)
    footprint = row.get("footprint", {})
    footprint = footprint if isinstance(footprint, Mapping) else {}
    resources = row.get("resource_metrics", {})
    resources = resources if isinstance(resources, Mapping) else {}
    for name in FOOTPRINT_COLUMNS:
        output[f"footprint_{name}"] = footprint.get(name)
    for name in RESOURCE_COLUMNS:
        output[f"resource_{name}"] = _resource_scalar(resources, name)
    # Do not carry executor-specific paths or raw trajectory payloads into the
    # public canonical table.  Those values are useful to the internal response
    # audit, but the canonical release only needs reproducible scalar summaries.
    output["footprint"] = {name: footprint.get(name) for name in FOOTPRINT_COLUMNS if name in footprint}
    output["resource_metrics"] = {
        name: _resource_scalar(resources, name)
        for name in RESOURCE_COLUMNS
        if _resource_scalar(resources, name) is not None
    }
    if strategic is not None and strategic.get("delta_strategic") is not None:
        output["delta_strategic"] = strategic.get("delta_strategic")
        output["strategic_reversal"] = bool(strategic.get("strategic_reversal", False))
    return output


def refresh_canonical_tables(root: Path, *, confirmatory: bool = False) -> dict[str, Any]:
    """Materialize canonical tables from the latest completed phase artifacts.

    The function is conservative: an incomplete or missing external phase is
    represented by an empty schema, while completed DEV/confirmatory artifacts
    are copied with their source hashes recorded in the manifest.
    """

    root = Path(root).resolve()
    output = root / "results/v15/canonical"
    transition_dir = _phase_directory(root, "external-transition-audit", confirmatory=confirmatory)
    strategic_dir = root / "results/v15" / ("external-strategic-response" if confirmatory else "dev-external-strategic-response")
    strategic_rows = {
        str(row.get("transition_id")): row
        for row in _read_jsonl(strategic_dir / "response_audits.jsonl")
        if row.get("transition_id")
    }
    transition_rows = [
        _flatten_transition(row, strategic_rows.get(str(row.get("transition_id"))))
        for row in _read_jsonl(transition_dir / "autonomous_transitions.jsonl")
    ] if transition_dir is not None else []
    candidate_rows = _read_jsonl(transition_dir / "promotion_candidates.jsonl") if transition_dir is not None else []
    promotion_dir = _phase_directory(root, "external-promotion", confirmatory=confirmatory)
    promotion_rows = _read_jsonl(promotion_dir / "promotion_results.jsonl") if promotion_dir is not None else []
    query_rows = _read_jsonl(promotion_dir / "hf_queries.jsonl") if promotion_dir is not None else []
    closed_dir = _phase_directory(root, "external-closed-loop", confirmatory=confirmatory)
    closed_rows = _read_jsonl(closed_dir / "closed_loop_results.jsonl") if closed_dir is not None else []
    transition_sources = [path for path in (transition_dir, strategic_dir, promotion_dir, closed_dir) if path is not None]
    tables = {
        "autonomous_transitions": transition_rows,
        "promotion_candidates": candidate_rows,
        "hf_queries": query_rows,
        "promotion_results": promotion_rows,
        "closed_loop_results": closed_rows,
    }
    schemas = {
        "autonomous_transitions": AUTONOMOUS_COLUMNS,
        "promotion_candidates": PROMOTION_CANDIDATE_COLUMNS,
        "hf_queries": HF_QUERY_COLUMNS,
        "promotion_results": PROMOTION_RESULT_COLUMNS,
        "closed_loop_results": CLOSED_LOOP_COLUMNS,
    }
    outputs: dict[str, dict[str, str]] = {}
    for name, rows in tables.items():
        paths = write_table(rows, output / name, columns=schemas[name])
        outputs[name] = {kind: str(path) for kind, path in paths.items()}
    manifest = {
        "schema_version": "pivot-v15-canonical-tables-2",
        "phase": "CONFIRMATORY" if confirmatory and transition_dir is not None else "DEV",
        "confirmatory": bool(confirmatory and transition_dir is not None),
        "rows": {name: len(rows) for name, rows in tables.items()},
        "schemas": {name: list(columns) for name, columns in schemas.items()},
        "outputs": outputs,
        "source_manifests": [str(path / "manifest.json") for path in transition_sources if (path / "manifest.json").is_file()],
        "source_hashes": {
            str(path / "manifest.json"): file_hash(path / "manifest.json")
            for path in transition_sources
            if (path / "manifest.json").is_file()
        },
        "strategic_response_rows": len(strategic_rows),
        "note": "Canonical tables are materialized from completed phase artifacts; incomplete phases remain empty and are not treated as scientific evidence.",
    }
    manifest["sha256"] = {
        relative: file_hash(Path(path))
        for table in outputs.values()
        for relative, path in table.items()
        if relative == "parquet"
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_empty_canonical_tables(root: Path, *, phase: str = "NOT_RUN") -> dict[str, Any]:
    """Create schema-bearing empty tables without inventing scientific rows."""

    output = Path(root) / "results/v15/canonical"
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "manifest.json"
    if phase == "NOT_RUN" and existing_manifest.is_file():
        try:
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, Mapping) and any(int(value) > 0 for value in existing.get("rows", {}).values()):
            return dict(existing)
    schemas = {
        "autonomous_transitions": AUTONOMOUS_COLUMNS,
        "promotion_candidates": PROMOTION_CANDIDATE_COLUMNS,
        "hf_queries": HF_QUERY_COLUMNS,
        "promotion_results": PROMOTION_RESULT_COLUMNS,
        "closed_loop_results": CLOSED_LOOP_COLUMNS,
    }
    outputs: dict[str, Any] = {}
    for name, columns in schemas.items():
        paths = write_table([], output / name, columns=columns)
        outputs[name] = {kind: str(path) for kind, path in paths.items()}
    manifest = {
        "schema_version": "pivot-v15-canonical-tables-1",
        "phase": phase,
        "confirmatory": False,
        "rows": {name: 0 for name in schemas},
        "schemas": {name: list(columns) for name, columns in schemas.items()},
        "outputs": outputs,
        "note": "Empty schema-bearing tables are placeholders until a locked external run is available.",
    }
    manifest["sha256"] = {
        relative: file_hash(Path(path))
        for table in outputs.values()
        for relative, path in table.items()
        if relative == "parquet"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "AUTONOMOUS_COLUMNS",
    "CLOSED_LOOP_COLUMNS",
    "FOOTPRINT_COLUMNS",
    "HF_QUERY_COLUMNS",
    "PROMOTION_CANDIDATE_COLUMNS",
    "PROMOTION_RESULT_COLUMNS",
    "RESOURCE_COLUMNS",
    "refresh_canonical_tables",
    "write_empty_canonical_tables",
]
