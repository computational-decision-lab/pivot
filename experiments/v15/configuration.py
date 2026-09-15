"""Frozen confirmatory protocol construction and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .control_plane import probe_adapters
from .planes import redact_task_manifest
from .protocol import canonical_json, file_hash


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping in {path}")
    return payload


def build_lock(root: Path) -> dict[str, Any]:
    """Build a pre-outcome lock from versioned local protocol inputs."""

    config_path = root / "configs/v15/confirmatory.yaml"
    task_path = root / "configs/v15/task_manifest.json"
    external_path = root / "configs/v15/external_versions.json"
    config = _load_yaml(config_path)
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    task_manifest_sha256 = file_hash(task_path)
    external = json.loads(external_path.read_text(encoding="utf-8"))
    adapter_probe = {item.name: item for item in probe_adapters(root)}

    def scaffold_runtime(name: str, configured: dict[str, Any]) -> dict[str, Any]:
        record = adapter_probe.get(name)
        if record is None:
            return {**configured, "status": "adapter_not_available"}
        return {
            **configured,
            "status": "available" if record.available else "adapter_not_available",
            "runtime_version": record.version,
            "runtime_source": record.source,
        }

    lock: dict[str, Any] = {
        "schema_version": "pivot-v15-confirmatory-lock-1",
        "phase": "pre_outcome_freeze",
        "frameworks": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "adapters": [item.__dict__ for item in probe_adapters(root)],
            "pinned_sources": external,
        },
        "scaffolds": {
            "primary": scaffold_runtime(
                "mini-SWE-agent", dict(config.get("scaffolds", {}).get("primary", {}))
            ),
            "replication": scaffold_runtime(
                "Pi", dict(config.get("scaffolds", {}).get("replication", {}))
            ),
        },
        "foundation_model": config.get(
            "foundation_model", {"status": "pending_authorization_and_pin", "identifier": None}
        ),
        "operator_implementations": ["harness_skill_evolution", "mutation_self_edit"],
        "operator_prompts": config.get("operator_prompts", {}),
        "operator_information": "D_proxy only; no gate or assessment outcomes",
        "allowed_policy_edit_space": config.get(
            "allowed_policy_edit_space",
            [
                "system_prompt",
                "agent_loop_config",
                "tool_policy",
                "search_policy",
                "test_policy",
                "context_policy",
            ],
        ),
        "sandbox": config.get("sandbox", {}),
        "dependency_lock": config.get("sandbox", {}).get("dependency_lock"),
        "figure_definitions": config.get("figure_definitions", {}),
        "strategic_response": config.get("strategic_response", {}),
        # Keep membership and hashes in the lock, but never embed hidden task
        # instructions or repository contents in a protocol artifact that may
        # be copied to reviewers.
        "sealed_planes": redact_task_manifest(
            task_manifest, source_sha256=task_manifest_sha256
        ),
        "resource_limits": config["resource_limits"],
        "T": int(config["rounds"]),
        "K": int(config["candidates_per_round"]),
        "seed_registry": config["seed_registry"],
        "pairing_rules": config["pairing_rules"],
        "metric_definitions": config["metrics"],
        "footprint_features": config["footprint_features"],
        "pivot": config["pivot"],
        "baselines": config["baselines"],
        "hf_budgets": config["hf_budgets"],
        "bootstrap": config["bootstrap"],
        "primary_hypotheses": config["primary_hypotheses"],
        "case_selection_rule": config["case_selection_rule"],
        "resource_plan": config.get("resource_plan", {}),
        "confirmatory_execution": "NOT_RUN",
        "protocol_input_hashes": {
            "configs/v15/confirmatory.yaml": file_hash(config_path),
            "configs/v15/task_manifest.json": task_manifest_sha256,
            "configs/v15/external_versions.json": file_hash(external_path),
        },
        "outcome_chasing": False,
    }
    lock["lock_hash"] = hashlib.sha256(canonical_json(lock).encode("utf-8")).hexdigest()
    return lock


def write_lock(root: Path, output: Path) -> Path:
    """Write a new pre-outcome lock.

    This low-level writer is intentionally explicit.  Callers that may run
    after the protocol has been frozen should use :func:`ensure_lock`, which
    verifies and preserves an existing lock instead of replacing it.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_lock(root), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _lock_digest(payload: dict[str, Any]) -> str:
    """Compute the canonical digest for a lock payload without its digest."""

    unsigned = dict(payload)
    unsigned.pop("lock_hash", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def is_confirmatory_open(value: Any) -> bool:
    """Return whether a lock status represents opened outcomes.

    Older locks used a string status while the current state machine records a
    structured opening event.  Centralizing this check avoids the subtle
    ``dict in {..}`` TypeError and lets audits read both formats safely.
    """

    if isinstance(value, Mapping):
        return str(value.get("status", "")).upper() not in {"", "NOT_RUN"}
    return value is not None and value != "NOT_RUN"


def open_confirmatory_lock(root: Path, output: Path, *, phase: str) -> Path:
    """Atomically mark the pre-outcome lock as opened for a phase.

    Opening is the boundary between protocol preparation and confirmatory
    outcome collection.  It is idempotent for later phases and refuses a
    malformed or already-terminal lock.  The protocol inputs remain hashed and
    unchanged; only the auditable lifecycle status is added.
    """

    output = Path(output)
    ensure_lock(root, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"invalid confirmatory lock: {output}")
    status = payload.get("confirmatory_execution")
    if is_confirmatory_open(status):
        if isinstance(status, Mapping) and str(status.get("status", "")).upper() == "OPENED":
            return output
        raise PermissionError("confirmatory lock has a terminal execution status")
    opened_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload["confirmatory_execution"] = {
        "status": "OPENED",
        "phase": str(phase),
        "opened_at_utc": opened_at,
    }
    payload["confirmatory_opened_phase"] = str(phase)
    payload["lock_hash"] = _lock_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def ensure_lock(root: Path, output: Path) -> Path:
    """Create a lock once, then preserve and verify it on later calls.

    A pre-outcome lock is an immutable scientific boundary.  Rebuilding it on
    every report/audit invocation would silently change environment metadata
    (for example adapter availability) and invalidate the preregistration.
    An existing lock with a missing or incorrect digest is rejected rather than
    repaired in place.
    """

    output = Path(output)
    if not output.is_file():
        return write_lock(root, output)
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid confirmatory lock: {output}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"invalid confirmatory lock: {output}")
    stored = payload.get("lock_hash")
    if not isinstance(stored, str) or stored != _lock_digest(payload):
        raise ValueError(f"lock hash mismatch: {output}")
    return output


def refresh_pre_outcome_lock(root: Path, output: Path) -> Path:
    """Refresh an unopened lock after protocol-input changes.

    Refresh is intentionally explicit and preserves the old signed payload in
    an append-only history file.  Once any confirmatory outcome is opened, the
    operation is refused.
    """

    output = Path(output)
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid confirmatory lock: {output}") from exc
        if not isinstance(previous, dict):
            raise TypeError(f"invalid confirmatory lock: {output}")
        if is_confirmatory_open(previous.get("confirmatory_execution")):
            raise PermissionError("cannot refresh lock after confirmatory outcomes are opened")
        stored = previous.get("lock_hash")
        if not isinstance(stored, str) or stored != _lock_digest(previous):
            raise ValueError(f"lock hash mismatch: {output}")
        history = output.parent / "confirmatory_lock_history.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(previous, sort_keys=True) + "\n")
        refreshed = build_lock(root)
        refreshed["predecessor_lock_hash"] = stored
        refreshed["lock_hash"] = hashlib.sha256(
            canonical_json({key: value for key, value in refreshed.items() if key != "lock_hash"}).encode("utf-8")
        ).hexdigest()
        output.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
    return write_lock(root, output)
