"""Audit V15 terminal-state closure, provenance, and role separation.

The execution runners intentionally keep their manifests lightweight.  This
module is the cross-phase audit that checks the stronger scientific contract:
every attempted phase has one valid terminal decision, hashes are internally
consistent, and no role-access log records a forbidden sealed-plane read.
It never infers a favourable outcome from a missing field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit
from .configuration import _lock_digest, is_confirmatory_open
from .planes import load_task_planes
from .protocol import TERMINAL_STATES, canonical_json, file_hash

_PHASES = {
    "external-transition-audit": "transition_analysis",
    "external-promotion": "promotion_analysis",
    "external-closed-loop": "closed_loop_analysis",
    "external-pi-replication": "pi_analysis",
    "external-strategic-response": "strategic_analysis",
    "external-ablations": "ablation_analysis",
}
_PHASE_ALIASES = {
    "external-pi-replication": ("external-pi-replication", "dev-pi-replication"),
}
_ROLE_FORBIDDEN = {
    ("operator", "gate"),
    ("operator", "assessment"),
    ("proxy_evaluator", "gate"),
    ("proxy_evaluator", "assessment"),
    ("pivot", "assessment"),
    ("promotion", "assessment"),
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _bool(value: object, default: bool = False) -> bool:
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


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def _phase_access_issues(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    logs = payload.get("role_access_log", payload.get("access_log", []))
    if not isinstance(logs, list):
        return ["access log is not a list"]
    for index, event in enumerate(logs):
        if not isinstance(event, Mapping):
            issues.append(f"access event {index} is not an object")
            continue
        role = str(event.get("role", ""))
        plane = str(event.get("plane", ""))
        outcome = str(event.get("outcome", "granted")).casefold()
        if outcome == "granted" and (role, plane) in _ROLE_FORBIDDEN:
            issues.append(f"forbidden granted access: role={role} plane={plane}")
    return issues


def _decision_for_phase(root: Path, phase: str, artifact_name: str) -> tuple[dict[str, Any] | None, str]:
    phase_dir = root / "results/v15" / phase
    local = phase_dir / "scientific_decision.json"
    global_path = root / "artifacts/v15" / f"{artifact_name.removesuffix('_analysis')}_scientific_decision.json"
    for path in (local, global_path):
        payload = _json(path)
        if isinstance(payload, Mapping):
            return dict(payload), str(path)
    return None, str(local)


def _check_decision(
    decision: Mapping[str, Any] | None,
    *,
    phase: str,
    attempted: bool,
    issues: list[str],
) -> str | None:
    if not attempted:
        return None
    if decision is None:
        issues.append(f"{phase}: attempted phase has no scientific_decision.json")
        return None
    state = decision.get("terminal_state")
    if not isinstance(state, str) or state not in TERMINAL_STATES:
        issues.append(f"{phase}: invalid terminal state {state!r}")
        return None
    return state


def _task_plane_check(root: Path, issues: list[str]) -> dict[str, Any]:
    path = root / "configs/v15/task_manifest.json"
    if not path.is_file():
        issues.append("sealed task manifest is missing")
        return {"present": False}
    try:
        planes = load_task_planes(path)
        manifest = planes.manifest()
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"sealed task manifest invalid: {type(exc).__name__}: {exc}")
        return {"present": True, "valid": False}
    ids = [item["task_id"] for values in manifest.values() for item in values]
    disjoint = len(ids) == len(set(ids))
    if not disjoint:
        issues.append("task IDs are not disjoint across planes")
    public_path = root / "configs/v15/task_manifest.public.json"
    public_match = None
    if public_path.is_file():
        public = _json(public_path)
        if isinstance(public, Mapping):
            public_match = public.get("planes") == manifest
            if not public_match:
                issues.append("public task manifest membership/hash summary does not match sealed manifest")
    return {
        "present": True,
        "valid": True,
        "plane_counts": {name: len(values) for name, values in manifest.items()},
        "disjoint": disjoint,
        "manifest_sha256": file_hash(path),
        "public_membership_match": public_match,
    }


def _candidate_archive_check(root: Path, issues: list[str]) -> dict[str, Any]:
    """Validate every materialized candidate archive without opening policies."""

    summaries: dict[str, Any] = {}
    for name in ("dev-external-candidate-archive", "external-candidate-archive", "candidate-archive"):
        directory = root / "results/v15" / name
        archive = directory / "promotion_candidates.jsonl"
        manifest_path = directory / "manifest.json"
        if not archive.is_file() and not manifest_path.is_file():
            continue
        payload = _json(manifest_path)
        valid = isinstance(payload, Mapping)
        if not valid:
            issues.append(f"{name}: candidate-archive manifest is invalid")
            summaries[name] = {"valid": False}
            continue
        manifest = dict(payload)
        archive_hash = file_hash(archive) if archive.is_file() else None
        digest_valid = manifest.get("archive_sha256") == archive_hash
        internal_hash = manifest.get("manifest_sha256") == _manifest_digest(manifest)
        if not digest_valid:
            issues.append(f"{name}: archive content hash mismatch")
        if not internal_hash:
            issues.append(f"{name}: candidate-archive manifest hash mismatch")
        if manifest.get("immutable") is not True or manifest.get("regeneration_allowed") is not False:
            issues.append(f"{name}: archive is not marked immutable")
        summaries[name] = {
            "valid": digest_valid and internal_hash and manifest.get("immutable") is True,
            "archive_sha256": archive_hash,
            "manifest_sha256": file_hash(manifest_path),
            "row_count": manifest.get("row_count"),
            "confirmatory": _bool(manifest.get("confirmatory"), default=False),
        }
    return summaries


def audit_terminal_states(root: Path) -> dict[str, Any]:
    """Return and persist the V15 terminal-state and leakage audit."""

    root = Path(root).resolve()
    issues: list[str] = []
    phases: dict[str, Any] = {}
    result_dirs = root / "results/v15"
    if result_dirs.is_dir():
        for phase_name, artifact_name in _PHASES.items():
            directory_names = _PHASE_ALIASES.get(phase_name, (phase_name, f"dev-{phase_name}"))
            for directory_name in directory_names:
                directory = result_dirs / directory_name
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                payload = _json(manifest_path)
                key = directory_name
                phase_result: dict[str, Any] = {"manifest": str(manifest_path.relative_to(root))}
                if not isinstance(payload, Mapping):
                    issues.append(f"{key}: manifest is not valid JSON object")
                    phases[key] = phase_result | {"valid": False}
                    continue
                manifest = dict(payload)
                status = str(manifest.get("status", ""))
                attempted = status not in {"", "NOT_RUN"} or bool(
                    manifest.get("execution_attempted", False)
                )
                digest = manifest.get("manifest_sha256")
                digest_ok = isinstance(digest, str) and digest == _manifest_digest(manifest)
                if not digest_ok and digest is not None:
                    issues.append(f"{key}: manifest_sha256 mismatch")
                outcome_chasing = _bool(manifest.get("outcome_chasing"), default=False)
                leakage = _bool(manifest.get("leakage_detected"), default=False)
                if outcome_chasing:
                    issues.append(f"{key}: outcome_chasing is true")
                if leakage:
                    issues.append(f"{key}: leakage_detected is true")
                access_issues = _phase_access_issues(manifest)
                issues.extend(f"{key}: {item}" for item in access_issues)
                decision, decision_path = _decision_for_phase(root, directory_name, artifact_name)
                terminal = _check_decision(decision, phase=key, attempted=attempted, issues=issues)
                raw_terminal = manifest.get("terminal_state")
                if isinstance(raw_terminal, str) and raw_terminal not in TERMINAL_STATES:
                    issues.append(f"{key}: manifest terminal state is invalid {raw_terminal!r}")
                if terminal is not None and isinstance(raw_terminal, str) and raw_terminal not in {terminal, ""}:
                    issues.append(f"{key}: manifest/decision terminal states disagree")
                phases[key] = {
                    "status": status,
                    "attempted": attempted,
                    "confirmatory": _bool(manifest.get("confirmatory"), default=False),
                    "manifest_hash_valid": digest_ok,
                    "terminal_state": terminal,
                    "manifest_terminal_state": raw_terminal,
                    "decision_path": decision_path,
                    "outcome_chasing": outcome_chasing,
                    "leakage_detected": leakage,
                    "access_issues": access_issues,
                    "valid": digest_ok
                    and not access_issues
                    and not outcome_chasing
                    and not leakage
                    and (not attempted or terminal is not None),
                }

    lock_path = root / "experiments/v15/confirmatory_lock.json"
    lock = _json(lock_path)
    lock_check: dict[str, Any] = {"present": lock_path.is_file()}
    if isinstance(lock, Mapping):
        lock_check["hash_valid"] = lock.get("lock_hash") == _lock_digest(dict(lock))
        if not lock_check["hash_valid"]:
            issues.append("confirmatory lock hash mismatch")
        lock_check["confirmatory_open"] = is_confirmatory_open(lock.get("confirmatory_execution"))
        lock_check["outcome_chasing"] = _bool(lock.get("outcome_chasing"), default=False)
        if lock_check["outcome_chasing"]:
            issues.append("confirmatory lock outcome_chasing is true")
    elif lock_path.is_file():
        issues.append("confirmatory lock is not a JSON object")

    task_check = _task_plane_check(root, issues)
    archive_check = _candidate_archive_check(root, issues)
    attempted_phases = [name for name, item in phases.items() if item.get("attempted")]
    valid = not issues and all(bool(item.get("valid")) for item in phases.values()) and bool(task_check.get("valid", True))
    result: dict[str, Any] = {
        "schema_version": "pivot-v15-terminal-state-audit-1",
        "status": "PASS" if valid else "BLOCKED",
        "valid": valid,
        "issues": issues,
        "phase_count": len(phases),
        "attempted_phase_count": len(attempted_phases),
        "phases": phases,
        "task_planes": task_check,
        "candidate_archives": archive_check,
        "lock": lock_check,
        "terminal_states": sorted(
            {str(item.get("terminal_state")) for item in phases.values() if item.get("terminal_state")}
        ),
        "no_outcome_chasing": not any(bool(item.get("outcome_chasing")) for item in phases.values()),
        "note": "This audit validates closure and provenance; it does not upgrade DEV or underpowered artifacts into confirmatory evidence.",
    }
    return write_audit(
        root,
        "terminal_state_audit",
        result,
        "Terminal-State and Leakage Audit",
        "Every attempted V15 phase must resolve to one closed terminal state without role leakage or outcome chasing.",
    )


if __name__ == "__main__":
    cli(audit_terminal_states, "Audit V15 terminal states and sealed-plane provenance")


__all__ = ["audit_terminal_states"]
