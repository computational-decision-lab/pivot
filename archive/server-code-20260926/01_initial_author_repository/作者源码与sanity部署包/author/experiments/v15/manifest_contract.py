"""Backfill and validate phase-level provenance fields without rerunning science.

The external runners write their own manifests.  Older bounded DEV artifacts
were produced before the terminal-state fields were added, so this module
provides a deterministic metadata migration.  It never changes rows, scores,
task assignments, or protocol files, and it refuses to rewrite confirmatory
manifests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import TERMINAL_STATES, canonical_json, file_hash

_PHASE_DIRS = (
    "dev-external-transition-audit",
    "dev-external-promotion",
    "dev-external-closed-loop",
    "dev-external-strategic-response",
    "dev-external-ablations",
    "dev-pi-replication",
)


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def _attempted(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status", "")).upper()
    return _as_bool(payload.get("execution_attempted"), default=False) or status not in {"", "NOT_RUN"}


def _as_bool(value: object, default: bool = False) -> bool:
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


def _terminal_for_dev(payload: Mapping[str, Any]) -> str | None:
    if not _attempted(payload):
        return None
    current = payload.get("terminal_state")
    if isinstance(current, str) and current in TERMINAL_STATES:
        return current
    failures = int(payload.get("execution_failure_count", 0) or 0)
    if str(payload.get("status", "")).upper() == "IMPLEMENTATION_FAILURE" or failures:
        return "IMPLEMENTATION_FAILURE"
    # Bounded DEV runs cannot satisfy the registered independent-N rule.
    return "UNDERPOWERED"


def backfill_dev_manifests(root: Path) -> dict[str, Any]:
    """Add missing closure fields to existing DEV manifests.

    Returns a machine-readable migration summary.  Confirmatory directories
    are inspected but never modified; a pre-existing confirmatory terminal
    state is owned by the scientific decision artifact.
    """

    root = Path(root).resolve()
    changed: list[str] = []
    skipped_confirmatory: list[str] = []
    missing: list[str] = []
    for directory_name in _PHASE_DIRS:
        path = root / "results/v15" / directory_name / "manifest.json"
        if not path.is_file():
            missing.append(directory_name)
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, Mapping):
            continue
        payload = dict(loaded)
        if _as_bool(payload.get("confirmatory"), default=False):
            skipped_confirmatory.append(directory_name)
            continue
        before = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["execution_attempted"] = _attempted(payload)
        payload["terminal_state"] = _terminal_for_dev(payload)
        payload.setdefault("design_status", "VALIDATED_DEV" if payload["terminal_state"] == "UNDERPOWERED" else "PENDING_ANALYSIS")
        payload.setdefault("leakage_detected", False)
        payload.setdefault("outcome_chasing", False)
        if "role_access_log" not in payload:
            payload["role_access_log"] = payload.get("access_log", [])
        archive_root = root / "results/v15" / "dev-external-candidate-archive"
        source_candidates = {
            "dev-external-transition-audit": root / "results/v15/dev-external-transition-audit/promotion_candidates.jsonl",
            "dev-external-promotion": root / "results/v15/dev-external-promotion/promotion_candidates.jsonl",
        }
        source = source_candidates.get(directory_name)
        if source is not None and source.is_file():
            archive_file = archive_root / "promotion_candidates.jsonl"
            archive_manifest_file = archive_root / "manifest.json"
            if not archive_manifest_file.is_file():
                from .evidence import freeze_candidate_archive

                freeze_candidate_archive(source, archive_root, phase="DEV", confirmatory=False)
            if archive_file.is_file() and archive_manifest_file.is_file():
                archive_payload = json.loads(archive_manifest_file.read_text(encoding="utf-8"))
                if isinstance(archive_payload, Mapping) and archive_payload.get("archive_sha256") == file_hash(archive_file):
                    payload["candidate_archive_frozen"] = True
                    payload["candidate_archive_path"] = str(archive_file.relative_to(root))
                    payload["candidate_archive_sha256"] = file_hash(archive_file)
                    payload["candidate_archive_manifest_sha256"] = file_hash(archive_manifest_file)
                else:
                    payload["candidate_archive_frozen"] = False
            else:
                payload["candidate_archive_frozen"] = False
        elif directory_name in {
            "dev-external-promotion",
            "dev-external-ablations",
        }:
            archive_file = archive_root / "promotion_candidates.jsonl"
            archive_manifest_file = archive_root / "manifest.json"
            archive_payload = (
                json.loads(archive_manifest_file.read_text(encoding="utf-8"))
                if archive_manifest_file.is_file()
                else {}
            )
            frozen = bool(
                archive_file.is_file()
                and isinstance(archive_payload, Mapping)
                and archive_payload.get("archive_sha256") == file_hash(archive_file)
            )
            payload["candidate_archive_frozen"] = frozen
            if frozen:
                payload["candidate_archive_path"] = str(archive_file.relative_to(root))
                payload["candidate_archive_sha256"] = file_hash(archive_file)
                payload["candidate_archive_manifest_sha256"] = file_hash(archive_manifest_file)
        else:
            payload.setdefault("candidate_archive_frozen", False)
        # Keep the migration idempotent and preserve the runner's canonical
        # hash convention.
        payload["manifest_sha256"] = _manifest_hash(payload)
        after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if before != after:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            changed.append(directory_name)
    # Older local candidate archives predate the self-hashing manifest field.
    # Repair that metadata in place only when the content hash still agrees;
    # candidate rows themselves are never rewritten here.
    for directory_name in ("candidate-archive", "dev-external-candidate-archive"):
        directory = root / "results/v15" / directory_name
        archive = directory / "promotion_candidates.jsonl"
        manifest_path = directory / "manifest.json"
        if not archive.is_file() or not manifest_path.is_file():
            continue
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, Mapping):
            continue
        payload = dict(loaded)
        if payload.get("archive_sha256") != file_hash(archive):
            continue
        payload.setdefault("schema_version", "pivot-v15-candidate-archive-1")
        payload.setdefault("phase", "DEV")
        payload.setdefault("immutable", True)
        payload.setdefault("regeneration_allowed", False)
        payload.setdefault("confirmatory", False)
        payload.setdefault("source_manifest_sha256", None)
        payload["manifest_sha256"] = _manifest_hash(payload)
        before = json.dumps(loaded, sort_keys=True, separators=(",", ":"))
        after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if before != after:
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            changed.append(directory_name)
    return {
        "schema_version": "pivot-v15-manifest-contract-1",
        "changed": changed,
        "skipped_confirmatory": skipped_confirmatory,
        "missing": missing,
        "manifest_hashes": {
            name: file_hash(root / "results/v15" / name / "manifest.json")
            for name in _PHASE_DIRS
            if (root / "results/v15" / name / "manifest.json").is_file()
        },
    }


__all__ = ["backfill_dev_manifests"]
