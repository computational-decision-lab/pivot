"""Role-separated proxy, gate, and assessment task planes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _validate_relative_file_path(value: str) -> str:
    """Reject task paths that could escape a fresh sandbox root."""

    path = value.replace("\\", "/")
    candidate = Path(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"task file paths must be relative and normalized: {value!r}")
    return path


def redact_task_manifest(
    payload: Mapping[str, Any], *, source_sha256: str | None = None
) -> dict[str, Any]:
    """Return a public task-plane summary without task instructions or files.

    The full manifest is required by the local execution harness, but it must
    not be copied into a confirmatory lock or reviewer-facing archive.  This
    function deliberately retains only identifiers, hashes, and family labels;
    those are enough to verify membership and pairing without reconstructing a
    hidden task.  It also accepts an already-redacted summary so sanitizers can
    be applied idempotently.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("task manifest must be a mapping")
    raw_planes: Any = payload.get("planes", payload)
    if not isinstance(raw_planes, Mapping):
        raise TypeError("task manifest must contain planes")
    public_planes: dict[str, list[dict[str, str]]] = {}
    for plane in ("proxy", "gate", "assessment"):
        raw_tasks = raw_planes.get(plane, [])
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise TypeError(f"{plane} plane must be a sequence")
        entries: list[dict[str, str]] = []
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise TypeError(f"{plane} task entries must be mappings")
            task_id = str(item.get("task_id", ""))
            family = str(item.get("family", ""))
            if not task_id or not family:
                raise ValueError(f"{plane} task entries require task_id and family")
            supplied_hash = item.get("task_hash")
            if isinstance(supplied_hash, str) and supplied_hash:
                task_hash = supplied_hash
            elif isinstance(item.get("files"), Mapping):
                task_hash = TaskSpec(
                    task_id=task_id,
                    family=family,
                    files=item["files"],
                    metadata=item.get("metadata", {}),
                ).task_hash
            else:
                raise ValueError(f"{plane} task {task_id} has no verifiable task hash")
            entries.append({"task_id": task_id, "task_hash": task_hash, "family": family})
        public_planes[plane] = entries
    summary: dict[str, Any] = {
        "schema_version": "pivot-v15-task-manifest-public-1",
        "sealed": True,
        "public_redaction": True,
        "plane_counts": {plane: len(entries) for plane, entries in public_planes.items()},
        "planes": public_planes,
        "outcome_visibility": {
            "operator": ["proxy"],
            "pivot": ["proxy", "gate"],
            "promotion": ["proxy", "gate"],
            "terminal_assessor": ["assessment"],
        },
    }
    if source_sha256:
        summary["source_sha256"] = str(source_sha256)
    return summary


class AccessDenied(PermissionError):
    """Raised when a component requests a sealed task plane it cannot observe."""


@dataclass(frozen=True)
class TaskSpec:
    """A reproducible task definition suitable for a fresh sandbox."""

    task_id: str
    family: str
    files: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.family:
            raise ValueError("task_id and family are required")
        normalized_files = {
            _validate_relative_file_path(str(path)): str(content) for path, content in self.files.items()
        }
        if not normalized_files:
            raise ValueError("task files must not be empty")
        object.__setattr__(self, "files", MappingProxyType(normalized_files))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        payload = {
            "task_id": self.task_id,
            "family": self.family,
            "files": normalized_files,
            "metadata": dict(self.metadata),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        object.__setattr__(self, "task_hash", digest)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "files": dict(self.files),
            "metadata": dict(self.metadata),
            "task_hash": self.task_hash,
        }


_ALLOWED_ROLES = {
    "proxy": frozenset({"operator", "proxy_evaluator", "audit"}),
    "gate": frozenset({"promotion", "pivot", "baseline", "audit"}),
    "assessment": frozenset({"terminal_assessor", "audit"}),
}


@dataclass(frozen=True)
class SealedDataPlanes:
    """Task pools with explicit, auditable role access.

    Task objects are immutable, so returning a tuple cannot permit an operator
    to alter the sealed definition.  The access log is intentionally retained
    for leakage audits.
    """

    proxy: Sequence[TaskSpec]
    gate: Sequence[TaskSpec]
    assessment: Sequence[TaskSpec]
    access_log: list[dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        proxy = tuple(self.proxy)
        gate = tuple(self.gate)
        assessment = tuple(self.assessment)
        object.__setattr__(self, "proxy", proxy)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "assessment", assessment)
        ids = [task.task_id for task in proxy + gate + assessment]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be disjoint across sealed planes")

    def tasks(self, plane: str, *, role: str) -> tuple[TaskSpec, ...]:
        """Return tasks only when the caller's scientific role is authorized."""

        if plane not in _ALLOWED_ROLES:
            raise ValueError(f"unknown task plane: {plane}")
        if role not in _ALLOWED_ROLES[plane]:
            self.access_log.append({"plane": plane, "role": role, "outcome": "denied"})
            raise AccessDenied(f"role '{role}' cannot access {plane} tasks")
        self.access_log.append({"plane": plane, "role": role, "outcome": "granted"})
        values = getattr(self, plane)
        return tuple(values)

    def manifest(self) -> dict[str, Any]:
        """Expose plane IDs/hashes without task contents for provenance checks."""

        return {
            plane: [
                {"task_id": task.task_id, "task_hash": task.task_hash, "family": task.family}
                for task in getattr(self, plane)
            ]
            for plane in ("proxy", "gate", "assessment")
        }

    def access_log_snapshot(self) -> tuple[dict[str, str], ...]:
        """Return an immutable copy suitable for a phase manifest."""

        return tuple(dict(event) for event in self.access_log)

    def manifest_sha256(self) -> str:
        """Hash only task IDs, hashes, and family labels (never task contents)."""

        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_task_planes(path: Path) -> SealedDataPlanes:
    """Load and validate a sealed task manifest without exposing contents."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("sealed") is not True:
        raise ValueError("task manifest must explicitly set sealed=true")
    raw_planes = payload.get("planes")
    if not isinstance(raw_planes, Mapping):
        raise TypeError("task manifest must contain planes")

    def parse(name: str) -> tuple[TaskSpec, ...]:
        raw_tasks = raw_planes.get(name, [])
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise TypeError(f"{name} plane must be a sequence")
        parsed: list[TaskSpec] = []
        for index, item in enumerate(raw_tasks):
            if not isinstance(item, Mapping):
                raise TypeError(f"{name} task entry {index} must be a mapping")
            try:
                task = TaskSpec(
                    task_id=str(item["task_id"]),
                    family=str(item["family"]),
                    files=item["files"],
                    metadata=item.get("metadata", {}),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {name} task entry {index}") from exc
            supplied_hash = item.get("task_hash")
            if supplied_hash is not None and str(supplied_hash) != task.task_hash:
                raise ValueError(f"{name} task {task.task_id} hash mismatch")
            parsed.append(task)
        return tuple(parsed)

    return SealedDataPlanes(proxy=parse("proxy"), gate=parse("gate"), assessment=parse("assessment"))
