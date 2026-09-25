"""Frozen response audits for executed coding-agent patches.

The response layer is deliberately identity-blind.  It receives only the
candidate's changed source files, the registered test command, and the actual
execution trace.  It runs a small deterministic mutation suite and records
whether the registered tests detect each mutation.  This is a response
diagnostic, not an invented deployment utility; callers must not turn the
mutation score into ``delta_strategic`` without a separately registered
causal utility.  The paired path reports a pre-registered response-utility
difference for incumbent and candidate trees; that difference is not a
deployment-causal strategic utility without a separately registered response
world and outcome definition.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .planes import TaskSpec
from .protocol import canonical_json

_MUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("add_to_subtract", " + ", " - "),
    ("subtract_to_add", " - ", " + "),
    ("divide_to_multiply", " / ", " * "),
    ("greater_to_equal", " > ", " >= "),
    ("equal_to_not_equal", " == ", " != "),
)


@dataclass(frozen=True)
class MutationAudit:
    """Deterministic response result stored beside an execution record."""

    status: str
    changed_files: tuple[str, ...]
    attempted: int
    killed: int
    mutation_score: float | None
    trace_digest: str
    response_digest: str
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed_files": list(self.changed_files),
            "mutation_attempts": self.attempted,
            "mutations_killed": self.killed,
            "mutation_score": self.mutation_score,
            "trace_digest": self.trace_digest,
            "response_digest": self.response_digest,
            "error": self.error,
        }


@dataclass(frozen=True)
class PairedMutationAudit:
    """Same response suite applied to incumbent and candidate trees."""

    status: str
    changed_files: tuple[str, ...]
    incumbent_attempted: int
    incumbent_killed: int
    incumbent_score: float | None
    candidate_attempted: int
    candidate_killed: int
    candidate_score: float | None
    delta_strategic: float | None
    incumbent_trace_digest: str
    candidate_trace_digest: str
    response_digest: str
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed_files": list(self.changed_files),
            "incumbent_attempted": self.incumbent_attempted,
            "incumbent_killed": self.incumbent_killed,
            "incumbent_score": self.incumbent_score,
            "candidate_attempted": self.candidate_attempted,
            "candidate_killed": self.candidate_killed,
            "candidate_score": self.candidate_score,
            "delta_strategic": self.delta_strategic,
            "incumbent_trace_digest": self.incumbent_trace_digest,
            "candidate_trace_digest": self.candidate_trace_digest,
            "response_digest": self.response_digest,
            "error": self.error,
        }


def changed_files(before: Mapping[str, bytes], root: Path) -> tuple[str, ...]:
    """Return normalized paths whose bytes changed or were added/removed."""

    after: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts:
            after[path.relative_to(root).as_posix()] = path.read_bytes()
    return tuple(sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path)))


def snapshot_files(root: Path) -> dict[str, bytes]:
    """Capture a small, deterministic sandbox snapshot before execution."""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    }


def _mutant(content: str) -> tuple[str, str] | None:
    for name, old, new in _MUTATIONS:
        if old in content:
            return name, content.replace(old, new, 1)
    # A common comparison spelling has no surrounding spaces in compact code.
    match = re.search(r"(?P<op>==|!=|>=|<=|>|<)", content)
    if match:
        replacement = "!=" if match.group("op") == "==" else "=="
        return "compact_comparison", content[: match.start()] + replacement + content[match.end() :]
    return None


def run_mutation_response(
    root: Path,
    task: TaskSpec,
    *,
    changed: Sequence[str],
    trace: Sequence[str],
    run_command: Callable[[str], int],
    timeout_seconds: int = 30,
) -> MutationAudit:
    """Run the frozen mutation response on candidate source files.

    ``run_command`` is supplied by the sandbox owner, so Docker-backed
    mini-SWE and host-isolated Pi use the same response algorithm while
    retaining their own execution boundary.
    """

    source_paths = [
        str(path)
        for path in changed
        if str(path).endswith(".py") and not str(path).startswith(("tests/", ".git/"))
    ]
    if not source_paths:
        return MutationAudit(
            status="NO_MUTABLE_SOURCE",
            changed_files=tuple(sorted(str(path) for path in changed)),
            attempted=0,
            killed=0,
            mutation_score=None,
            trace_digest=hashlib.sha256(canonical_json(list(trace)).encode("utf-8")).hexdigest(),
            response_digest=hashlib.sha256(canonical_json({"changed": list(changed), "trace": list(trace)}).encode("utf-8")).hexdigest(),
        )
    attempts = 0
    killed = 0
    details: list[dict[str, Any]] = []
    try:
        for relative in source_paths:
            path = root / relative
            if not path.is_file():
                continue
            original = path.read_text(encoding="utf-8")
            mutation = _mutant(original)
            if mutation is None:
                continue
            name, mutated = mutation
            path.write_text(mutated, encoding="utf-8")
            try:
                returncode = int(run_command(str(task.metadata.get("test_command", "python -m unittest discover -v"))))
            finally:
                path.write_text(original, encoding="utf-8")
            detected = returncode != 0
            attempts += 1
            killed += int(detected)
            details.append({"path": relative, "mutation": name, "returncode": returncode, "detected": detected})
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        return MutationAudit(
            status="IMPLEMENTATION_FAILURE",
            changed_files=tuple(sorted(str(path) for path in changed)),
            attempted=attempts,
            killed=killed,
            mutation_score=(killed / attempts) if attempts else None,
            trace_digest=hashlib.sha256(canonical_json(list(trace)).encode("utf-8")).hexdigest(),
            response_digest=hashlib.sha256(canonical_json(details).encode("utf-8")).hexdigest(),
            error=f"{type(exc).__name__}: {exc}",
        )
    digest_payload = {"changed": list(changed), "trace": list(trace), "mutations": details}
    return MutationAudit(
        status="COMPLETED" if attempts else "NO_MUTATION_OPERATOR",
        changed_files=tuple(sorted(str(path) for path in changed)),
        attempted=attempts,
        killed=killed,
        mutation_score=(killed / attempts) if attempts else None,
        trace_digest=hashlib.sha256(canonical_json(list(trace)).encode("utf-8")).hexdigest(),
        response_digest=hashlib.sha256(canonical_json(digest_payload).encode("utf-8")).hexdigest(),
    )


def run_paired_mutation_response(
    incumbent_root: Path,
    candidate_root: Path,
    task: TaskSpec,
    *,
    changed: Sequence[str],
    incumbent_trace: Sequence[str],
    candidate_trace: Sequence[str],
    run_command: Callable[..., int],
) -> PairedMutationAudit:
    """Apply one frozen mutation response contract to both executed trees.

    The mutation-detection rate is a response diagnostic.  Its paired
    difference is kept separate from task success and is never silently
    promoted to deployment utility.
    """

    incumbent_path = Path(incumbent_root)
    candidate_path = Path(candidate_root)
    left = run_mutation_response(
        incumbent_path,
        task,
        changed=changed,
        trace=incumbent_trace,
        run_command=lambda command: int(run_command(command, cwd=incumbent_path)),
    )
    right = run_mutation_response(
        candidate_path,
        task,
        changed=changed,
        trace=candidate_trace,
        run_command=lambda command: int(run_command(command, cwd=candidate_path)),
    )
    if "IMPLEMENTATION_FAILURE" in {left.status, right.status}:
        status = "IMPLEMENTATION_FAILURE"
    elif left.status == right.status == "NO_MUTABLE_SOURCE":
        status = "NO_MUTABLE_SOURCE"
    elif left.status == right.status == "COMPLETED":
        status = "COMPLETED"
    else:
        status = "NO_MUTATION_OPERATOR"
    response_payload = {"incumbent": left.to_record(), "candidate": right.to_record()}
    return PairedMutationAudit(
        status=status,
        changed_files=tuple(sorted(set(left.changed_files) | set(right.changed_files))),
        incumbent_attempted=left.attempted,
        incumbent_killed=left.killed,
        incumbent_score=left.mutation_score,
        candidate_attempted=right.attempted,
        candidate_killed=right.killed,
        candidate_score=right.mutation_score,
        delta_strategic=(
            right.mutation_score - left.mutation_score
            if left.mutation_score is not None and right.mutation_score is not None
            else None
        ),
        incumbent_trace_digest=left.trace_digest,
        candidate_trace_digest=right.trace_digest,
        response_digest=hashlib.sha256(canonical_json(response_payload).encode("utf-8")).hexdigest(),
        error=left.error or right.error,
    )


__all__ = [
    "MutationAudit",
    "PairedMutationAudit",
    "changed_files",
    "run_mutation_response",
    "run_paired_mutation_response",
    "snapshot_files",
]
