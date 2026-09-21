"""Crash-safe checkpoints for bounded experiment units.

Every attempt is journaled as ``running`` before its worker starts, then that
record is updated to a terminal state. A completed unit is reused only while
its output still matches the recorded size and SHA-256 hash.

Duplicate prevention uses a non-blocking POSIX advisory lock. That lock proves
liveness only among processes on the host and filesystem lock domain sharing
this store. The kernel releases a local holder's lock when its process exits;
the store never infers that a worker on another host died from elapsed time.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias


UnitResult: TypeAlias = bytes | bytearray | str | Path
UnitWorker: TypeAlias = Callable[[], UnitResult]


class UnitLockedError(RuntimeError):
    """Raised when another process is already running the same unit."""


class ProtocolMismatchError(FileExistsError):
    """Raised when an output is already owned by another run identity."""


class CheckpointStore:
    """Persist experiment-unit outputs and resumable attempt metadata."""

    def __init__(
        self,
        root: Path,
        *,
        manifest_name: str = "manifest.json",
        hostname: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / manifest_name
        self._manifest_lock_path = self.root / f".{manifest_name}.lock"
        self._locks_dir = self.root / ".unit-locks"
        self._locks_dir.mkdir(exist_ok=True)
        self.hostname = hostname or socket.gethostname()
        self.git_commit = git_commit if git_commit is not None else _current_git_commit()

    def records(self) -> list[dict[str, Any]]:
        """Return all persisted attempt records in manifest order."""

        with self._manifest_lock(shared=True):
            return self._read_records()

    def is_valid(
        self,
        unit_key: str,
        *,
        output_path: Path,
        config_hash: str,
        code_hash: str,
        dependency_hash: str,
        git_commit: str | None = None,
    ) -> bool:
        """Return whether the latest completed attempt can be resumed."""

        with self._manifest_lock(shared=True):
            return self._is_valid(
                unit_key,
                Path(output_path),
                config_hash,
                code_hash,
                dependency_hash,
                self.git_commit if git_commit is None else git_commit,
                self._read_records(),
            )

    def run_unit(
        self,
        unit_key: str,
        worker: UnitWorker,
        *,
        output_path: Path,
        config_hash: str,
        code_hash: str,
        dependency_hash: str,
        instance_id: str | None = None,
        git_commit: str | None = None,
    ) -> dict[str, Any]:
        """Run one unit, or return its intact completed record without running it.

        ``worker`` returns bytes (or text) for the output, or a path whose
        bytes will be copied to ``output_path``. Outputs and the manifest are
        both replaced atomically. ``config_hash``, ``code_hash``, and
        ``dependency_hash`` are required resume identities. Locks are local to
        the host/filesystem lock domain; remote attempts are never timed out.
        """

        output = Path(output_path)
        effective_git_commit = self.git_commit if git_commit is None else git_commit
        with self._unit_lock(unit_key):
            with self._manifest_lock(shared=False):
                records = self._read_records()
                self._assert_output_owner(unit_key, output, records)
                valid = self._valid_completed_record(output, records)
                if valid is not None:
                    if valid.get("unit_key") == unit_key and _identity_matches(
                        valid, config_hash, code_hash, dependency_hash, effective_git_commit
                    ):
                        return valid
                    raise ProtocolMismatchError(
                        f"valid completed output {output} belongs to a different "
                        "checkpoint identity; "
                        "use a new output path or run"
                    )
                self._resolve_running_attempts(unit_key, records)
                attempt = _next_attempt(unit_key, records)
                prior_output = self._preserve_output(output, unit_key, attempt, "prior")
                running = self._record(
                    unit_key=unit_key,
                    status="running",
                    attempt=attempt,
                    started_at=_timestamp(),
                    completed_at=None,
                    instance_id=instance_id,
                    git_commit=effective_git_commit,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    dependency_hash=dependency_hash,
                    output_path=output,
                    prior_output=prior_output,
                )
                records.append(running)
                self._write_records(records)

            try:
                result = worker()
                payload = _result_bytes(result)
                _atomic_write(output, payload)
                digest, byte_count = _file_digest(output)
                completed_at = _timestamp()
                updates = {
                    "status": "completed",
                    "completed_at": completed_at,
                    "finished_at": completed_at,
                    "sha256": digest,
                    "bytes": byte_count,
                }
            except Exception as error:
                failed_output = self._preserve_output(output, unit_key, attempt, "failed")
                completed_at = _timestamp()
                updates = {
                    "status": "failed",
                    "completed_at": completed_at,
                    "finished_at": completed_at,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "failed_output_path": failed_output[0] if failed_output else None,
                    "failed_output_sha256": failed_output[1] if failed_output else None,
                    "failed_output_bytes": failed_output[2] if failed_output else None,
                }

            with self._manifest_lock(shared=False):
                records = self._read_records()
                record = _attempt_record(unit_key, attempt, records)
                record.update(updates)
                self._write_records(records)
            return record

    # Short aliases keep the utility convenient for small runners.
    run = run_unit
    execute = run_unit

    def _record(
        self,
        *,
        unit_key: str,
        status: str,
        attempt: int,
        started_at: str,
        completed_at: str | None,
        instance_id: str | None,
        git_commit: str | None,
        config_hash: str,
        code_hash: str,
        dependency_hash: str,
        output_path: Path,
        sha256: str | None = None,
        byte_count: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        prior_output: tuple[str, str, int] | None = None,
    ) -> dict[str, Any]:
        return {
            "unit_key": unit_key,
            "status": status,
            "attempt": attempt,
            "started_at": started_at,
            "completed_at": completed_at,
            "finished_at": completed_at,
            "hostname": self.hostname,
            "instance_id": instance_id,
            "git_commit": self.git_commit if git_commit is None else git_commit,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "dependency_hash": dependency_hash,
            "output_path": str(output_path),
            "sha256": sha256,
            "bytes": byte_count,
            "error_type": error_type,
            "error_message": error_message,
            "prior_output_path": prior_output[0] if prior_output else None,
            "prior_output_sha256": prior_output[1] if prior_output else None,
            "prior_output_bytes": prior_output[2] if prior_output else None,
            "failed_output_path": None,
            "failed_output_sha256": None,
            "failed_output_bytes": None,
        }

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"invalid checkpoint manifest: {self.manifest_path}")
        return [dict(item) for item in payload]

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        payload = json.dumps(records, indent=2, sort_keys=True).encode() + b"\n"
        _atomic_write(self.manifest_path, payload)

    def _is_valid(
        self,
        unit_key: str,
        output_path: Path,
        config_hash: str,
        code_hash: str,
        dependency_hash: str,
        git_commit: str | None,
        records: list[dict[str, Any]],
    ) -> bool:
        record = self._valid_completed_record(output_path, records)
        return bool(
            record is not None
            and record.get("unit_key") == unit_key
            and _identity_matches(record, config_hash, code_hash, dependency_hash, git_commit)
        )

    def _valid_completed_record(
        self, output_path: Path, records: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not output_path.is_file():
            return None
        matching = [
            record
            for record in records
            if record.get("status") == "completed"
            and _same_path(record.get("output_path"), output_path)
        ]
        if not matching:
            return None
        digest, byte_count = _file_digest(output_path)
        return next(
            (
                record
                for record in reversed(matching)
                if record.get("sha256") == digest and record.get("bytes") == byte_count
            ),
            None,
        )

    def _assert_output_owner(
        self, unit_key: str, output_path: Path, records: list[dict[str, Any]]
    ) -> None:
        owner = next(
            (
                record
                for record in records
                if record.get("unit_key") != unit_key
                and _same_path(record.get("output_path"), output_path)
            ),
            None,
        )
        if owner is not None:
            raise ProtocolMismatchError(
                f"output path {output_path} is already reserved by unit {owner.get('unit_key')!r}; "
                "use a new output path or run"
            )

    def _resolve_running_attempts(
        self, unit_key: str, records: list[dict[str, Any]]
    ) -> None:
        running = [
            record
            for record in records
            if record.get("unit_key") == unit_key and record.get("status") == "running"
        ]
        remote = next(
            (record for record in running if record.get("hostname") != self.hostname),
            None,
        )
        if remote is not None:
            raise UnitLockedError(
                f"unit may still be running on host {remote.get('hostname')!r}; "
                "cannot verify remote liveness with a host-local flock"
            )
        for record in running:
            completed_at = _timestamp()
            record.update(
                status="interrupted",
                completed_at=completed_at,
                finished_at=completed_at,
                error_type="InterruptedAttempt",
                error_message="previous local process exited without recording a terminal state",
            )

    def _preserve_output(
        self, output_path: Path, unit_key: str, attempt: int, kind: str
    ) -> tuple[str, str, int] | None:
        if not output_path.is_file():
            return None
        digest, byte_count = _file_digest(output_path)
        preserved = (
            self.root
            / "preserved-outputs"
            / hashlib.sha256(unit_key.encode()).hexdigest()
            / f"attempt-{attempt}-{kind}-{digest[:12]}.bin"
        )
        if not preserved.exists():
            _atomic_write(preserved, output_path.read_bytes())
        return str(preserved), digest, byte_count

    @contextmanager
    def _manifest_lock(self, *, shared: bool) -> Iterator[None]:
        self._manifest_lock_path.touch(exist_ok=True)
        with self._manifest_lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _unit_lock(self, unit_key: str) -> Iterator[None]:
        lock = self._locks_dir / f"{hashlib.sha256(unit_key.encode()).hexdigest()}.lock"
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise UnitLockedError(f"unit is already running: {unit_key}") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _attempt_record(
    unit_key: str, attempt: int, records: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        return next(
            record
            for record in reversed(records)
            if record.get("unit_key") == unit_key and record.get("attempt") == attempt
        )
    except StopIteration as error:
        message = f"checkpoint attempt disappeared: {unit_key} attempt {attempt}"
        raise RuntimeError(message) from error


def _next_attempt(unit_key: str, records: list[dict[str, Any]]) -> int:
    attempts = [
        int(record.get("attempt", 0))
        for record in records
        if record.get("unit_key") == unit_key
    ]
    return max(attempts, default=0) + 1


def _identity_matches(
    record: dict[str, Any],
    config_hash: str,
    code_hash: str,
    dependency_hash: str,
    git_commit: str | None,
) -> bool:
    return (
        record.get("config_hash") == config_hash
        and record.get("code_hash") == code_hash
        and record.get("dependency_hash") == dependency_hash
        and record.get("git_commit") == git_commit
    )


def _same_path(recorded_path: Any, output_path: Path) -> bool:
    if not isinstance(recorded_path, str) or not recorded_path:
        return False
    return Path(recorded_path).resolve(strict=False) == output_path.resolve(strict=False)


def _result_bytes(result: UnitResult) -> bytes:
    if isinstance(result, bytes):
        return result
    if isinstance(result, bytearray):
        return bytes(result)
    if isinstance(result, str):
        return result.encode()
    if isinstance(result, Path):
        return result.read_bytes()
    raise TypeError(f"unsupported worker result: {type(result).__name__}")


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
