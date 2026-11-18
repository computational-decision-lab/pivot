"""Content inventories and verified recovery receipts for owned cloud results."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _files(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("archive root must be a directory, not a symlink")
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not permitted in result archives")
        if path.is_file():
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("result changed during inventory")
            records[path.relative_to(root).as_posix()] = {"sha256": digest, "bytes": after.st_size}
        elif not path.is_dir():
            raise ValueError("result archive contains a non-regular file")
    if not records:
        raise ValueError("result archive must contain at least one artifact (including failures)")
    return records


def create_inventory(root: Path, output: Path, *, instance_id: str, run_id: str) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("inventory must be stored outside its result tree")
    files = _files(root)
    record = {
        "version": 1,
        "instance_id": instance_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }
    atomic_json(output, record)
    return record


def _check_tree(root: Path, inventory: dict[str, Any]) -> None:
    actual = _files(root)
    if actual != inventory["files"]:
        raise ValueError("archive checksum, path or size mismatch")
    if (
        len(actual) != inventory["file_count"]
        or sum(row["bytes"] for row in actual.values()) != inventory["total_bytes"]
    ):
        raise ValueError("archive counts do not match inventory")


def verify_archive(
    inventory_path: Path, downloaded: Path, archived: Path, receipt_path: Path
) -> dict[str, Any]:
    """Verify every object in two copies and randomly restore one to a third path.

    The inventory must originate on the worker after outputs have stopped changing.
    This proves content integrity; persistence/remote-storage availability is a separate gate.
    """
    downloaded, archived = downloaded.resolve(), archived.resolve()
    if downloaded.is_relative_to(archived) or archived.is_relative_to(downloaded):
        raise ValueError("download and archive must be independent trees")
    inventory = json.loads(inventory_path.read_text())
    _check_tree(downloaded, inventory)
    _check_tree(archived, inventory)
    selected = secrets.choice(sorted(inventory["files"]))
    with tempfile.TemporaryDirectory(prefix="pivot-restore-") as directory:
        restored = Path(directory) / "artifact"
        shutil.copyfile(archived / selected, restored)
        restored_hash = sha256_file(restored)
        if restored_hash != inventory["files"][selected]["sha256"]:
            raise ValueError("archive restore hash mismatch")
    receipt = {
        "version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": inventory["instance_id"],
        "run_id": inventory["run_id"],
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "downloaded": str(downloaded),
        "archived": str(archived),
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "restore_path": selected,
        "restore_sha256": restored_hash,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def validate_receipt(receipt_path: Path, *, instance_id: str, run_id: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    if receipt["instance_id"] != instance_id or receipt["run_id"] != run_id:
        raise ValueError("archive receipt belongs to another instance or run")
    inventory_path = Path(receipt["inventory_path"])
    if sha256_file(inventory_path) != receipt["inventory_sha256"]:
        raise ValueError("archive inventory was changed")
    inventory = json.loads(inventory_path.read_text())
    if inventory["instance_id"] != instance_id or inventory["run_id"] != run_id:
        raise ValueError("archive inventory identity mismatch")
    for key in ("file_count", "total_bytes"):
        if receipt[key] != inventory[key]:
            raise ValueError("archive receipt count mismatch")
    restored = inventory["files"].get(receipt["restore_path"])
    if not restored or restored["sha256"] != receipt["restore_sha256"]:
        raise ValueError("archive restore evidence mismatch")
    downloaded, archived = Path(receipt["downloaded"]), Path(receipt["archived"])
    if downloaded.resolve().is_relative_to(archived.resolve()) or archived.resolve().is_relative_to(
        downloaded.resolve()
    ):
        raise ValueError("archive copies must be independent")
    _check_tree(downloaded, inventory)
    _check_tree(archived, inventory)
    return receipt
