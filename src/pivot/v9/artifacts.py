from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def current_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_provenance(path: Path, *, experiment_id: str, config: dict[str, Any], root: Path, seed_list: list[int]) -> None:
    write_json(
        path,
        {
            "schema_version": "pivot-v9-provenance-v1",
            "experiment_id": experiment_id,
            "config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "seed_list": seed_list,
            "source_commit": current_commit(root),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "paired_context": True,
            "raw_outcomes_preserved": True,
        },
    )


def build_manifest(directory: Path, *, experiment_id: str, status: str) -> dict[str, Any]:
    files = {
        path.relative_to(directory).as_posix(): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    payload = {
        "schema_version": "pivot-v9-manifest-v1",
        "experiment_id": experiment_id,
        "status": status,
        "files": files,
    }
    write_json(directory / "manifest.json", payload)
    return payload


def validate_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        return {"valid": False, "errors": ["manifest.json missing"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, record in payload.get("files", {}).items():
        target = directory / name
        if not target.is_file():
            errors.append(f"missing:{name}")
        elif sha256(target) != str(record.get("sha256")):
            errors.append(f"hash:{name}")
    return {"valid": not errors, "errors": errors, "status": payload.get("status")}
