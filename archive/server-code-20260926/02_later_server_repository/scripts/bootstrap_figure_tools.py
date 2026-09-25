#!/usr/bin/env python3
"""Install and verify the pinned external figure-style repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lock(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("figure-tools lock must be a schema_version=1 object")
    repositories = payload.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("figure-tools lock repositories must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for entry in repositories:
        if not isinstance(entry, dict):
            raise TypeError("figure-tools repository entries must be objects")
        for key in ("name", "repository", "commit", "license", "files"):
            if key not in entry:
                raise ValueError(f"figure-tools repository is missing {key!r}")
        if not isinstance(entry["files"], dict) or not entry["files"]:
            raise ValueError(f"figure-tools files must be non-empty: {entry['name']}")
        normalized.append(cast(dict[str, Any], entry))
    return normalized


def _verify_checkout(root: Path, lock: dict[str, Any]) -> dict[str, str]:
    if not (root / ".git").exists():
        raise RuntimeError(f"figure-tools destination is not a git checkout: {root}")
    current = _run(["git", "rev-parse", "HEAD"], cwd=root)
    expected = str(lock["commit"])
    if current != expected:
        raise RuntimeError(f"{lock['name']} checkout is at {current}; expected {expected}")
    for relative, expected_hash in cast(dict[str, Any], lock["files"]).items():
        candidate = root / str(relative)
        if not candidate.is_file():
            raise RuntimeError(f"{lock['name']} lock file is missing: {relative}")
        actual = _sha256(candidate)
        if actual != str(expected_hash):
            raise RuntimeError(f"{lock['name']} hash mismatch for {relative}: {actual} != {expected_hash}")
    return {"name": str(lock["name"]), "commit": current, "license": str(lock["license"])}


def bootstrap_figure_tools(lock_path: Path, destination: Path) -> dict[str, Any]:
    """Install detached checkouts, refusing drifted destinations."""

    repositories = _load_lock(Path(lock_path))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, str]] = {}
    for lock in repositories:
        root = destination / str(lock["name"])
        if not root.exists():
            _run(["git", "init", "--initial-branch=main", str(root)])
            _run(["git", "remote", "add", "origin", str(lock["repository"])], cwd=root)
            _run(["git", "fetch", "--depth=1", "origin", str(lock["commit"])], cwd=root)
            _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=root)
        results[str(lock["name"])] = _verify_checkout(root, lock)
    return {"destination": str(destination), "tools": results}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=project_root / "configs/tooling/figure_tools.json")
    parser.add_argument("--destination", type=Path, default=project_root / ".tools/figure-tools")
    args = parser.parse_args()
    print(json.dumps(bootstrap_figure_tools(args.lock, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
