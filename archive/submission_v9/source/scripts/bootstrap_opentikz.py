#!/usr/bin/env python3
"""Install and verify the pinned OpenTikZ checkout used by the paper."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("OpenTikZ lock must be a JSON object")
    for key in ("repository", "commit", "template_id", "files"):
        if key not in payload:
            raise ValueError(f"OpenTikZ lock is missing {key!r}")
    if not isinstance(payload["files"], dict) or not payload["files"]:
        raise ValueError("OpenTikZ lock files must be a non-empty object")
    return cast(dict[str, Any], payload)


def _verify_checkout(root: Path, lock: dict[str, Any]) -> dict[str, str]:
    if not (root / ".git").exists():
        raise RuntimeError(f"OpenTikZ destination is not a git checkout: {root}")
    current = _run(["git", "rev-parse", "HEAD"], cwd=root)
    expected = str(lock["commit"])
    if current != expected:
        raise RuntimeError(
            f"OpenTikZ checkout is at a different commit ({current}); expected {expected}."
        )
    for relative, expected_hash in lock["files"].items():
        candidate = root / str(relative)
        if not candidate.is_file():
            raise RuntimeError(f"OpenTikZ lock file is missing from checkout: {relative}")
        actual_hash = _sha256(candidate)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                f"OpenTikZ hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
    template = root / "templates" / str(lock["template_id"])
    if not template.is_dir():
        raise RuntimeError(f"OpenTikZ template directory is missing: {template}")
    return {"destination": str(root), "commit": current, "template_id": str(lock["template_id"])}


def bootstrap_opentikz(lock_path: Path, destination: Path) -> dict[str, str]:
    """Install a detached pinned checkout, refusing to mutate a drifted one."""

    lock = _load_lock(Path(lock_path))
    root = Path(destination).resolve()
    if root.exists():
        return _verify_checkout(root, lock)
    root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "--initial-branch=main", str(root)])
    _run(["git", "remote", "add", "origin", str(lock["repository"])], cwd=root)
    _run(["git", "fetch", "--depth=1", "origin", str(lock["commit"])], cwd=root)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=root)
    return _verify_checkout(root, lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("configs/tooling/opentikz.json"))
    parser.add_argument("--destination", type=Path, default=Path(".tools/opentikz"))
    args = parser.parse_args()
    print(json.dumps(bootstrap_opentikz(args.lock, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
