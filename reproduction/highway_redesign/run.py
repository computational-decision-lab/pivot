#!/usr/bin/env python3
"""Verify and run the frozen HighwayEnv B=4 redesign source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_source(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "source-manifest.json").read_text())
    source = root / "source"
    paths = list(source.rglob("*"))
    if source.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("frozen redesign source contains a symlink")
    actual = {path.relative_to(source).as_posix() for path in paths if path.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("frozen redesign source membership mismatch")
    for name, record in manifest["files"].items():
        path = source / name
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"frozen redesign source checksum mismatch: {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is used")
    root = Path(__file__).resolve().parent
    manifest = verify_source(root)
    if args.check_only:
        print(json.dumps({"valid": True, "files": len(manifest["files"])}, sort_keys=True))
        return 0
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output directory to preserve existing evidence")
    workspace = Path(tempfile.mkdtemp(prefix="highway-redesign-reproduction-"))
    shutil.copytree(root / "source", workspace, dirs_exist_ok=True)
    git = ["git", "-C", str(workspace)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=Artifact reproduction",
            "-c",
            "user.email=reproduction@invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Import verified HighwayEnv redesign source snapshot",
        ],
        check=True,
    )
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    env = dict(
        os.environ,
        PYTHONPATH=str(workspace) + os.pathsep + str(workspace / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [
        sys.executable,
        str(workspace / "scripts/run_highway_b4_redesign.py"),
        "--root",
        str(workspace),
        "--output",
        str(output),
        "--expected-commit",
        commit,
    ]
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "snapshot_commit": commit,
                "source_manifest_sha256": hashlib.sha256(
                    (root / "source-manifest.json").read_bytes()
                ).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return subprocess.run(command, cwd=workspace, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
