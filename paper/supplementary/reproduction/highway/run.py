#!/usr/bin/env python3
"""Run the frozen HighwayEnv code from a clone or an extracted supplement."""

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


def verify_source(root: Path) -> dict:
    manifest = json.loads((root / "source-manifest.json").read_text())
    source = root / "source"
    paths = list(source.rglob("*"))
    if source.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("frozen source contains a symlink")
    actual = {p.relative_to(source).as_posix() for p in paths if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("frozen source membership mismatch")
    for name, record in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid source manifest path")
        data = (source / name).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"frozen source checksum mismatch: {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("original", "replication"), default="replication")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify hashes and imports; no rollouts")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required for an experiment")
    output = args.output.resolve() if args.output is not None else None
    if output is not None and output.exists():
        parser.error("use a new output directory to preserve existing evidence")
    root = Path(__file__).resolve().parent
    manifest = verify_source(root)
    workspace = Path(tempfile.mkdtemp(prefix="highway-reproduction-"))
    shutil.copytree(root / "source", workspace, dirs_exist_ok=True)
    runner = workspace / "scripts/run_highway_calibrated.py"
    if args.cohort == "original":
        shutil.copyfile(workspace / "scripts/run_highway_original.py", runner)
    # The exported snapshot gets its own honest Git identity. It never pretends
    # to be the historical commit from which the experiment was launched.
    git = ["git", "-C", str(workspace)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "-c", "user.name=Artifact reproduction", "-c",
                    "user.email=reproduction@invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Import verified frozen source snapshot"], check=True)
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    env = dict(os.environ, PYTHONPATH=str(workspace) + os.pathsep + str(workspace / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(runner)]
    if args.check_only:
        command += ["--help"]
    else:
        command += ["--root", str(workspace), "--output", str(output), "--expected-commit", commit]
        if args.cohort == "replication":
            command.append("--replication")
    receipt = {
        "cohort": args.cohort, "historical_commits": {k: v for k, v in manifest.items() if k != "files"},
        "snapshot_commit": commit, "source_manifest_sha256": hashlib.sha256(
            (root / "source-manifest.json").read_bytes()).hexdigest(),
        "check_only": args.check_only,
    }
    (workspace / "reproduction-launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"workspace": str(workspace), **receipt}), flush=True)
    return subprocess.run(command, cwd=workspace, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
