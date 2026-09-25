from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.bootstrap_figure_tools import bootstrap_figure_tools


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def test_bootstrap_figure_tools_pins_each_repository_and_checks_hashes(tmp_path: Path) -> None:
    repositories = []
    for name in ("skills", "plots"):
        remote = tmp_path / f"{name}-remote"
        remote.mkdir()
        _git(remote, "init", "--initial-branch=main")
        _git(remote, "config", "user.email", "test@example.invalid")
        _git(remote, "config", "user.name", "figure-tools-test")
        file = remote / "README.md"
        file.write_text(f"{name}\n", encoding="utf-8")
        _git(remote, "add", "README.md")
        _git(remote, "commit", "-m", "initial")
        repositories.append(
            {
                "name": name,
                "repository": str(remote),
                "commit": _git(remote, "rev-parse", "HEAD"),
                "license": "MIT" if name == "skills" else "not-declared",
                "files": {"README.md": hashlib.sha256(file.read_bytes()).hexdigest()},
            }
        )
    lock = tmp_path / "figure-tools.json"
    lock.write_text(json.dumps({"schema_version": 1, "repositories": repositories}), encoding="utf-8")

    result = bootstrap_figure_tools(lock, tmp_path / "installed")
    assert set(result["tools"]) == {"skills", "plots"}
    assert bootstrap_figure_tools(lock, tmp_path / "installed")["tools"]["skills"]["license"] == "MIT"
