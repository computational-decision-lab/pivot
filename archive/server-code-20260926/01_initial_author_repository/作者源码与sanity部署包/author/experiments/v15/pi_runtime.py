"""Pinned Pi runtime helpers used by the cross-scaffold DEV adapter.

The source checkout and the built CLI live below ``.tools`` and are excluded
from the anonymous paper release.  This module records only public paths,
commit hashes, and exit metadata; credentials and task contents stay outside
the manifest.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .protocol import AgentPolicy, canonical_json


def pi_source_root(root: Path) -> Path:
    """Return the pinned Pi source checkout."""

    return Path(root).resolve() / ".tools/v15/external/pi"


def pi_cli_path(root: Path) -> Path:
    """Resolve the built Pi CLI, honoring an explicit non-secret override."""

    override = os.getenv("PIVOT_PI_CLI")
    candidates = [
        Path(override) if override else None,
        pi_source_root(root) / "packages/coding-agent/dist/cli.js",
        pi_source_root(root) / "packages/coding-agent/dist/bundle/cli.js",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return (pi_source_root(root) / "packages/coding-agent/dist/cli.js").resolve()


def _git_commit(source: Path) -> str | None:
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _node_version() -> str | None:
    result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False, timeout=10)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def pi_runtime_status(root: Path) -> dict[str, Any]:
    """Describe the local Pi build without starting a model session."""

    source = pi_source_root(root)
    cli = pi_cli_path(root)
    node = _node_version()
    return {
        "scaffold": "Pi",
        "source_root": str(source),
        "source_commit": _git_commit(source),
        "cli_path": str(cli),
        "cli_exists": cli.is_file(),
        "node_available": node is not None,
        "node_version": node,
        "bwrap_available": shutil.which("bwrap") is not None,
        "extension": str(Path(root).resolve() / "experiments/v15/pi_gateway_extension.ts"),
        "execution_status": "NOT_RUN",
        "model_calls_performed": 0,
        "container_executions": 0,
        "confirmatory_claim_allowed": False,
    }


def pi_policy_prompt(policy: AgentPolicy) -> str:
    """Map the registered policy dimensions to Pi's system prompt."""

    search = dict(policy.search_policy)
    tests = dict(policy.test_policy)
    context = dict(policy.context_policy)
    return (
        f"{policy.system_prompt}\n\n"
        "Use the available repository tools to inspect and edit only the task repository. "
        "Do not modify evaluator files, task metadata, or files outside the repository. "
        "Run the registered tests before finishing and stop after the requested repair. "
        f"Search depth target: {search.get('depth', 2)}; file budget: {search.get('max_files', 12)}. "
        f"Run tests: {bool(tests.get('run_tests', True))}; "
        f"context budget: {context.get('max_tokens', 2048)} tokens."
    )


def pi_request_digest(task_id: str, policy: AgentPolicy, seed: int) -> str:
    """Hash a Pi request without embedding task contents or credentials."""

    return hashlib.sha256(
        canonical_json({"task_id": task_id, "policy_hash": policy.policy_hash, "seed": int(seed)}).encode("utf-8")
    ).hexdigest()


__all__ = ["pi_cli_path", "pi_policy_prompt", "pi_request_digest", "pi_runtime_status", "pi_source_root"]
