"""Pi cross-scaffold adapter and non-invasive status contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..pi_runtime import pi_cli_path, pi_runtime_status
from .base import AdapterRequest, AdapterResult


class PiAdapter:
    scaffold = "Pi"
    command = "pi"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    def status(self) -> dict[str, Any]:
        runtime = pi_runtime_status(self.root)
        runtime.update(
            {
                "import_name": None,
                "command": self.command,
                "available": bool(runtime["cli_exists"] and runtime["node_available"]),
                "reason": "status probe only; model and sealed tasks are not opened",
            }
        )
        return runtime

    def command_preview(self, task_id: str, policy_hash: str) -> dict[str, Any]:
        request = AdapterRequest(task_id=task_id, policy_hash=policy_hash, seed=0)
        return {
            "scaffold": self.scaffold,
            "request_hash": request.request_hash,
            "command": [
                "node",
                str(pi_cli_path(self.root)),
                "--mode",
                "json",
                "--print",
                "<sealed-task>",
                "--policy-hash",
                policy_hash,
            ],
            "policy_hash": policy_hash,
            "dry_run": True,
            "task_id_redacted": task_id != "<sealed-task>",
        }

    def execute(self, request: AdapterRequest) -> AdapterResult:
        return AdapterResult(
            status="NOT_RUN",
            scaffold=self.scaffold,
            request_hash=request.request_hash,
            reason="Pi execution requires the pinned CLI and an authorized model",
        )
