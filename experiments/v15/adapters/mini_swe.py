"""Dry-run contract for the pinned mini-SWE-agent scaffold."""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any

from .base import AdapterRequest, AdapterResult


class MiniSWEAdapter:
    scaffold = "mini-SWE-agent"
    import_name = "minisweagent"
    command = "mini-swe-agent"

    def status(self) -> dict[str, Any]:
        imported = importlib.util.find_spec(self.import_name) is not None
        executable = shutil.which(self.command) is not None
        return {
            "scaffold": self.scaffold,
            "import_name": self.import_name,
            "command": self.command,
            "available": imported or executable,
            "execution_status": "NOT_RUN",
            "model_calls_performed": 0,
            "container_executions": 0,
            "reason": "adapter contract is dry-run only; model and sealed tasks are not opened",
        }

    def command_preview(self, task_id: str, policy_hash: str) -> dict[str, Any]:
        request = AdapterRequest(task_id=task_id, policy_hash=policy_hash, seed=0)
        return {
            "scaffold": self.scaffold,
            "request_hash": request.request_hash,
            "command": [self.command, "--task-id", "<sealed-task>", "--policy-hash", "<policy-hash>"],
            "dry_run": True,
            "task_id_redacted": task_id != "<sealed-task>",
        }

    def execute(self, request: AdapterRequest) -> AdapterResult:
        return AdapterResult(
            status="NOT_RUN",
            scaffold=self.scaffold,
            request_hash=request.request_hash,
            reason="mini-SWE-agent execution requires an installed pinned scaffold and an authorized model",
        )
