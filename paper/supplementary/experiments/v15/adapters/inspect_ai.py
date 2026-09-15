"""Inspect AI control-plane boundary with an explicit dry-run contract."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any


class InspectControlPlane:
    """Describe Inspect orchestration without opening a model or task plane."""

    name = "Inspect AI"
    import_name = "inspect_ai"

    def status(self) -> dict[str, Any]:
        return {
            "control_plane": self.name,
            "available": importlib.util.find_spec(self.import_name) is not None,
            "execution_status": "NOT_RUN",
            "model_calls_performed": 0,
            "container_executions": 0,
        }

    def build_manifest(self, task_ids: Sequence[str], *, role: str) -> dict[str, Any]:
        if role not in {"proxy_evaluator", "promotion", "pivot", "terminal_assessor"}:
            raise ValueError("unsupported Inspect role")
        normalized = [str(task_id) for task_id in task_ids]
        if len(normalized) != len(set(normalized)):
            raise ValueError("task IDs must be unique in an evaluation manifest")
        return {
            "control_plane": self.name,
            "role": role,
            "task_ids": normalized,
            "sandbox": "fresh_container_per_policy",
            "limits": "loaded from confirmatory lock",
            "dry_run": True,
        }

    def run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Return a non-result until a pinned runner is deliberately wired."""

        if not manifest.get("dry_run", False):
            raise ValueError("only dry-run manifests are accepted by this boundary adapter")
        return {
            "status": "NOT_RUN",
            "control_plane": self.name,
            "model_calls_performed": 0,
            "container_executions": 0,
            "reason": "Inspect execution requires an installed package, pinned image, and authorized model",
        }
