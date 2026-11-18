"""Durable per-simulator-call provenance, including failed/killed attempts.

An unresolved intent does not assert that a remote worker has died. Its cost is
unknown until reconciled with a terminal worker record or provider telemetry.
The checkpoint coordinator handles ownership; this journal only retains evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pivot.cloud.archive import atomic_json


class EvaluationJournal:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.attempt = uuid.uuid4().hex
        self.directory = self.root / self.attempt
        self.directory.mkdir(parents=True)
        self.counter = 0

    def evaluate(self, work: Callable[[], Any], **context: Any) -> Any:
        self.counter += 1
        call_id = f"{self.counter:06d}"
        # Only predeclared simulator metadata, never arbitrary kwargs/credentials.
        fields = {
            key: context[key]
            for key in ("phase", "method", "round_id", "policy_id", "seed", "mode")
            if key in context
        }
        common = {"attempt": self.attempt, "call_id": call_id, **fields}
        atomic_json(
            self.directory / f"{call_id}.intent.json",
            {
                **common,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
        )
        start = time.perf_counter()
        try:
            result = work()
        except BaseException as exc:
            atomic_json(
                self.directory / f"{call_id}.result.json",
                {
                    **common,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "wall_time": time.perf_counter() - start,
                    "environment_steps": None,
                    "simulator_calls": None,
                    "compute_cost": None,
                    "cost_complete": False,
                },
            )
            raise
        atomic_json(
            self.directory / f"{call_id}.result.json",
            {
                **common,
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "wall_time": time.perf_counter() - start,
                "environment_steps": result.environment_steps,
                "simulator_calls": result.simulator_calls,
                "compute_cost": result.compute_cost,
                "cost_complete": result.compute_cost is not None
                and result.metadata.get("compute_cost_complete", True),
            },
        )
        return result


def audit_evaluations(root: Path) -> dict[str, Any]:
    root = Path(root)
    summary = {
        "attempts": 0,
        "completed_evaluations": 0,
        "failed_evaluations": 0,
        "unresolved_evaluations": 0,
        "unmeasured_evaluations": 0,
        "known_environment_steps": 0,
        "known_simulator_calls": 0,
        "known_compute_cost": 0.0,
        "recorded_wall_time": 0.0,
        "cost_complete": True,
        "files": {},
    }
    if not root.exists():
        summary["cost_complete"] = False
        return summary
    attempts = set()
    for intent_path in sorted(root.glob("*/*.intent.json")):
        intent = json.loads(intent_path.read_text())
        attempts.add(intent["attempt"])
        result_path = intent_path.with_name(intent_path.name.replace(".intent.", ".result."))
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result["attempt"], result["call_id"]) != (intent["attempt"], intent["call_id"]):
                raise ValueError("evaluation journal identity mismatch")
            if result["status"] == "completed":
                summary["completed_evaluations"] += 1
                summary["known_environment_steps"] += result["environment_steps"]
                summary["known_simulator_calls"] += result["simulator_calls"]
                summary["known_compute_cost"] += result["compute_cost"] or 0.0
            else:
                summary["failed_evaluations"] += 1
            summary["recorded_wall_time"] += result["wall_time"]
            complete = result["cost_complete"]
        else:
            summary["unresolved_evaluations"] += 1
            complete = False
        if not complete:
            summary["unmeasured_evaluations"] += 1
            summary["cost_complete"] = False
        for path in (intent_path, result_path):
            if path.exists():
                data = path.read_bytes()
                summary["files"][str(path.relative_to(root))] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
    summary["attempts"] = len(attempts)
    return summary
