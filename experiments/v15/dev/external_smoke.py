"""One-task DEV smoke for the Inspect/mini-SWE external runtime.

This command is intentionally never confirmatory: it opens only the proxy
plane, writes under a DEV-specific artifact directory, and does not modify the
confirmatory lock or candidate archive.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..external_runtime import ExecutionRecord, evaluate_with_inspect, resolve_runtime_settings
from ..planes import load_task_planes
from ..protocol import AgentPolicy, file_hash


def summarize_records(records: Sequence[ExecutionRecord]) -> dict[str, Any]:
    """Summarize external execution counters without treating rows as N."""

    if not records:
        return {
            "status": "IMPLEMENTATION_FAILURE",
            "terminal_state": "IMPLEMENTATION_FAILURE",
            "container_executions": 0,
            "model_calls_performed": 0,
            "success_rate": 0.0,
        }
    failures = [record for record in records if record.status != "COMPLETED"]
    return {
        "status": "IMPLEMENTATION_FAILURE" if failures else "COMPLETED",
        "terminal_state": "IMPLEMENTATION_FAILURE" if failures else None,
        "container_executions": len(records),
        "model_calls_performed": int(
            sum(record.resource_metrics.get("model_calls", 0.0) for record in records)
        ),
        "success_rate": sum(record.success for record in records) / len(records),
        "task_ids": [record.task_id for record in records],
        "trajectory_paths": [record.trajectory for record in records],
    }


def run_external_smoke(root: Path, *, seed: int = 10001, task_limit: int = 1) -> dict[str, Any]:
    """Run a bounded real external task and emit only DEV evidence."""

    root = Path(root).resolve()
    manifest_path = root / "configs/v15/task_manifest.json"
    planes = load_task_planes(manifest_path)
    tasks = planes.tasks("proxy", role="operator")[: max(1, int(task_limit))]
    artifact_root = root / "results/v15/dev-external-smoke/artifacts"
    log_root = root / "results/v15/dev-external-smoke/inspect-logs"
    settings = resolve_runtime_settings(root, artifact_root=artifact_root, log_root=log_root)
    policy = AgentPolicy.minimal().with_updates(metadata={"phase": "DEV_EXTERNAL_SMOKE"})
    records = evaluate_with_inspect(
        tasks,
        policy,
        settings,
        seed=seed,
        phase="dev_external_smoke",
        role="proxy_evaluator",
        run_id="smoke",
    )
    summary = summarize_records(records)
    payload: dict[str, Any] = {
        **summary,
        "phase": "DEV",
        "confirmatory": False,
        "outcome_chasing": False,
        "task_manifest_sha256": file_hash(manifest_path),
        "runtime": settings.to_manifest(),
        "policy_hash": policy.policy_hash,
        "access_log": list(planes.access_log),
        "note": "External control-plane smoke only; no hidden gate or assessment task was opened.",
    }
    output = root / "results/v15/dev-external-smoke/manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DEV Inspect/mini-SWE task")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--task-limit", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_external_smoke(args.root, seed=args.seed, task_limit=args.task_limit), sort_keys=True))


if __name__ == "__main__":
    main()
