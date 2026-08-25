#!/usr/bin/env python3
"""Run the controlled value-fidelity versus improvement-fidelity contrast."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone
from pivot.transfer.evaluator_contrast import EvaluatorContrastConfig, run_evaluator_contrast
from pivot.transfer.sampling import stratified_transition_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled value versus improvement fidelity contrast")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sweeps/e4_value_vs_improvement.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("results/raw/e4-value-vs-improvement"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "source"
    manifest = run_first_milestone(
        source_dir,
        PerformativeConfig(**payload.get("world", {})),
        payload.get("seeds", []),
        payload.get("candidate_scales", []),
        response_strengths=payload.get("response_strengths"),
        optimization_strengths=payload.get("optimization_strengths"),
    )
    rows = [
        json.loads(line)
        for line in (source_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hf_budget = int(payload.get("hf_budget", max(1, len(rows) // 2)))
    train_rows = stratified_transition_sample(rows, min(hf_budget, len(rows) - 1))
    contrast_payload = payload.get("contrast", {})
    result = run_evaluator_contrast(
        rows,
        {str(row["transition_id"]) for row in train_rows},
        EvaluatorContrastConfig(hf_budget=hf_budget, **contrast_payload),
    )
    result["source_manifest"] = manifest.__dict__
    result["config_path"] = str(args.config)
    (args.output / "comparison.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for evaluator, evaluator_rows in result["rows"].items():
        _write_csv(args.output / f"{evaluator}_rows.csv", evaluator_rows)
    provenance = {
        "experiment": result["experiment"],
        "diagnostic_type": result["diagnostic_type"],
        "config_hash": _sha256(args.config),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_version": "pivot-controlled-v1",
        "dataset_version": "synthetic-performative-v1",
        "paired": True,
        "source_manifest": manifest.__dict__,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        evaluator: {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "policy_value_mae",
                "policy_rank_correlation",
                "improvement_differential_error",
                "improvement_sign_consistency",
                "improvement_reversal_rate",
                "update_selection_regret",
                "cumulative_true_improvement",
            }
        }
        for evaluator, metrics in result["metrics"].items()
    }
    print(json.dumps({"rows": len(rows), "hf_budget": hf_budget, "metrics": summary}, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
