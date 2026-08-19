#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.evaluation.uncertainty import bootstrap_mean_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent PIVOT run artifacts")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for directory in args.inputs:
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            records.append({"run_dir": str(directory), "status": "missing_metrics"})
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            records.append({"run_dir": str(directory), "status": "ok", "metrics": metrics})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            records.append({"run_dir": str(directory), "status": "invalid_metrics", "error": str(error)})
    valid = [record for record in records if record["status"] == "ok"]
    metric_names = sorted(
        {
            key
            for record in valid
            for key, value in record["metrics"].items()
            if isinstance(value, (int, float))
        }
    )
    summary: dict[str, object] = {}
    for name in metric_names:
        values = [float(record["metrics"][name]) for record in valid if record["metrics"].get(name) is not None]
        if not values:
            continue
        low, high = bootstrap_mean_ci(values, seed=20260819)
        summary[name] = {"mean": mean(values), "ci_low": low, "ci_high": high, "n_runs": len(values)}
    payload = {"summary": summary, "runs": records, "n_valid": len(valid), "n_inputs": len(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"aggregated valid={len(valid)} total={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
