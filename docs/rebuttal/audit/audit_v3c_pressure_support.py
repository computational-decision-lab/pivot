#!/usr/bin/env python3
"""Report calibration/test pressure support for the frozen v3c ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float], lower: float, upper: float) -> dict[str, Any]:
    outside = [value for value in values if value < lower or value > upper]
    return {
        "n_states": len(values),
        "min": min(values),
        "max": max(values),
        "outside_calibration_support": len(outside),
        "outside_fraction": len(outside) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads((args.input / "calibration_training_samples.json").read_text())
    ledger = json.loads((args.input / "trajectory_ledger.json").read_text())
    calibration = [float(row["features"][0]) for row in samples]
    lower, upper = min(calibration), max(calibration)
    by_regime: dict[str, list[float]] = defaultdict(list)
    by_regime_method: dict[str, list[float]] = defaultdict(list)
    for row in ledger:
        pressure = float(row["pressure_before"])
        by_regime[row["regime"]].append(pressure)
        by_regime_method[f"{row['regime']}:{row['method']}"].append(pressure)
    report = {
        "diagnostic_id": "V3C-PRESSURE-SUPPORT-2026-11-10",
        "input_hashes": {
            "calibration_training_samples": sha256_file(args.input / "calibration_training_samples.json"),
            "ledger": sha256_file(args.input / "trajectory_ledger.json"),
        },
        "calibration_support": {"min": lower, "max": upper, "n_candidate_rows": len(calibration)},
        "test_states_by_regime": {
            regime: summarize(values, lower, upper)
            for regime, values in sorted(by_regime.items())
        },
        "test_states_by_regime_method": {
            key: summarize(values, lower, upper)
            for key, values in sorted(by_regime_method.items())
        },
        "interpretation": "Pressure values outside the calibration min/max are extrapolative relative to observed calibration support; this is a descriptive coverage diagnostic, not a test exclusion rule.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
