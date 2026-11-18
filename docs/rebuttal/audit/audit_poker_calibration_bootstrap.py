#!/usr/bin/env python3
"""Independently verify the Poker calibration-root bootstrap artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    calibration_dir = args.calibration.resolve()
    test_dir = args.test.resolve()
    manifest = _read(result / "content-manifest.json")["files"]
    for name, metadata in manifest.items():
        path = result / name
        if not path.is_file() or _hash(path) != metadata["sha256"]:
            raise ValueError(f"content-manifest mismatch: {name}")

    config = _read(result / "config.json")
    summary = _read(result / "summary.json")
    decisions = _read(result / "decisions_sealed.json")
    scored = _read(result / "scored_decisions.json")
    bootstrap_rows = _read(result / "bootstrap_rows.json")
    reference = _read(result / "reference_full_calibration.json")
    if summary["status"] != "PASS" or config["protocol_id"] != "poker-v2b-calibration-root-bootstrap-v1":
        raise ValueError("unexpected experiment status or protocol")
    if summary["default_prior_reproduced"] is not True or summary["physical_simulator_called"] is not False:
        raise ValueError("default-prior or offline-only assertion failed")

    calibration_roots = sorted(
        int(path.parent.name.removeprefix("seed_"))
        for path in calibration_dir.glob("seed_*/summary.json")
    )
    test_roots = sorted(
        int(path.parent.name.removeprefix("seed_"))
        for path in test_dir.glob("seed_*/summary.json")
    )
    if len(calibration_roots) != 12 or len(test_roots) != 30 or set(calibration_roots) & set(test_roots):
        raise ValueError("calibration/test root inventory is invalid")
    if config["calibration_roots"] != calibration_roots or config["test_roots"] != test_roots:
        raise ValueError("run configuration root inventory differs from input banks")

    methods = config["methods"]
    replicates = int(config["replicates"])
    if len(decisions) != replicates * len(test_roots) * len(methods):
        raise ValueError("sealed decision count is incomplete")
    decision_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    forbidden = {"isr", "gain", "audit_label", "truth", "paired_delta"}
    for row in decisions:
        if forbidden.intersection(row):
            raise ValueError("sealed decision row contains an audit label")
        key = (int(row["replicate"]), int(row["root"]), str(row["method"]))
        if key in decision_map:
            raise ValueError(f"duplicate sealed decision: {key}")
        if len(row["sampled_calibration_roots"]) != len(calibration_roots):
            raise ValueError("bootstrap draw has wrong sample size")
        if not set(row["sampled_calibration_roots"]) <= set(calibration_roots):
            raise ValueError("bootstrap draw contains an unknown calibration root")
        decision_map[key] = row

    sys_path = str(Path(__file__).resolve().parent)
    import sys

    sys.path.insert(0, sys_path)
    from run_poker_sensitivity import _load_code, _load_cohort

    pivot_v2 = _load_code(Path(config["input_paths"]["code"])) if "input_paths" in config else None
    if pivot_v2 is None:
        code_dir = calibration_dir / "../code"
        pivot_v2 = _load_code(code_dir.resolve())
    test = _load_cohort(test_dir, pivot_v2)
    worlds = {int(world["root"]): world for world in test}
    scored_map: dict[tuple[int, int, str], float] = {}
    for row in scored:
        key = (int(row["replicate"]), int(row["root"]), str(row["method"]))
        if key in scored_map:
            raise ValueError(f"duplicate scored decision: {key}")
        decision = decision_map[key]
        labels = np.append(worlds[key[1]]["per_h"][8]["audit_label"], 0.0)
        expected_isr = float(np.max(labels) - labels[int(decision["selected"])])
        if not _close(float(row["isr"]), expected_isr) or int(row["selected"]) != int(decision["selected"]):
            raise ValueError(f"scored decision mismatch: {key}")
        scored_map[key] = expected_isr
    if len(scored_map) != len(decision_map):
        raise ValueError("scored decision coverage differs from sealed decision coverage")

    contrasts: dict[str, list[float]] = {
        method: [] for method in methods if method != "pivot_kg"
    }
    flips: dict[str, list[float]] = {method: [] for method in methods}
    for replicate in range(replicates):
        for method in contrasts:
            contrasts[method].append(float(np.mean([
                scored_map[(replicate, root, method)]
                - scored_map[(replicate, root, "pivot_kg")]
                for root in test_roots
            ])))
        for method in methods:
            flips[method].append(float(np.mean([
                decision_map[(replicate, root, method)]["selected"]
                != decision_map[(replicate, root, "pivot_kg")]["selected"]
                for root in test_roots
            ])))

    aggregate = summary["aggregate"]
    for method, values in contrasts.items():
        stats = aggregate[method]
        expected_quantiles = np.percentile(values, [2.5, 50, 97.5])
        saved_quantiles = stats["bootstrap_percentiles_2_5_50_97_5"]
        if not _close(float(np.mean(values)), float(stats["bootstrap_mean_contrast"])):
            raise ValueError(f"bootstrap mean mismatch for {method}")
        if any(not _close(float(left), float(right)) for left, right in zip(expected_quantiles, saved_quantiles)):
            raise ValueError(f"bootstrap quantile mismatch for {method}")
        if not _close(float(np.mean(np.asarray(values) > 1e-12)), float(stats["positive_fraction"])):
            raise ValueError(f"bootstrap sign rate mismatch for {method}")

    # The default full-cohort decisions must match the already audited sensitivity replay.
    sensitivity_rows = _read(args.sensitivity.resolve() / "sensitivity_rows.json")
    default_id = "r0.25_shrink0.47826087"
    for method in ("uniform_v2", "ivr_v2"):
        saved = next(row for row in sensitivity_rows
                     if row["config_id"] == default_id and row["adaptation"] == 8 and row["method"] == method)
        if not _close(float(reference["contrasts_baseline_minus_pivot"][method]),
                      float(saved["baseline_minus_pivot_isr"]["mean"])):
            raise ValueError(f"reference contrast mismatch with prior sensitivity for {method}")

    report = {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "replicates": replicates,
        "calibration_roots": len(calibration_roots),
        "fixed_test_roots": len(test_roots),
        "sealed_decisions": len(decisions),
        "scored_decisions": len(scored),
        "content_manifest_files_checked": len(manifest),
        "checks": {
            "disjoint_root_sets": True,
            "all_bootstrap_samples_valid": True,
            "decisions_sealed_without_audit_labels": True,
            "all_scores_recomputed_from_saved_audits": True,
            "bootstrap_aggregates_recomputed": True,
            "full_fit_reproduces_prior_sensitivity": True,
        },
        "aggregate": aggregate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
