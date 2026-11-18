#!/usr/bin/env python3
"""Refit the frozen Poker correction prior across calibration-root resamples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "poker-v2b-calibration-root-bootstrap-v1"
METHODS = ("pivot_kg", "uniform_v2", "ivr_v2", "lucb_v2", "proxy_only")
PRIMARY_BASELINES = ("uniform_v2", "ivr_v2")
R_FLOOR = 0.25
SHRINKAGE = 11.0 / 23.0
HORIZON = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _manifest(output: Path) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "content-manifest.json":
            data = path.read_bytes()
            files[str(path.relative_to(output))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    _write_json(output / "content-manifest.json", {"files": files})


def _isr(world: dict[str, Any], selected: int) -> float:
    labels = np.append(world["per_h"][HORIZON]["audit_label"], 0.0)
    return float(np.max(labels) - labels[selected])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_poker_sensitivity import _decision, _fit_spec, _load_code, _load_cohort

    code_dir = args.code.resolve()
    calibration_dir = args.calibration.resolve()
    test_dir = args.test.resolve()
    pivot_v2 = _load_code(code_dir)
    calibration = _load_cohort(calibration_dir, pivot_v2)
    test = _load_cohort(test_dir, pivot_v2)
    calibration_roots = [int(world["root"]) for world in calibration]
    test_roots = [int(world["root"]) for world in test]
    if len(calibration) != 12 or len(test) != 30:
        raise ValueError("expected the frozen 12-root calibration and 30-root test cohorts")
    if set(calibration_roots) & set(test_roots):
        raise ValueError("calibration and test roots overlap")

    frozen_prior_path = test_dir.parent / "analysis" / "posterior_v2_spec.json"
    frozen_prior = json.loads(frozen_prior_path.read_text())[str(HORIZON)]
    reference_spec = _fit_spec(calibration, HORIZON, pivot_v2, R_FLOOR, SHRINKAGE)
    for field in ("mu", "K_prior", "R"):
        if not np.allclose(
            np.asarray(getattr(reference_spec, field)),
            np.asarray(frozen_prior[field]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(f"full-cohort refit does not reproduce frozen prior field {field}")

    reference_decisions = {
        method: {
            int(world["root"]): _decision(
                world, reference_spec, HORIZON, method, pivot_v2, seed=int(world["root"])
            )
            for world in test
        }
        for method in METHODS
    }
    reference_scores = {
        method: {
            root: _isr(next(world for world in test if int(world["root"]) == root), decision["selected"])
            for root, decision in rows.items()
        }
        for method, rows in reference_decisions.items()
    }
    reference_contrasts = {
        method: float(np.mean([
            reference_scores[method][root] - reference_scores["pivot_kg"][root]
            for root in test_roots
        ]))
        for method in METHODS if method != "pivot_kg"
    }

    preregistration = Path(__file__).with_name(
        "POKER_CALIBRATION_BOOTSTRAP_PREREG_2026-11-09.md"
    )
    config = {
        "protocol_id": PROTOCOL_ID,
        "replicates": args.replicates,
        "rng_seed": args.seed,
        "calibration_sample_size": len(calibration),
        "calibration_roots": calibration_roots,
        "test_roots": test_roots,
        "horizon": HORIZON,
        "r_floor": R_FLOOR,
        "covariance_shrinkage": SHRINKAGE,
        "methods": list(METHODS),
        "primary_baselines": list(PRIMARY_BASELINES),
        "test_cohort_fixed": True,
        "physical_simulator_called": False,
        "preregistration_sha256": _sha256(preregistration),
    }
    config_hash = _digest(config)
    rng = np.random.default_rng(args.seed)
    sealed: list[dict[str, Any]] = []
    for replicate in range(args.replicates):
        indices = rng.integers(0, len(calibration), size=len(calibration))
        sampled = [calibration[int(index)] for index in indices]
        sampled_roots = [int(world["root"]) for world in sampled]
        spec = _fit_spec(sampled, HORIZON, pivot_v2, R_FLOOR, SHRINKAGE)
        prior_hash = _digest(spec.__dict__)
        for world in test:
            root = int(world["root"])
            for method in METHODS:
                decision = _decision(
                    world, spec, HORIZON, method, pivot_v2, seed=root
                )
                sealed.append({
                    "replicate": replicate,
                    "root": root,
                    "method": method,
                    "selected": int(decision["selected"]),
                    "queried": [int(index) for index in decision["queried"]],
                    "hf_queries": int(decision["hf_queries"]),
                    "charged_cost": float(decision["charged_cost"]),
                    "stop_reason": str(decision["stop_reason"]),
                    "calibration_model_sha256": prior_hash,
                    "sampled_calibration_roots": sampled_roots,
                })
    decision_blob = json.dumps(sealed, sort_keys=True, separators=(",", ":")).encode()
    (args.output / "decisions_sealed.json").write_bytes(decision_blob)
    decision_hash = hashlib.sha256(decision_blob).hexdigest()

    grouped: dict[tuple[int, str], dict[int, int]] = {}
    for row in sealed:
        grouped.setdefault((int(row["replicate"]), str(row["method"])), {})[
            int(row["root"])
        ] = int(row["selected"])
    scored: list[dict[str, Any]] = []
    by_replicate: dict[int, dict[str, dict[int, float]]] = {}
    for (replicate, method), selected_by_root in grouped.items():
        method_scores: dict[int, float] = {}
        for world in test:
            root = int(world["root"])
            value = _isr(world, selected_by_root[root])
            method_scores[root] = value
            scored.append({
                "replicate": replicate,
                "root": root,
                "method": method,
                "selected": selected_by_root[root],
                "isr": value,
            })
        by_replicate.setdefault(replicate, {})[method] = method_scores

    bootstrap_rows: list[dict[str, Any]] = []
    for replicate in range(args.replicates):
        method_scores = by_replicate[replicate]
        contrasts = {
            method: float(np.mean([
                method_scores[method][root] - method_scores["pivot_kg"][root]
                for root in test_roots
            ]))
            for method in METHODS if method != "pivot_kg"
        }
        flips = {
            method: float(np.mean([
                grouped[(replicate, method)][root]
                != reference_decisions[method][root]["selected"]
                for root in test_roots
            ]))
            for method in METHODS
        }
        bootstrap_rows.append({
            "replicate": replicate,
            "contrasts_baseline_minus_pivot": contrasts,
            "selection_flip_rate_vs_full_calibration": flips,
        })

    _write_json(args.output / "reference_full_calibration.json", {
        "calibration_model_sha256": _digest(reference_spec.__dict__),
        "decision_hash": _digest(reference_decisions),
        "contrasts_baseline_minus_pivot": reference_contrasts,
        "test_roots": test_roots,
    })
    _write_json(args.output / "scored_decisions.json", scored)
    _write_json(args.output / "bootstrap_rows.json", bootstrap_rows)
    _write_json(args.output / "config.json", {**config, "config_sha256": config_hash})

    aggregate: dict[str, Any] = {}
    for method in METHODS:
        if method == "pivot_kg":
            continue
        values = np.asarray([
            row["contrasts_baseline_minus_pivot"][method]
            for row in bootstrap_rows
        ], dtype=float)
        flips = np.asarray([
            row["selection_flip_rate_vs_full_calibration"]["pivot_kg"]
            for row in bootstrap_rows
        ], dtype=float)
        aggregate[method] = {
            "reference_full_calibration_contrast": reference_contrasts[method],
            "bootstrap_mean_contrast": float(values.mean()),
            "bootstrap_percentiles_2_5_50_97_5": [
                float(value) for value in np.percentile(values, [2.5, 50, 97.5])
            ],
            "positive_fraction": float(np.mean(values > 1e-12)),
            "negative_fraction": float(np.mean(values < -1e-12)),
            "zero_fraction": float(np.mean(np.abs(values) <= 1e-12)),
            "mean_pivot_selection_flip_rate": float(flips.mean()),
            "max_pivot_selection_flip_rate": float(flips.max()),
        }
    summary = {
        "status": "PASS",
        "analysis_kind": "offline_poker_calibration_root_bootstrap",
        "protocol_id": PROTOCOL_ID,
        "replicates": args.replicates,
        "calibration_roots": len(calibration),
        "test_roots": len(test),
        "decision_count": len(sealed),
        "decision_sha256": decision_hash,
        "default_prior_reproduced": True,
        "physical_simulator_called": False,
        "test_cohort_fixed": True,
        "aggregate": aggregate,
        "input_hashes": {
            "protocol": _sha256(test_dir / "protocol.json"),
            "pivot_v2": _sha256(code_dir / "pivot_v2.py"),
            "frozen_prior": _sha256(frozen_prior_path),
            "calibration_summaries": {
                str(root): _sha256(calibration_dir / f"seed_{root}" / "summary.json")
                for root in calibration_roots
            },
            "test_summaries": {
                str(root): _sha256(test_dir / f"seed_{root}" / "summary.json")
                for root in test_roots
            },
        },
        "limits": [
            "Calibration roots are resampled from the existing 12-root cohort; no new calibration cohort was collected.",
            "The bootstrap percentile range varies calibration fits conditional on the fixed 30 test roots.",
            "Test audit labels are used only after the full decision ledger has been sealed.",
            "LUCB-v2 is the source implementation's heuristic selector, not formal LUCB.",
        ],
    }
    _write_json(args.output / "summary.json", summary)
    _manifest(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
