#!/usr/bin/env python3
"""Replay frozen Leduc v2b decisions under a declared prior sensitivity grid.

This is an offline analysis.  It consumes the sealed calibration/test banks,
uses only proxy values and the per-root selection bank while making decisions,
then reads the audit labels for scoring after the decision ledger is sealed.
No OpenSpiel rollouts are executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boot(values: list[float], *, seed: int, draws: int = 10000) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.percentile(sample, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(x))}


def _load_code(code_dir: Path) -> Any:
    sys.path.insert(0, str(code_dir))
    import pivot_v2  # type: ignore[import-not-found]
    return pivot_v2


def _load_cohort(directory: Path, pivot_v2: Any) -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    for root_dir in sorted(directory.glob("seed_*")):
        summary_path = root_dir / "summary.json"
        if not summary_path.exists():
            continue
        status_path = root_dir / "status.json"
        summary = json.loads(summary_path.read_text())
        if status_path.exists():
            if json.loads(status_path.read_text()).get("status") != "complete":
                continue
        elif summary.get("status") != "complete":
            continue
        k = len(summary["proxy_deltas"])
        per_h: dict[int, dict[str, Any]] = {}
        for h in [1, 8]:
            selection = np.array([summary["selection_bank"][str(h)][str(i)] for i in range(k)], dtype=float)
            audit_a = np.array(
                [next(m["deployment_delta_audit"] for m in summary["mechanism_audit"]
                 if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "A")
                for i in range(k)
                ]
            )
            audit_b = np.array(
                [next(m["deployment_delta_audit"] for m in summary["mechanism_audit"]
                 if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "B")
                for i in range(k)
                ]
            )
            proxy_a = np.array(
                [next(m["proxy_delta_audit"] for m in summary["mechanism_audit"]
                 if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "A")
                for i in range(k)
                ]
            )
            proxy_b = np.array(
                [next(m["proxy_delta_audit"] for m in summary["mechanism_audit"]
                 if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "B")
                for i in range(k)
                ]
            )
            per_h[h] = {
                "selection": selection,
                "audit_label": 0.5 * (audit_a + audit_b),
                "audit_a": audit_a,
                "audit_b": audit_b,
                "proxy_a": proxy_a,
                "proxy_b": proxy_b,
            }
        worlds.append({
            "root": int(summary["seed"]),
            "proxy": np.array([summary["proxy_deltas"][str(i)] for i in range(k)], dtype=float),
            "per_h": per_h,
        })
    if not worlds:
        raise RuntimeError(f"no complete roots in {directory}")
    return worlds


def _fit_spec(train: list[dict[str, Any]], h: int, pivot_v2: Any, r_floor: float, shrinkage: float) -> Any:
    labels = np.array([w["per_h"][h]["audit_label"] for w in train], dtype=float)
    proxies = np.array([w["proxy"] for w in train], dtype=float)
    g = labels - proxies
    audit_diff = np.array([
        w["per_h"][h]["audit_a"] - w["per_h"][h]["audit_b"] for w in train
    ], dtype=float)
    selection = np.array([w["per_h"][h]["selection"] for w in train], dtype=float)
    a = np.array([w["per_h"][h]["audit_a"] for w in train], dtype=float)
    b = np.array([w["per_h"][h]["audit_b"] for w in train], dtype=float)
    s_minus = (selection - a) * (selection - b)
    k = g.shape[1]
    sample = np.cov(g, rowvar=False, ddof=1)
    omega = np.diag(np.mean(audit_diff ** 2, axis=0) / 4.0)
    latent = sample - omega
    target = np.diag(np.diag(latent))
    shrunk = (1.0 - shrinkage) * latent + shrinkage * target
    covariance = pivot_v2.psd_project(shrunk)
    covariance = covariance + covariance / len(train)
    raw_r = np.mean(s_minus, axis=0)
    # v2b uses candidate-index distance for this alpha grid, yielding singleton
    # pools and therefore the raw per-candidate instrument variance.
    observation = np.maximum(raw_r, r_floor)
    return pivot_v2.PosteriorV2Spec(
        mu=g.mean(axis=0).tolist(),
        K_prior=covariance.tolist(),
        R=observation.tolist(),
        n_roots=len(train),
        shrinkage_alpha=float(shrinkage),
        notes={"raw_R": raw_r.tolist(), "r_floor": float(r_floor)},
    )


def _decision(world: dict[str, Any], spec: Any, h: int, method: str, pivot_v2: Any, seed: int) -> dict[str, Any]:
    post = pivot_v2.make_posterior(spec, world["proxy"])
    selection = world["per_h"][h]["selection"]
    costs = np.full(len(world["proxy"]), 8192.0)
    result = pivot_v2.run_selector(
        post,
        world["proxy"],
        lambda j: float(selection[j]),
        method=method,
        budget=2,
        costs=costs,
        seed=seed,
    )
    return {
        "root": int(world["root"]),
        "adaptation": int(h),
        "method": method,
        "selected": int(result["selected"]),
        "queried": [int(v) for v in result["queried"]],
        "hf_queries": int(result["hf_queries"]),
        "charged_cost": float(result["charged_cost"]),
        "stop_reason": str(result["stop_reason"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists; choose a new directory: {args.output}")
    args.output.mkdir(parents=True)
    pivot_v2 = _load_code(args.code)
    calibration = _load_cohort(args.calibration, pivot_v2)
    test = _load_cohort(args.test, pivot_v2)
    if {w["root"] for w in calibration} & {w["root"] for w in test}:
        raise RuntimeError("calibration and test roots overlap")

    floors = [0.125, 0.25, 0.5, 1.0]
    shrinkages = [0.0, 0.25, 11.0 / 23.0, 0.75, 1.0]
    methods = ["pivot_kg", "uniform_v2", "ivr_v2", "lucb_v2", "no_hf_v2", "proxy_only"]
    frozen_prior_path = args.test.parent / "analysis" / "posterior_v2_spec.json"
    frozen_prior = json.loads(frozen_prior_path.read_text())
    reference_h = 8
    reference_spec = _fit_spec(calibration, reference_h, pivot_v2, 0.25, 11.0 / 23.0)
    for field in ("mu", "K_prior", "R"):
        if not np.allclose(np.asarray(getattr(reference_spec, field)), np.asarray(frozen_prior[str(reference_h)][field]), rtol=1e-10, atol=1e-12):
            raise RuntimeError(f"default sensitivity fit does not reproduce frozen prior field {field}")
    decisions: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    for r_floor in floors:
        for shrinkage in shrinkages:
            config_id = f"r{r_floor:g}_shrink{shrinkage:.8f}"
            specs = {h: _fit_spec(calibration, h, pivot_v2, r_floor, shrinkage) for h in (1, 8)}
            prior_record = {str(h): specs[h].__dict__ for h in (1, 8)}
            prior_hash = _sha256_bytes(json.dumps(prior_record, sort_keys=True).encode())
            configurations.append({
                "config_id": config_id,
                "r_floor": r_floor,
                "covariance_shrinkage": shrinkage,
                "prior_hash": prior_hash,
            })
            for world in test:
                for h in (1, 8):
                    for method in methods:
                        row = _decision(world, specs[h], h, method, pivot_v2, seed=int(world["root"]))
                        row.update({"config_id": config_id, "prior_hash": prior_hash})
                        decisions.append(row)

    decision_blob = json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode()
    (args.output / "decisions_sealed.json").write_bytes(decision_blob)
    decision_hash = _sha256_bytes(decision_blob)

    scored: list[dict[str, Any]] = []
    for decision in decisions:
        world = next(w for w in test if w["root"] == decision["root"])
        label = np.concatenate([world["per_h"][decision["adaptation"]]["audit_label"], [0.0]])
        selected = decision["selected"]
        gain = float(label[selected])
        best = float(np.max(label))
        scored.append({**decision, "gain": gain, "isr": best - gain})
    (args.output / "scored_decisions.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")

    config_rows: list[dict[str, Any]] = []
    for cfg in configurations:
        for h in (1, 8):
            by_method = {
                method: {r["root"]: r for r in scored if r["config_id"] == cfg["config_id"] and r["adaptation"] == h and r["method"] == method}
                for method in methods
            }
            pivot_by_root = by_method["pivot_kg"]
            for method in methods:
                method_by_root = by_method[method]
                differences = [float(method_by_root[root]["isr"] - pivot_by_root[root]["isr"]) for root in sorted(pivot_by_root)]
                flips = [int(method_by_root[root]["selected"] != pivot_by_root[root]["selected"]) for root in sorted(pivot_by_root)]
                config_rows.append({
                    **cfg,
                    "adaptation": h,
                    "method": method,
                    "mean_isr": float(np.mean([r["isr"] for r in method_by_root.values()])),
                    "baseline_minus_pivot_isr": _boot(differences, seed=20261120 + h + len(method), draws=args.bootstrap_draws),
                    "selection_disagreement_rate_vs_pivot": float(np.mean(flips)),
                    "mean_hf_queries": float(np.mean([r["hf_queries"] for r in method_by_root.values()])),
                    "mean_hf_cost": float(np.mean([r["charged_cost"] for r in method_by_root.values()])),
                })
    (args.output / "sensitivity_rows.json").write_text(json.dumps(config_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "analysis_kind": "offline_poker_prior_sensitivity",
        "physical_simulator_called": False,
        "protocol": "frozen Leduc v2b, primary cap 16384, two fixed 4096-hand paired queries",
        "calibration_roots": len(calibration),
        "test_roots": len(test),
        "grid": {"r_floor": floors, "covariance_shrinkage": shrinkages},
        "decision_hash": decision_hash,
        "input_hashes": {
            "protocol": _sha256_file(args.test / "protocol.json"),
            "pivot_v2": _sha256_file(args.code / "pivot_v2.py"),
            "frozen_prior": _sha256_file(frozen_prior_path),
        },
        "default_prior_reproduced": True,
        "notes": [
            "Decisions were sealed before audit labels were used for scoring.",
            "This is a sensitivity replay over saved sealed banks, not a new rollout.",
            "Uniform, IVR, LUCB-v2, no-HF calibrated, and Proxy-Only use the same frozen root set and terminal decision rule.",
            "No-HF calibrated and Proxy-Only are zero-query diagnostics; LUCB-v2 retains the source implementation's label.",
            "The default long-horizon prior matches the frozen specification before any sensitivity result is emitted.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
