#!/usr/bin/env python3
"""Audit and aggregate disjoint physical HighwayEnv selector shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_METHODS = ("PIVOT-KG", "IVR-Trace", "Top-Proxy", "Uniform-Random")
EXPECTED_PROTOCOL_ID = "highway-independent-calibration-v1"
EXPECTED_TEST_SEEDS = tuple(4_300_003 + 59 * index for index in range(120))
EXPECTED_CALIBRATION_SEEDS = tuple(4_100_003 + 53 * index for index in range(40))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _verify_manifest(root: Path) -> None:
    manifest = json.loads((root / "content-manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")


def _bootstrap(values: list[float], seed: int = 20261109, draws: int = 4000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-runstates", nargs="+", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    test_runstates = [path.resolve() for path in args.test_runstates]
    calibration = args.calibration.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    _verify_manifest(calibration)
    calibration_summary = json.loads((calibration / "summary.json").read_text())
    if calibration_summary.get("status") != "CALIBRATION_READY":
        raise ValueError("calibration cohort is not sealed and complete")
    if calibration_summary.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("calibration cohort has the wrong independent protocol ID")
    if tuple(calibration_summary.get("calibration_seeds", [])) != EXPECTED_CALIBRATION_SEEDS:
        raise ValueError("calibration cohort differs from the frozen 40-root cohort")

    shard_identities: set[str] = set()
    proxy_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shard_seed_sets: list[set[int]] = []
    for shard_arg in args.shards:
        shard = shard_arg.resolve()
        _verify_manifest(shard)
        identity = json.loads((shard / "identity.json").read_text())
        if identity["config"].get("protocol_id") != EXPECTED_PROTOCOL_ID:
            raise ValueError("test shard has the wrong independent protocol ID")
        shard_identities.add(identity["config"]["study_config_hash"])
        if identity["source"]["historical_source_hash"] != calibration_summary["source_hash"]:
            raise ValueError("test shard and calibration source hashes differ")
        summary = json.loads((shard / "summary.json").read_text())
        if summary.get("protocol_id") != EXPECTED_PROTOCOL_ID:
            raise ValueError("test shard summary has the wrong independent protocol ID")
        failures.extend({"shard": str(shard), **item} for item in summary.get("failures", []))
        proxy = json.loads((shard / "test-proxy-rows.json").read_text())["rows"]
        queries = json.loads((shard / "query-ledger.json").read_text())["rows"]
        decisions = json.loads((shard / "decision-rows.json").read_text())["rows"]
        promotions = json.loads((shard / "promotion-results.json").read_text())["rows"]
        truths = json.loads((shard / "truth-audit.json").read_text())["rows"]
        subsets = json.loads((shard / "uniform-subset-audit.json").read_text())["rows"]
        proxy_rows.extend(proxy)
        query_rows.extend(queries)
        decision_rows.extend(decisions)
        promotion_rows.extend(promotions)
        truth_rows.extend(truths)
        subset_rows.extend(subsets)
        shard_seed_sets.append({int(row["seed"]) for row in decisions})

    if len(shard_identities) != 1:
        raise ValueError("test shards do not share the same full-study config hash")
    if any(shard_seed_sets[i] & shard_seed_sets[j] for i in range(len(shard_seed_sets)) for j in range(i)):
        raise ValueError("test shard seed sets overlap")
    expected = set(EXPECTED_TEST_SEEDS)
    observed = set().union(*shard_seed_sets)
    if observed != expected:
        raise ValueError(
            f"test seed coverage mismatch; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if failures:
        raise ValueError(f"one or more test shards report failures: {failures}")

    proxies_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in proxy_rows:
        proxies_by_seed.setdefault(int(row["seed"]), []).append(row)
    decisions_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in decision_rows:
        seed = int(row["seed"])
        method = str(row["method"])
        if method in decisions_by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate method decision at seed={seed}, method={method}")
        decisions_by_seed[seed][method] = row
    truths_by_seed: dict[int, dict[str, float]] = {}
    for row in truth_rows:
        seed = int(row["seed"])
        truths_by_seed.setdefault(seed, {})[str(row["candidate_id"])] = float(row["paired_delta"])
    logical_query_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in query_rows:
        if row.get("logical_hf_query"):
            logical_query_groups.setdefault((int(row["seed"]), str(row["method"])), []).append(row)

    for seed in EXPECTED_TEST_SEEDS:
        if len(proxies_by_seed.get(seed, [])) != 8:
            raise ValueError(f"expected eight proxy candidates at seed {seed}")
        if set(decisions_by_seed.get(seed, {})) != set(EXPECTED_METHODS):
            raise ValueError(f"method decision set mismatch at seed {seed}")
        truth = truths_by_seed.get(seed, {})
        if set(truth) != {"incumbent", "early_slower", "early_lane_left", "early_lane_right", "early_faster",
                          "late_slower", "late_lane_left", "late_lane_right", "late_faster"}:
            raise ValueError(f"truth audit candidate set mismatch at seed {seed}")
        best = max([0.0, *[truth[candidate] for candidate in truth if candidate != "incumbent"]])
        for method in EXPECTED_METHODS:
            decision = decisions_by_seed[seed][method]
            if len(decision["queries"]) != 4 or len(set(decision["queries"])) != 4:
                raise ValueError(f"selector query count/uniqueness mismatch at seed {seed}, method={method}")
            queries = logical_query_groups.get((seed, method), [])
            if len(queries) != 4:
                raise ValueError(f"physical HF query ledger mismatch at seed {seed}, method={method}")
            if [row["candidate_id"] for row in queries] != decision["queries"]:
                raise ValueError(f"query ledger and sealed decision order mismatch at seed {seed}, method={method}")
            for row in queries:
                if row.get("analysis_truth_reused") is not False:
                    raise ValueError("decision query reused analysis truth")
                if abs(float(row["paired_delta"]) - truth[row["candidate_id"]]) > 1e-12:
                    raise ValueError(f"query/audit mismatch at seed {seed}, method={method}")
            selected = str(decision["selected_candidate"])
            selected_value = 0.0 if selected == "incumbent" else truth[selected]
            stored = next(row for row in promotion_rows if row["seed"] == seed and row["method"] == method)
            if not np.isclose(float(stored["ISR"]), best - selected_value, rtol=0.0, atol=1e-12):
                raise ValueError(f"ISR recomputation mismatch at seed {seed}, method={method}")

    by_method = {
        method: {int(row["seed"]): float(row["ISR"]) for row in promotion_rows if row["method"] == method}
        for method in EXPECTED_METHODS
    }
    if any(len(rows) != 120 for rows in by_method.values()):
        raise ValueError("each physical method must have one outcome on every test root")
    contrasts: dict[str, Any] = {}
    for method in ("IVR-Trace", "Top-Proxy", "Uniform-Random"):
        values = [by_method[method][seed] - by_method["PIVOT-KG"][seed] for seed in EXPECTED_TEST_SEEDS]
        contrasts[method] = {
            "estimand": f"ISR({method}, B=4) - ISR(PIVOT-KG, B=4)",
            "n_root_pairs": len(values),
            "mean": float(np.mean(values)),
            "ci95_root_bootstrap": _bootstrap(values),
            "positive_favors": "PIVOT-KG",
            "positive_pairs": sum(value > 1e-12 for value in values),
            "negative_pairs": sum(value < -1e-12 for value in values),
            "zero_pairs": sum(abs(value) <= 1e-12 for value in values),
            "p_value": None,
        }
    exact_uniform = {
        int(row["seed"]): float(row["ISR"])
        for row in promotion_rows if row["method"] == "Uniform-Exact-Expectation-Audit"
    }
    if len(exact_uniform) != 120:
        raise ValueError("post-decision exact Uniform audit is incomplete")
    exact_values = [exact_uniform[seed] - by_method["PIVOT-KG"][seed] for seed in EXPECTED_TEST_SEEDS]
    contrasts["Uniform-Exact-Expectation-Audit"] = {
        "estimand": "post-decision exact Uniform subset expectation ISR - PIVOT-KG ISR",
        "n_root_pairs": len(exact_values),
        "mean": float(np.mean(exact_values)),
        "ci95_root_bootstrap": _bootstrap(exact_values),
        "positive_favors": "PIVOT-KG",
        "positive_pairs": sum(value > 1e-12 for value in exact_values),
        "negative_pairs": sum(value < -1e-12 for value in exact_values),
        "zero_pairs": sum(abs(value) <= 1e-12 for value in exact_values),
        "p_value": None,
        "audit_only": True,
        "physical_selector": False,
    }

    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.evaluation_journal import audit_evaluations

    shard_audits = [audit_evaluations(root / "evaluation-journal") for root in test_runstates]
    journal_audit = {
        "attempts": sum(item["attempts"] for item in shard_audits),
        "completed_evaluations": sum(item["completed_evaluations"] for item in shard_audits),
        "failed_evaluations": sum(item["failed_evaluations"] for item in shard_audits),
        "unresolved_evaluations": sum(item["unresolved_evaluations"] for item in shard_audits),
        "unmeasured_evaluations": sum(item["unmeasured_evaluations"] for item in shard_audits),
        "known_environment_steps": sum(item["known_environment_steps"] for item in shard_audits),
        "known_simulator_calls": sum(item["known_simulator_calls"] for item in shard_audits),
        "known_compute_cost": sum(item["known_compute_cost"] for item in shard_audits),
        "recorded_wall_time": sum(item["recorded_wall_time"] for item in shard_audits),
        "cost_complete": all(item["cost_complete"] for item in shard_audits),
        "files": {
            f"shard-{index}/{name}": metadata
            for index, item in enumerate(shard_audits)
            for name, metadata in item["files"].items()
        },
    }
    if journal_audit["failed_evaluations"] or journal_audit["unresolved_evaluations"]:
        raise ValueError("physical evaluation journal has failures or unresolved calls")
    summary = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "status": "DEV_INDEPENDENT_PHYSICAL_REPLICATION",
        "calibration_roots": 40,
        "test_roots": 120,
        "candidate_count": 8,
        "budget": 4,
        "methods": list(EXPECTED_METHODS),
        "primary_method": "IVR-Trace",
        "primary_contrast": contrasts["IVR-Trace"],
        "contrasts": contrasts,
        "failures": [],
        "physical_evaluation_journal": journal_audit,
        "calibration_model_sha256": calibration_summary["calibration_model_sha256"],
        "study_config_hash": next(iter(shard_identities)),
        "promotion_status": "READY_FOR_REVIEW",
        "scope": "single-decision physical HighwayEnv selector comparison; no multi-round claim",
    }
    atomic = lambda name, payload: (output / name).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    atomic("summary.json", summary)
    atomic("paired-root-results.json", {
        "rows": [{
            "seed": seed,
            **{f"ISR_{method}": by_method[method][seed] for method in EXPECTED_METHODS},
            "ISR_Uniform_Exact_Audit": exact_uniform[seed],
            **{f"contrast_{method}_minus_PIVOT": by_method[method][seed] - by_method["PIVOT-KG"][seed]
               for method in EXPECTED_METHODS[1:]},
            "contrast_Uniform-Exact-Expectation-Audit_minus_PIVOT": exact_uniform[seed] - by_method["PIVOT-KG"][seed],
        } for seed in EXPECTED_TEST_SEEDS]
    })
    atomic("decision-rows.json", {"rows": decision_rows})
    atomic("query-ledger.json", {"rows": query_rows})
    atomic("truth-audit.json", {"rows": truth_rows})
    atomic("test-proxy-rows.json", {"rows": proxy_rows})
    atomic("promotion-results.json", {"rows": promotion_rows})
    atomic("uniform-subset-audit.json", {"rows": subset_rows})
    atomic("cost-ledger.json", {"physical_evaluation_journal": journal_audit})
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "content-manifest.json":
            data = path.read_bytes()
            files[str(path.relative_to(output))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    atomic("content-manifest.json", {"files": files})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
