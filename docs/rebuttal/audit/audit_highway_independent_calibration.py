#!/usr/bin/env python3
"""Replay sealed independent-cohort decisions and audit physical provenance."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import fit_calibration_model, GaussianPosterior
    from experiments.revision.highway_b4_redesign import exact_kg_next, uniform_expected_selection
    from experiments.revision.evaluation_journal import audit_evaluations

    runner_path = Path(__file__).with_name("run_highway_independent_physical.py")
    spec = importlib.util.spec_from_file_location("independent_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner._verify_source(source)["valid"]
    roots = [run / "smoke", run / "calibration", run / "aggregate"] + [run / f"test-shard{i}" for i in range(4)]
    checked_files = 0
    for root in roots:
        for name, metadata in read(root / "content-manifest.json")["files"].items():
            assert sha(root / name) == metadata["sha256"], str(root / name)
            checked_files += 1
    checks = {"source_and_manifests": True}
    calibration = run / "calibration"
    cp, ct = defaultdict(list), defaultdict(dict)
    for row in read(calibration / "calibration-proxy-rows.json")["rows"]:
        cp[row["seed"]].append({k: v for k, v in row.items() if k != "seed"})
    for row in read(calibration / "calibration-truth.json")["rows"]:
        ct[row["seed"]][row["candidate_id"]] = row["paired_delta"]
    assert set(cp) == set(ct) == set(runner.CALIBRATION_SEEDS)
    model = fit_calibration_model(cp, ct, covariance_shrinkage=0.5)
    assert model.to_record() == read(calibration / "calibration-model.json")
    assert digest(model.to_record()) == read(calibration / "summary.json")["calibration_model_sha256"]
    new_seeds = set(runner.CALIBRATION_SEEDS) | set(runner.TEST_SEEDS)
    assert len(new_seeds) == 160
    assert not new_seeds & (set(runner.SMOKE_CALIBRATION_SEEDS) | set(runner.SMOKE_TEST_SEEDS))
    earlier = run.parent / "20261109-highway-matched-physical-v2"
    old_calibration = read(earlier / "calibration-complete/summary.json")["calibration_seeds"]
    old_test = [r["seed"] for r in read(earlier / "physical-aggregate-v2/paired-root-results.json")["rows"]]
    assert not new_seeds & (set(old_calibration) | set(old_test))
    checks["refit_and_disjoint_seed_cohorts"] = True

    aggregate = run / "aggregate"
    summary = read(aggregate / "summary.json")
    proxies, truths, queries, subsets = defaultdict(list), defaultdict(dict), defaultdict(list), defaultdict(list)
    for row in read(aggregate / "test-proxy-rows.json")["rows"]:
        proxies[row["seed"]].append(row)
    for row in read(aggregate / "truth-audit.json")["rows"]:
        truths[row["seed"]][row["candidate_id"]] = row["paired_delta"]
    for row in read(aggregate / "query-ledger.json")["rows"]:
        if row["logical_hf_query"]:
            queries[row["seed"], row["method"]].append(row)
    decisions = {(r["seed"], r["method"]): r for r in read(aggregate / "decision-rows.json")["rows"]}
    promotions = {(r["seed"], r["method"]): r for r in read(aggregate / "promotion-results.json")["rows"]}
    for row in read(aggregate / "uniform-subset-audit.json")["rows"]:
        subsets[row["seed"]].append(row)
    assert set(proxies) == set(truths) == set(runner.TEST_SEEDS)
    assert len(decisions) == 480 and len(promotions) == 600
    query_count = 0
    for seed in runner.TEST_SEEDS:
        rows = proxies[seed]
        truth = truths[seed]
        assert len(rows) == 8 and len(truth) == 9
        for method in runner.METHODS:
            posterior = GaussianPosterior(rows, model)
            observed = queries[seed, method]
            assert len(observed) == 4
            top = sorted(rows, key=lambda r: (-r["delta_proxy"], r["candidate_index"]))
            uniform = runner._uniform_order(seed, model.candidate_ids)
            for i, query in enumerate(observed):
                if method == "PIVOT-KG":
                    expected, score = exact_kg_next(posterior)
                    close(score, query["acquisition_score"])
                elif method == "IVR-Trace":
                    expected, score = runner._ivr_trace_next(posterior)
                    close(score, query["acquisition_score"])
                elif method == "Top-Proxy":
                    expected = top[i]["candidate_id"]
                else:
                    expected = uniform[i]
                assert query["candidate_id"] == expected and query["analysis_truth_reused"] is False
                close(query["posterior_mean_before"], posterior.value(expected))
                close(query["posterior_sd_before"], posterior.uncertainty(expected))
                close(query["paired_delta"], truth[expected])
                posterior.observe(expected, query["paired_delta"])
                close(query["posterior_mean_after"], posterior.value(expected))
                query_count += 1
            decision = decisions[seed, method]
            assert decision["queries"] == [q["candidate_id"] for q in observed]
            selected = posterior.select()
            assert selected == decision["selected_candidate"]
            close(decision["selected_posterior_mean"], posterior.value(selected))
            close(promotions[seed, method]["ISR"], max(truth.values()) - truth[selected])
        expected_uniform = uniform_expected_selection(rows, model, truth, 4)
        assert len(subsets[seed]) == 70
        for actual, expected in zip(subsets[seed], expected_uniform["rows"], strict=True):
            assert actual["queried_candidates"] == expected["queried_candidates"]
            assert actual["selected_candidate"] == expected["selected_candidate"]
            close(actual["ISR"], expected["ISR"])
        close(promotions[seed, "Uniform-Exact-Expectation-Audit"]["ISR"], expected_uniform["mean_ISR"])
    checks["all_query_acquisitions_updates_terminal_and_isr"] = True
    checks["all_8400_uniform_subsets"] = True

    journals = {}
    for name in ["smoke-runstate", "calibration-runstate"] + [f"test-runstate-shard{i}" for i in range(4)]:
        journal_root = run / name / "evaluation-journal"
        audit = audit_evaluations(journal_root)
        assert audit["failed_evaluations"] == audit["unresolved_evaluations"] == audit["unmeasured_evaluations"] == 0
        assert audit["cost_complete"]
        journals[name] = {k: v for k, v in audit.items() if k != "files"}
        if name.startswith("test-"):
            starts, ends = defaultdict(list), defaultdict(list)
            for path in journal_root.glob("*/*.intent.json"):
                row = read(path)
                if row["phase"] == "truth_audit":
                    starts[row["seed"]].append(row["started_at"])
            for path in journal_root.glob("*/*.result.json"):
                row = read(path)
                if row["phase"] == "hf_query":
                    ends[row["seed"]].append(row["completed_at"])
            assert len(starts) == len(ends) == 30
            for seed in starts:
                assert max(ends[seed]) < min(starts[seed])
            index = int(name[-1])
            shard = run / f"test-shard{index}"
            identity = read(shard / "identity.json")
            assert identity["runner"]["sha256"] == sha(runner_path)
            assert identity["config"]["study_config_hash"] == summary["study_config_hash"]
            assert read(shard / "calibration-model.json") == model.to_record()
    checks["journal_complete_and_truth_after_queries"] = True
    checks["shards_share_frozen_model_and_code"] = True
    for method, contrast in summary["contrasts"].items():
        values = np.array([promotions[s, method]["ISR"] - promotions[s, "PIVOT-KG"]["ISR"] for s in runner.TEST_SEEDS])
        close(np.mean(values), contrast["mean"])
        rng = np.random.default_rng(20261109)
        close(np.percentile(rng.choice(values, (4000, 120), replace=True).mean(axis=1), [2.5, 97.5]), contrast["ci95_root_bootstrap"])
    checks["all_contrasts_and_intervals_recomputed"] = True
    report = {
        "status": "PASS", "checks": checks, "verified_manifest_files": checked_files,
        "query_decisions_replayed": query_count, "terminal_decisions_replayed": len(decisions),
        "uniform_subsets_replayed": 8400, "journals": journals,
        "calibration_plus_test_evaluations": sum(j["completed_evaluations"] for n, j in journals.items() if n != "smoke-runstate"),
        "calibration_plus_test_simulator_calls": sum(j["known_simulator_calls"] for n, j in journals.items() if n != "smoke-runstate"),
        "summary_sha256": sha(aggregate / "summary.json"), "runner_sha256": sha(runner_path),
        "note": "Replay of sealed decisions, not new simulator runs. Calibration and test cohorts both change; intervals condition on the new fitted model."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "journals"}, indent=2))


if __name__ == "__main__":
    main()
