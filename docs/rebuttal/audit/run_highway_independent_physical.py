#!/usr/bin/env python3
"""Run a fresh-root physical HighwayEnv comparison of matched selectors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROTOCOL_ID = "highway-independent-calibration-v1"
CALIBRATION_SEEDS = tuple(4_100_003 + 53 * index for index in range(40))
TEST_SEEDS = tuple(4_300_003 + 59 * index for index in range(120))
SMOKE_CALIBRATION_SEEDS = tuple(4_090_003 + 61 * index for index in range(2))
SMOKE_TEST_SEEDS = tuple(4_090_503 + 67 * index for index in range(2))
METHODS = ("PIVOT-KG", "IVR-Trace", "Top-Proxy", "Uniform-Random")
BUDGET = 4


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_imports(source: Path) -> None:
    sys.path[:0] = [str(source), str(source / "src")]


def _verify_source(source: Path) -> dict[str, Any]:
    manifest_path = source.parent / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    checked = 0
    for name, metadata in expected.items():
        path = source / name
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"frozen source mismatch: {name}")
        checked += 1
    identity = json.loads((source.parent / "evidence" / "identity.json").read_text())
    source_names = [
        "scripts/run_highway_b4_redesign.py",
        "experiments/revision/highway_b4_redesign.py",
        "experiments/revision/highway_calibration.py",
        "src/pivot/environments/revision_highway.py",
        "requirements/revision-highway.txt",
    ]
    source_hash = _digest({name: _sha256(source / name) for name in source_names})
    if source_hash != identity["source"]["source_hash"]:
        raise ValueError("frozen source key-file hash differs from archived run identity")
    return {
        "historical_git_commit": identity["source"]["git_commit"],
        "historical_source_hash": source_hash,
        "source_manifest_sha256": _sha256(manifest_path),
        "checked_files": checked,
        "valid": True,
    }


def _dependencies() -> dict[str, Any]:
    distributions = ("highway-env", "gymnasium", "numpy", "PyYAML")
    installed: dict[str, str] = {}
    for name in distributions:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = "unavailable"
    return {"python": platform.python_version(), "distributions": installed}


def _bootstrap_ci(values: Sequence[float], *, seed: int, draws: int = 4000) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(item) for item in np.percentile(samples, [2.5, 97.5])]


def _ivr_trace_next(posterior: Any) -> tuple[str, float]:
    available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
    if not available:
        raise ValueError("no candidates remain to query")
    scores: dict[str, float] = {}
    for candidate in available:
        index = posterior._index[candidate]
        denominator = float(
            posterior.covariance[index, index] + posterior.observation_variance[index]
        )
        scores[candidate] = (
            float(np.dot(posterior.covariance[:, index], posterior.covariance[:, index]) / denominator)
            if denominator > 0.0
            else 0.0
        )
    chosen = max(available, key=lambda value: (scores[value], -posterior._index[value]))
    return chosen, scores[chosen]


def _uniform_order(seed: int, candidate_ids: Sequence[str]) -> list[str]:
    rng = np.random.default_rng(seed ^ 0x5A17C9E3)
    order = rng.permutation(len(candidate_ids))
    return [str(candidate_ids[int(index)]) for index in order]


def _write_manifest(output: Path) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "content-manifest.json":
            data = path.read_bytes()
            files[str(path.relative_to(output))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    from pivot.cloud.archive import atomic_json

    atomic_json(output / "content-manifest.json", {"files": files})


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _configure_imports(source)

    from experiments.revision.evaluation_journal import EvaluationJournal, audit_evaluations
    from experiments.revision.highway_b4_redesign import (
        discovery_panel_b4,
        enumerate_subsets,
        exact_kg_next,
        uniform_expected_selection,
        validate_panel,
    )
    from experiments.revision.highway_calibration import (
        CalibrationModel,
        GaussianPosterior,
        fit_calibration_model,
    )
    from experiments.revision.seed_registry import SeedRegistry
    from pivot.cloud.archive import atomic_json
    from pivot.core.policy import Policy
    from pivot.environments.revision_highway import HighwayPhysicalConfig, HighwayPhysicalWorld
    from pivot.runner.checkpoint import CheckpointStore

    source_identity = _verify_source(source)
    dependencies = _dependencies()
    expected_dependencies = {
        "highway-env": "1.12.1",
        "gymnasium": "1.3.0",
        "numpy": "2.5.3",
        "PyYAML": "6.0.3",
    }
    if dependencies["distributions"] != expected_dependencies:
        raise RuntimeError(
            "runtime dependencies differ from the frozen protocol: "
            f"observed={dependencies['distributions']}, expected={expected_dependencies}"
        )
    full_calibration_seeds = SMOKE_CALIBRATION_SEEDS if args.smoke else CALIBRATION_SEEDS
    calibration_start = 0 if args.smoke else args.calibration_start
    calibration_stop = len(full_calibration_seeds) if args.smoke else args.calibration_stop
    if not 0 <= calibration_start < calibration_stop <= len(full_calibration_seeds):
        raise ValueError("calibration index range must select a non-empty frozen cohort subset")
    calibration_seeds = full_calibration_seeds[calibration_start:calibration_stop]
    full_test_seeds = SMOKE_TEST_SEEDS if args.smoke else TEST_SEEDS
    test_start = 0 if args.smoke else args.test_start
    test_stop = len(full_test_seeds) if args.smoke else args.test_stop
    if args.calibration_only and not args.smoke:
        test_start, test_stop = 0, 0
    if not 0 <= test_start <= test_stop <= len(full_test_seeds) or (
        test_start == test_stop and not args.calibration_only
    ):
        raise ValueError("test index range must select a non-empty subset of the frozen cohort")
    test_seeds = full_test_seeds[test_start:test_stop]
    run_root = (args.run_root or output).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    preregistration_path = Path(__file__).with_name(
        "HIGHWAY_INDEPENDENT_PHYSICAL_PREREGISTRATION_2026-11-09.md"
    )
    preregistration_hash = _sha256(preregistration_path)
    config = {
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": preregistration_hash,
        "status_scope": "smoke_only" if args.smoke else "independent_physical_dev_replication",
        "calibration_seeds": list(calibration_seeds),
        "calibration_index_range": [calibration_start, calibration_stop],
        "test_seeds": list(test_seeds),
        "test_index_range": [test_start, test_stop],
        "candidate_actions": ["SLOWER", "LANE_LEFT", "LANE_RIGHT", "FASTER"],
        "candidate_phases": ["early", "late"],
        "candidate_count": 8,
        "methods": list(METHODS),
        "primary_method_contrast": "ISR(IVR-Trace) - ISR(PIVOT-KG)",
        "primary_budget": BUDGET,
        "secondary_methods": ["Top-Proxy", "Uniform-Random"],
        "uniform_exact_expectation": "audit_only_after_all_decisions",
        "formal_lucb": "not_run; initialization requires 8 queries while B=4",
        "highway_env_version": "1.12.1",
        "runtime_dependencies": expected_dependencies,
        "horizon": 20,
        "lanes_count": 4,
        "initial_lane_id": 1,
        "vehicles_count": 20,
        "vehicles_density": 1.0,
        "simulation_frequency": 5,
        "action_hold_steps": 1,
        "calibration_covariance_shrinkage": 0.5,
        "paired_evaluation": "actor candidate minus actor incumbent at same seed",
        "physical_observation_bank": "one deterministic actor pair per seed/candidate; selector logical queries may reuse sealed pair outcomes",
        "terminal_rule": "posterior mean over eight candidates and zero-gain incumbent",
        "bootstrap_draws": 4000,
        "primary_bootstrap_seed": 20261109,
        "truth_audit": "after every test-root decision is sealed",
    }
    if set(calibration_seeds).intersection(test_seeds):
        raise ValueError("calibration and test seeds overlap")
    study_config = {**config, "test_seeds": list(full_test_seeds)}
    study_config.pop("test_index_range")
    study_config["calibration_seeds"] = list(full_calibration_seeds)
    study_config.pop("calibration_index_range")
    study_config["execution_partitioning"] = "four contiguous disjoint calibration/test shards with shared checkpoints"
    base_config_hash = _digest(study_config)
    config["study_config_hash"] = base_config_hash
    script_path = Path(__file__).resolve()
    script_hash = _sha256(script_path)
    dependency_hash = _digest(dependencies)
    code_hash = _digest({"runner_sha256": script_hash, "source_identity": source_identity})
    atomic_json(output / "identity.json", {
        "config": config,
        "source": source_identity,
        "runner": {"path": str(script_path), "sha256": script_hash},
        "dependencies": dependencies,
    })

    policies: dict[str, Policy] = {"incumbent": Policy.from_mapping({"intensity": 0.18, "bias": 0.0}, metadata={"v9": "initial"})}
    policies.update(discovery_panel_b4())
    validate_panel(policies)
    atomic_json(output / "policies.json", {key: value.to_record() for key, value in policies.items()})
    world = HighwayPhysicalWorld(HighwayPhysicalConfig(
        horizon=20,
        lanes_count=4,
        vehicles_count=20,
        vehicles_density=1.0,
        simulation_frequency=5,
        initial_lane_id=1,
        action_hold_steps=1,
    ))
    journal = EvaluationJournal(run_root / "evaluation-journal")
    checkpoints = CheckpointStore(run_root / "checkpoints", git_commit=source_identity["historical_git_commit"])
    registry = SeedRegistry(args.seed_registry or run_root / "seed-registry.sqlite")
    registry.register(
        [(seed, {"namespace": f"{PROTOCOL_ID}:calibration", "seed": seed}) for seed in calibration_seeds]
        + [(seed, {"namespace": f"{PROTOCOL_ID}:test", "seed": seed}) for seed in test_seeds]
    )
    physical_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    def evaluate_unit(
        *, cohort: str, seed: int, candidate_id: str, phase: str, method: str,
        mode: str, config_hash: str,
    ) -> dict[str, Any]:
        cache_key = (cohort, seed, candidate_id, phase, mode)
        if phase == "hf_query" and cache_key in physical_cache:
            cached = dict(physical_cache[cache_key])
            cached["_physical_evaluation_reused"] = True
            return cached
        policy = policies[candidate_id]
        policy_hash = hashlib.sha256(
            json.dumps(policy.to_record(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        unit = f"{PROTOCOL_ID}:{cohort}:{seed}:{phase}:{method}:{mode}:{candidate_id}"
        result_path = run_root / "unit-results" / cohort / str(seed) / phase / method / mode / f"{candidate_id}-{policy_hash}.json"
        config_hash_for_unit = config_hash

        def work() -> str:
            action_phase = policy.metadata.get("intervention_phase")
            if action_phase in {"early", "late"}:
                result = journal.evaluate(
                    lambda: world.evaluate_scheduled(
                        policy, seed=seed, mode=mode, action_phase=action_phase
                    ),
                    phase=phase, method=method, round_id=0,
                    policy_id=policy.policy_id, seed=seed, mode=mode,
                )
            else:
                result = journal.evaluate(
                    lambda: world.evaluate(policy, seed=seed, mode=mode),
                    phase=phase, method=method, round_id=0,
                    policy_id=policy.policy_id, seed=seed, mode=mode,
                )
            return json.dumps({
                "seed": seed, "cohort": cohort, "candidate_id": candidate_id,
                "phase": phase, "method": method, "mode": mode,
                "value": float(result.value),
                "environment_steps": int(result.environment_steps),
                "simulator_calls": int(result.simulator_calls),
                "compute_cost": result.compute_cost,
                "metadata": dict(result.metadata),
            }, sort_keys=True)

        record = checkpoints.run_unit(
            unit, work, output_path=result_path,
            config_hash=config_hash_for_unit, code_hash=code_hash,
            dependency_hash=dependency_hash,
            git_commit=source_identity["historical_git_commit"],
        )
        if record["status"] != "completed":
            raise RuntimeError(f"simulator evaluation failed: {unit}")
        result = json.loads(result_path.read_text())
        result["_physical_evaluation_reused"] = False
        if phase == "hf_query":
            physical_cache[cache_key] = dict(result)
        return result

    started = time.perf_counter()
    failures: list[dict[str, str]] = []
    calibration_proxy: dict[int, list[dict[str, Any]]] = {}
    calibration_truth: dict[int, dict[str, float]] = {}
    test_proxy_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    try:
        panel_ids = list(discovery_panel_b4())
        if args.calibration_input:
            calibration_input = args.calibration_input.resolve()
            calibration_proxy_rows = json.loads(
                (calibration_input / "calibration-proxy-rows.json").read_text()
            )["rows"]
            calibration_truth_rows = json.loads(
                (calibration_input / "calibration-truth.json").read_text()
            )["rows"]
            calibration_proxy = defaultdict(list)
            calibration_truth = defaultdict(dict)
            for row in calibration_proxy_rows:
                calibration_proxy[int(row["seed"])].append(
                    {key: value for key, value in row.items() if key != "seed"}
                )
            for row in calibration_truth_rows:
                calibration_truth[int(row["seed"])][str(row["candidate_id"])] = float(
                    row["paired_delta"]
                )
            if sorted(calibration_proxy) != sorted(CALIBRATION_SEEDS):
                raise ValueError("calibration input does not contain the frozen 40-root cohort")
            model = CalibrationModel.from_record(
                json.loads((calibration_input / "calibration-model.json").read_text())
            )
            if model.calibration_seed_count != len(CALIBRATION_SEEDS):
                raise ValueError("calibration input model has the wrong seed count")
        else:
            for seed in calibration_seeds:
                observer_values = {
                    candidate: evaluate_unit(
                        cohort="calibration", seed=seed, candidate_id=candidate,
                        phase="calibration_proxy", method="Calibration", mode="observer",
                        config_hash=base_config_hash,
                    )
                    for candidate in policies
                }
                incumbent_proxy = float(observer_values["incumbent"]["value"])
                proxy_rows = [{
                    "candidate_id": candidate,
                    "candidate_index": index,
                    "delta_proxy": float(observer_values[candidate]["value"]) - incumbent_proxy,
                    "highway_action": policies[candidate].metadata["highway_action"],
                    "intervention_phase": policies[candidate].metadata["intervention_phase"],
                } for index, candidate in enumerate(panel_ids)]
                actor_values = {
                    candidate: evaluate_unit(
                        cohort="calibration", seed=seed, candidate_id=candidate,
                        phase="calibration_actor", method="Calibration", mode="actor",
                        config_hash=base_config_hash,
                    )
                    for candidate in policies
                }
                incumbent_actor = float(actor_values["incumbent"]["value"])
                calibration_proxy[seed] = proxy_rows
                calibration_truth[seed] = {
                    candidate: float(actor_values[candidate]["value"]) - incumbent_actor
                    for candidate in panel_ids
                }

            model = fit_calibration_model(
                calibration_proxy,
                calibration_truth,
                covariance_shrinkage=0.5,
            )
        model_record = model.to_record()
        decision_config_hash = _digest({"base_config_hash": base_config_hash, "model": model_record})
        atomic_json(output / "calibration-proxy-rows.json", {
            "rows": [{"seed": seed, **row} for seed, rows in calibration_proxy.items() for row in rows]
        })
        atomic_json(output / "calibration-truth.json", {
            "rows": [{"seed": seed, "candidate_id": candidate, "paired_delta": value}
                     for seed, values in calibration_truth.items() for candidate, value in values.items()]
        })
        atomic_json(output / "calibration-model.json", model_record)
        atomic_json(output / "decision-identity.json", {
            "base_config_hash": base_config_hash,
            "decision_config_hash": decision_config_hash,
            "calibration_model": model_record,
        })

        if args.calibration_only:
            summary = {
                "protocol_id": PROTOCOL_ID,
                "status": "CALIBRATION_READY",
                "calibration_seed_count": len(calibration_proxy),
                "calibration_seeds": sorted(calibration_proxy),
                "calibration_model_sha256": _digest(model_record),
                "source_hash": source_identity["historical_source_hash"],
                "study_config_hash": base_config_hash,
                "failures": [],
                "scope": "sealed calibration cohort; no test outcomes opened",
            }
            atomic_json(output / "summary.json", summary)
            atomic_json(output / "cost-ledger.json", {
                "evaluation_journal": audit_evaluations(run_root / "evaluation-journal"),
                "wall_time_seconds": time.perf_counter() - started,
            })
            _write_manifest(output)
            world.close()
            return summary

        for seed in test_seeds:
            observer_values = {
                candidate: evaluate_unit(
                    cohort="test", seed=seed, candidate_id=candidate,
                    phase="proxy", method="Proxy", mode="observer",
                    config_hash=decision_config_hash,
                )
                for candidate in policies
            }
            incumbent_proxy = float(observer_values["incumbent"]["value"])
            rows = [{
                "candidate_id": candidate,
                "candidate_index": index,
                "delta_proxy": float(observer_values[candidate]["value"]) - incumbent_proxy,
                "highway_action": policies[candidate].metadata["highway_action"],
                "intervention_phase": policies[candidate].metadata["intervention_phase"],
            } for index, candidate in enumerate(panel_ids)]
            test_proxy_rows.extend({"seed": seed, **row} for row in rows)

            root_decisions: list[dict[str, Any]] = []
            for method in METHODS:
                posterior = GaussianPosterior(rows, model)
                incumbent_actor = evaluate_unit(
                    cohort="test", seed=seed, candidate_id="incumbent",
                    phase="hf_query", method=method, mode="actor",
                    config_hash=decision_config_hash,
                )
                incumbent_value = float(incumbent_actor["value"])
                query_rows.append({
                    "seed": seed, "method": method, "budget": BUDGET,
                    "query_index": -1, "candidate_id": "incumbent",
                    "paired_delta": 0.0, "logical_hf_query": False,
                    "physical_pair_evaluation": True,
                    "physical_evaluation_reused_across_methods": bool(
                        incumbent_actor.get("_physical_evaluation_reused", False)
                    ),
                    "simulator_calls": int(incumbent_actor["simulator_calls"]),
                    "physical_simulator_calls": 0 if incumbent_actor.get("_physical_evaluation_reused", False) else int(incumbent_actor["simulator_calls"]),
                    "environment_steps": int(incumbent_actor["environment_steps"]),
                    "compute_cost": incumbent_actor["compute_cost"],
                    "outcome_chasing": False,
                })
                fixed_order = sorted(
                    rows,
                    key=lambda row: (-float(row["delta_proxy"]), int(row["candidate_index"])),
                )
                uniform_order = _uniform_order(seed, panel_ids)
                for query_index in range(BUDGET):
                    acquisition_score: float | None = None
                    if method == "PIVOT-KG":
                        candidate, acquisition_score = exact_kg_next(posterior)
                    elif method == "IVR-Trace":
                        candidate, acquisition_score = _ivr_trace_next(posterior)
                    elif method == "Top-Proxy":
                        candidate = next(
                            str(row["candidate_id"]) for row in fixed_order
                            if str(row["candidate_id"]) not in posterior.observed
                        )
                    else:
                        candidate = next(
                            item for item in uniform_order if item not in posterior.observed
                        )
                    before_mean = posterior.value(candidate)
                    before_sd = posterior.uncertainty(candidate)
                    actor_value = evaluate_unit(
                        cohort="test", seed=seed, candidate_id=candidate,
                        phase="hf_query", method=method, mode="actor",
                        config_hash=decision_config_hash,
                    )
                    paired_delta = float(actor_value["value"]) - incumbent_value
                    posterior.observe(candidate, paired_delta)
                    query_rows.append({
                        "seed": seed, "method": method, "budget": BUDGET,
                        "query_index": query_index, "candidate_id": candidate,
                        "paired_delta": paired_delta,
                        "posterior_mean_before": before_mean,
                        "posterior_sd_before": before_sd,
                        "posterior_mean_after": posterior.value(candidate),
                        "acquisition_score": acquisition_score,
                        "logical_hf_query": True,
                        "physical_pair_evaluation": True,
                        "physical_evaluation_reused_across_methods": bool(
                            actor_value.get("_physical_evaluation_reused", False)
                        ),
                        "simulator_calls": int(actor_value["simulator_calls"]),
                        "physical_simulator_calls": 0 if actor_value.get("_physical_evaluation_reused", False) else int(actor_value["simulator_calls"]),
                        "environment_steps": int(actor_value["environment_steps"]),
                        "compute_cost": actor_value["compute_cost"],
                        "analysis_truth_reused": False,
                        "outcome_chasing": False,
                    })
                selected = posterior.select()
                root_decisions.append({
                    "seed": seed, "method": method, "budget": BUDGET,
                    "queries": [row["candidate_id"] for row in query_rows[-BUDGET:]],
                    "selected_candidate": selected,
                    "selected_posterior_mean": posterior.value(selected),
                    "incumbent_posterior_mean": 0.0,
                    "outcome_chasing": False,
                })
            decision_rows.extend(root_decisions)
            atomic_json(output / "sealed-decisions" / f"seed-{seed}.json", {
                "seed": seed,
                "decision_config_hash": decision_config_hash,
                "methods": root_decisions,
                "truth_opened": False,
            })

            # Actor truth is opened only after all four selectors have finished.
            truth_incumbent = evaluate_unit(
                cohort="test", seed=seed, candidate_id="incumbent",
                phase="truth_audit", method="Truth-Audit", mode="actor",
                config_hash=decision_config_hash,
            )
            truth_base = float(truth_incumbent["value"])
            truth_values: dict[str, float] = {}
            for candidate in panel_ids:
                result = evaluate_unit(
                    cohort="test", seed=seed, candidate_id=candidate,
                    phase="truth_audit", method="Truth-Audit", mode="actor",
                    config_hash=decision_config_hash,
                )
                truth_values[candidate] = float(result["value"]) - truth_base
                truth_rows.append({
                    "seed": seed, "candidate_id": candidate,
                    "paired_delta": truth_values[candidate],
                    "audit_only": True, "logical_hf_query": False,
                    "simulator_calls": int(result["simulator_calls"]),
                    "environment_steps": int(result["environment_steps"]),
                })
            truth_rows.append({
                "seed": seed, "candidate_id": "incumbent", "paired_delta": 0.0,
                "audit_only": True, "logical_hf_query": False,
                "simulator_calls": int(truth_incumbent["simulator_calls"]),
                "environment_steps": int(truth_incumbent["environment_steps"]),
            })
            for method in METHODS:
                method_queries = next(row["queries"] for row in root_decisions if row["method"] == method)
                for candidate in method_queries:
                    physical_result = evaluate_unit(
                        cohort="test", seed=seed, candidate_id=candidate,
                        phase="hf_query", method=method, mode="actor",
                        config_hash=decision_config_hash,
                    )
                    observed_delta = float(physical_result["value"]) - float(
                        evaluate_unit(
                            cohort="test", seed=seed, candidate_id="incumbent",
                            phase="hf_query", method=method, mode="actor",
                            config_hash=decision_config_hash,
                        )["value"]
                    )
                    if not math.isclose(observed_delta, truth_values[candidate], rel_tol=0.0, abs_tol=1e-12):
                        raise RuntimeError(f"query/audit mismatch at seed={seed}, method={method}, candidate={candidate}")

            best_value = max(0.0, *truth_values.values())
            root_results: list[dict[str, Any]] = []
            for decision in root_decisions:
                selected = str(decision["selected_candidate"])
                selected_value = 0.0 if selected == "incumbent" else truth_values[selected]
                isr = best_value - selected_value
                result = {
                    **decision,
                    "selected_actor_delta": selected_value,
                    "actor_best_delta": best_value,
                    "ISR": isr,
                }
                promotion_rows.append(result)
                root_results.append(result)
            uniform_expected = uniform_expected_selection(rows, model, truth_values, BUDGET)
            for subset in uniform_expected["rows"]:
                subset_rows.append({"seed": seed, **subset})
            promotion_rows.append({
                "seed": seed, "method": "Uniform-Exact-Expectation-Audit",
                "budget": BUDGET, "selected_candidate": "expected_policy",
                "ISR": float(uniform_expected["mean_ISR"]),
                "subset_count": int(uniform_expected["subset_count"]),
                "audit_only": True,
            })
            atomic_json(output / "sealed-decisions" / f"seed-{seed}.json", {
                "seed": seed,
                "decision_config_hash": decision_config_hash,
                "methods": root_decisions,
                "truth_opened": True,
                "truth_audit_sha256": _digest(truth_values),
            })
    except Exception as error:  # Persist partial evidence and mark the protocol as failed.
        failures.append({"error_type": type(error).__name__, "error": str(error)})
    finally:
        world.close()

    contrasts: dict[str, Any] = {}
    pivot_by_seed = {
        int(row["seed"]): float(row["ISR"])
        for row in promotion_rows if row["method"] == "PIVOT-KG"
    }
    for method in ("IVR-Trace", "Top-Proxy", "Uniform-Random", "Uniform-Exact-Expectation-Audit"):
        method_by_seed = {
            int(row["seed"]): float(row["ISR"])
            for row in promotion_rows if row["method"] == method
        }
        shared = sorted(set(pivot_by_seed).intersection(method_by_seed))
        values = [method_by_seed[seed] - pivot_by_seed[seed] for seed in shared]
        contrasts[method] = {
            "estimand": f"ISR({method}) - ISR(PIVOT-KG), B=4",
            "n_root_pairs": len(values),
            "mean": float(np.mean(values)) if values else None,
            "ci95": _bootstrap_ci(values, seed=20261109),
            "positive_favors": "PIVOT-KG",
            "positive_pairs": sum(value > 1e-12 for value in values),
            "negative_pairs": sum(value < -1e-12 for value in values),
            "zero_pairs": sum(abs(value) <= 1e-12 for value in values),
            "primary": method == "IVR-Trace",
            "physical_method": method != "Uniform-Exact-Expectation-Audit",
        }

    status = "DEV_SMOKE" if args.smoke else "DEV_INDEPENDENT_PHYSICAL_REPLICATION"
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "calibration_seed_count": len(calibration_seeds),
        "test_seed_count": len(test_seeds),
        "candidate_count": 8,
        "budget": BUDGET,
        "methods": list(METHODS),
        "failures": failures,
        "decision_row_count": len(decision_rows),
        "query_row_count": len(query_rows),
        "truth_audit_row_count": len(truth_rows),
        "subset_audit_row_count": len(subset_rows),
        "contrasts": contrasts,
        "primary_contrast": contrasts["IVR-Trace"],
        "promotion_status": "NOT_PROMOTED" if failures else "READY_FOR_REVIEW",
        "analysis_note": (
            "IVR-Trace is the primary paired physical comparison. Exact Uniform is a post-decision "
            "expectation over audit truth, not an executed selector."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "test-proxy-rows.json", {"rows": test_proxy_rows})
    atomic_json(output / "query-ledger.json", {"rows": query_rows})
    atomic_json(output / "decision-rows.json", {"rows": decision_rows})
    atomic_json(output / "promotion-results.json", {"rows": promotion_rows})
    atomic_json(output / "truth-audit.json", {"rows": truth_rows})
    atomic_json(output / "uniform-subset-audit.json", {"rows": subset_rows})
    atomic_json(output / "cost-ledger.json", {
        "evaluation_journal": audit_evaluations(run_root / "evaluation-journal"),
        "wall_time_seconds": time.perf_counter() - started,
    })
    _write_manifest(output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--seed-registry", type=Path)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-stop", type=int, default=len(TEST_SEEDS))
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-stop", type=int, default=len(CALIBRATION_SEEDS))
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
