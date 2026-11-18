#!/usr/bin/env python3
"""Run the preregistered eight-candidate HighwayEnv B=4 redesign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.revision.evaluation_journal import EvaluationJournal, audit_evaluations
from experiments.revision.highway_b4_redesign import (
    candidate_panel,
    deterministic_kg_next,
    discovery_panel_b4,
    enumerate_subsets,
    exact_kg_next,
    uniform_expected_selection,
    validate_panel,
)
from experiments.revision.highway_calibration import GaussianPosterior, fit_calibration_model
from experiments.revision.seed_registry import SeedRegistry
from pivot.cloud.archive import atomic_json
from pivot.core.policy import Policy
from pivot.environments.revision_highway import HighwayPhysicalConfig, HighwayPhysicalWorld
from pivot.runner.checkpoint import CheckpointStore

CALIBRATION_SEEDS = tuple(3300003 + 41 * index for index in range(40))
TEST_SEEDS = tuple(3400003 + 41 * index for index in range(120))
PROTOCOL_ID = "highway-physical-reactive-b4-redesign-v1"
CALIBRATION_NAMESPACE = "highway-b4-redesign-calibration-20260921-v1"
TEST_NAMESPACE = "highway-b4-redesign-test-20260921-v1"
METHODS = ("Uniform HF", "Calibrated PIVOT-KG", "All-HF Oracle")
BUDGETS = (2, 4)
PRIMARY_BUDGET = 4
ALL_HF_BUDGET = 8
CALIBRATION_SHRINKAGE = 0.5
SAMPLE_COUNTS = (64, 256, 1024)
PILOT_COUNT = 20


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


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _source(root: Path, expected_commit: str | None) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit and commit != expected_commit:
        raise ValueError("HEAD differs from expected commit")
    names = [
        "scripts/run_highway_b4_redesign.py",
        "experiments/revision/highway_b4_redesign.py",
        "experiments/revision/highway_calibration.py",
        "src/pivot/environments/revision_highway.py",
        "requirements/revision-highway.txt",
    ]
    hashes = {name: _sha256(root / name) for name in names}
    return {"git_commit": commit, "source_hash": _digest(hashes), "files": hashes}


def _dependencies() -> dict[str, Any]:
    names = ("highway-env", "gymnasium", "numpy", "pivot-research", "PyYAML")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"python": platform.python_version(), "distributions": versions}


def _slug(value: str) -> str:
    return value.lower().replace(" ", "-")


def _unit_key(cohort: str, seed: int, phase: str, method: str, budget: int, policy_id: str) -> str:
    return f"{PROTOCOL_ID}:{cohort}:{seed}:{phase}:{_slug(method)}:{budget}:{policy_id}"


def _write_manifest(root: Path) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "content-manifest.json":
            data = path.read_bytes()
            files[str(path.relative_to(root))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    atomic_json(root / "content-manifest.json", {"files": files})


def _bootstrap_ci(values: Sequence[float], *, seed: int, draws: int = 4000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def _contrast(rows: Sequence[Mapping[str, Any]], *, budget: int) -> dict[str, Any]:
    paired = [
        float(row["uniform_expected_ISR"]) - float(row["pivot_ISR"])
        for row in rows
        if int(row["budget"]) == budget
    ]
    low, high = _bootstrap_ci(paired, seed=20260927 + budget)
    return {
        "budget": budget,
        "estimand": "ISR_uniform_expected_minus_Calibrated_PIVOT-KG",
        "n_seed_pairs": len(paired),
        "mean": float(np.mean(paired)) if paired else None,
        "ci95": [low, high],
        "sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
        "positive_favors": "Calibrated PIVOT-KG",
        "positive_pairs": sum(value > 1e-12 for value in paired),
        "negative_pairs": sum(value < -1e-12 for value in paired),
        "zero_pairs": sum(abs(value) <= 1e-12 for value in paired),
    }


def _leave_one_out(rows: Sequence[Mapping[str, Any]], budget: int) -> dict[str, Any]:
    values = [
        float(row["uniform_expected_ISR"]) - float(row["pivot_ISR"])
        for row in rows
        if int(row["budget"]) == budget
    ]
    if len(values) < 2:
        return {"min": None, "max": None, "range": None}
    estimates = [(sum(values) - value) / (len(values) - 1) for value in values]
    return {"min": float(min(estimates)), "max": float(max(estimates)), "range": float(max(estimates) - min(estimates))}


def _pilot_decision(
    rows: Sequence[Mapping[str, Any]],
    model: Any,
    truth: Mapping[str, float],
    budget: int,
    sample_count: int | None,
) -> dict[str, Any]:
    posterior = GaussianPosterior(rows, model)
    queries: list[str] = []
    scores: list[float | None] = []
    for _ in range(budget):
        if sample_count is None:
            candidate, score = exact_kg_next(posterior)
        else:
            candidate = deterministic_kg_next(posterior, sample_count=sample_count)
            score = None
        queries.append(candidate)
        scores.append(score)
        posterior.observe(candidate, float(truth[candidate]))
    return {"queries": queries, "scores": scores, "selected": posterior.select()}


def _scheduled_phase(policy: Policy) -> str | None:
    phase = policy.metadata.get("intervention_phase")
    return str(phase) if phase in {"early", "late"} else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite redesign output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = _source(root, args.expected_commit)
    dependencies = _dependencies()
    calibration_seeds = CALIBRATION_SEEDS[:2] if args.smoke else CALIBRATION_SEEDS
    test_seeds = TEST_SEEDS[:2] if args.smoke else TEST_SEEDS
    config = {
        "protocol_id": PROTOCOL_ID,
        "calibration_namespace": CALIBRATION_NAMESPACE,
        "test_namespace": TEST_NAMESPACE,
        "calibration_seeds": list(calibration_seeds),
        "test_seeds": list(test_seeds),
        "candidate_actions": ["SLOWER", "LANE_LEFT", "LANE_RIGHT", "FASTER"],
        "candidate_phases": ["early", "late"],
        "candidate_count": 8,
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "all_hf_budget": ALL_HF_BUDGET,
        "subset_counts": {str(budget): len(enumerate_subsets(tuple(discovery_panel_b4()), budget)) for budget in BUDGETS},
        "sample_counts": list(SAMPLE_COUNTS),
        "pilot_count": PILOT_COUNT,
        "horizon": args.horizon,
        "lanes_count": 4,
        "initial_lane_id": 1,
        "vehicles_count": 20,
        "vehicles_density": 1.0,
        "simulation_frequency": 5,
        "calibration_covariance_shrinkage": CALIBRATION_SHRINKAGE,
        "truth_audit": "post_decision_actor_evaluations_excluded_from_logical_hf_budget",
        "paired_baseline_policy": "one_deterministic_incumbent_actor_rollout_shared_within_seed_and_budget",
        "analysis_plan": "exact_expected_uniform_subset_policy_vs_exact_affine_normal_pivot_with_physical_queries",
        "claim_scope": "new_highway_b4_redesign_not_historical_paper_result",
    }
    if set(calibration_seeds).intersection(test_seeds):
        raise ValueError("calibration and test seeds overlap")
    base_config_hash = _digest(config)
    dependency_hash = _digest(dependencies)
    atomic_json(output / "identity.json", {"config": config, "source": source, "dependencies": dependencies})
    policies = candidate_panel()
    validate_panel(policies)
    atomic_json(output / "policies.json", {key: value.to_record() for key, value in policies.items()})
    world = HighwayPhysicalWorld(
        HighwayPhysicalConfig(
            horizon=args.horizon,
            lanes_count=4,
            vehicles_count=20,
            vehicles_density=1.0,
            simulation_frequency=5,
            initial_lane_id=1,
        )
    )
    journal = EvaluationJournal(output / "evaluation-journal")
    checkpoints = CheckpointStore(output / "checkpoints", git_commit=source["git_commit"])
    registry = SeedRegistry(output / "seed-registry.sqlite")
    registry.register(
        [(seed, {"namespace": CALIBRATION_NAMESPACE, "cohort": "calibration", "seed": seed}) for seed in calibration_seeds]
        + [(seed, {"namespace": TEST_NAMESPACE, "cohort": "test", "seed": seed}) for seed in test_seeds]
    )
    failures: list[dict[str, Any]] = []
    calibration_proxy: dict[int, list[dict[str, Any]]] = {}
    calibration_truth: dict[int, dict[str, float]] = {}
    proxy_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def evaluate_unit(*, cohort: str, seed: int, candidate_id: str, policy: Policy, phase: str, method: str, budget: int, mode: str, config_hash: str) -> dict[str, Any]:
        result_path = output / "results" / cohort / str(seed) / _slug(method) / str(budget) / phase / f"{candidate_id}.json"
        key = _unit_key(cohort, seed, phase, method, budget, candidate_id)

        def work() -> str:
            intervention_phase = _scheduled_phase(policy)
            if intervention_phase is None:
                result = journal.evaluate(
                    lambda: world.evaluate(policy, seed=seed, mode=mode),
                    phase=phase, method=PROTOCOL_ID, round_id=0, policy_id=policy.policy_id, seed=seed, mode=mode,
                )
            else:
                result = journal.evaluate(
                    lambda: world.evaluate_scheduled(policy, seed=seed, mode=mode, action_phase=intervention_phase),
                    phase=phase, method=PROTOCOL_ID, round_id=0, policy_id=policy.policy_id, seed=seed, mode=mode,
                )
            return json.dumps({
                "seed": seed, "cohort": cohort, "candidate_id": candidate_id, "mode": mode,
                "phase": phase, "method": method, "budget": budget, "value": float(result.value),
                "environment_steps": result.environment_steps, "simulator_calls": result.simulator_calls,
                "compute_cost": result.compute_cost, "metadata": dict(result.metadata),
            }, sort_keys=True)

        record = checkpoints.run_unit(key, work, output_path=result_path, config_hash=config_hash, code_hash=source["source_hash"], dependency_hash=dependency_hash, git_commit=source["git_commit"])
        if record["status"] != "completed":
            raise RuntimeError(f"redesign unit failed: {key}")
        return json.loads(result_path.read_text())

    def proxy_rows_from_values(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        incumbent_value = float(values["incumbent"]["value"])
        return [
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "delta_proxy": float(values[candidate_id]["value"]) - incumbent_value,
                "highway_action": policies[candidate_id].metadata["highway_action"],
                "intervention_phase": policies[candidate_id].metadata["intervention_phase"],
                "intervention_start": int(policies[candidate_id].metadata["intervention_start"]),
                "intervention_end": int(policies[candidate_id].metadata["intervention_end"]),
                "policy_id": policies[candidate_id].policy_id,
            }
            for index, candidate_id in enumerate(discovery_panel_b4())
        ]

    try:
        for seed in calibration_seeds:
            proxy_values = {
                candidate_id: evaluate_unit(cohort="calibration", seed=seed, candidate_id=candidate_id, policy=policy, phase="calibration_proxy", method="Calibration", budget=0, mode="observer", config_hash=base_config_hash)
                for candidate_id, policy in policies.items()
            }
            proxy = proxy_rows_from_values(proxy_values)
            calibration_proxy[seed] = proxy
            actor_values = {
                candidate_id: evaluate_unit(cohort="calibration", seed=seed, candidate_id=candidate_id, policy=policy, phase="calibration_actor", method="Calibration", budget=0, mode="actor", config_hash=base_config_hash)
                for candidate_id, policy in policies.items()
            }
            incumbent_value = float(actor_values["incumbent"]["value"])
            calibration_truth[seed] = {row["candidate_id"]: float(actor_values[row["candidate_id"]]["value"]) - incumbent_value for row in proxy}
        model = fit_calibration_model(calibration_proxy, calibration_truth, covariance_shrinkage=CALIBRATION_SHRINKAGE)
        atomic_json(output / "calibration-proxy-rows.json", {"rows": [{"seed": seed, **row} for seed, rows in calibration_proxy.items() for row in rows]})
        atomic_json(output / "calibration-truth.json", {"rows": [{"seed": seed, "candidate_id": candidate, "paired_delta": value} for seed, values in calibration_truth.items() for candidate, value in values.items()]})
        atomic_json(output / "calibration-model.json", model.to_record())
        decision_config_hash = _digest({"config": config, "calibration_model": model.to_record()})
        atomic_json(output / "decision-identity.json", {"base_config_hash": base_config_hash, "decision_config_hash": decision_config_hash, "calibration_model": model.to_record()})

        for seed in calibration_seeds[:PILOT_COUNT]:
            proxy = calibration_proxy[seed]
            truth = calibration_truth[seed]
            for budget in BUDGETS:
                exact = _pilot_decision(proxy, model, truth, budget, None)
                exact_replay = _pilot_decision(proxy, model, truth, budget, None)
                grids = {
                    str(sample_count): _pilot_decision(
                        proxy, model, truth, budget, sample_count
                    )
                    for sample_count in SAMPLE_COUNTS
                }
                pilot_rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "exact": exact,
                        "exact_replay": exact_replay,
                        "grid_diagnostics": grids,
                    }
                )
        convergence_ok = all(
            row["exact"] == row["exact_replay"]
            and all(
                score is not None and np.isfinite(score) and score >= 0.0
                for score in row["exact"]["scores"]
            )
            for row in pilot_rows
        )
        atomic_json(
            output / "convergence-pilot.json",
            {
                "decision_rule": "exact_affine_normal_upper_envelope",
                "sample_counts": list(SAMPLE_COUNTS),
                "sample_grid_role": "diagnostic_only_not_used_for_decisions",
                "pilot_cohort": "calibration",
                "pilot_count": min(PILOT_COUNT, len(calibration_seeds)),
                "rows": pilot_rows,
                "convergence_ok": convergence_ok,
            },
        )
        if not convergence_ok:
            raise RuntimeError("exact KG determinism gate failed")

        for seed in test_seeds:
            values = {
                candidate_id: evaluate_unit(cohort="test", seed=seed, candidate_id=candidate_id, policy=policy, phase="proxy", method="Proxy", budget=0, mode="observer", config_hash=decision_config_hash)
                for candidate_id, policy in policies.items()
            }
            rows = proxy_rows_from_values(values)
            proxy_rows.extend({"seed": seed, **row} for row in rows)
            # Run each PIVOT decision against fresh actor rollouts.  The full
            # actor audit is intentionally delayed until every adaptive query
            # has been selected, so unqueried test outcomes cannot influence
            # acquisition.
            pivot_decisions: dict[int, dict[str, Any]] = {}
            for budget in BUDGETS:
                posterior = GaussianPosterior(rows, model)
                observed: dict[str, float] = {}
                incumbent_actor = evaluate_unit(
                    cohort="test",
                    seed=seed,
                    candidate_id="incumbent",
                    policy=policies["incumbent"],
                    phase="hf_query",
                    method="Calibrated PIVOT-KG",
                    budget=budget,
                    mode="actor",
                    config_hash=decision_config_hash,
                )
                incumbent_value = float(incumbent_actor["value"])
                query_rollouts = int(incumbent_actor["simulator_calls"])
                query_compute_cost = float(incumbent_actor["compute_cost"] or 0.0)
                for query_index in range(budget):
                    candidate_id, evsi = exact_kg_next(posterior)
                    before = posterior.value(candidate_id)
                    candidate_actor = evaluate_unit(
                        cohort="test",
                        seed=seed,
                        candidate_id=candidate_id,
                        policy=policies[candidate_id],
                        phase="hf_query",
                        method="Calibrated PIVOT-KG",
                        budget=budget,
                        mode="actor",
                        config_hash=decision_config_hash,
                    )
                    paired_delta = float(candidate_actor["value"]) - incumbent_value
                    posterior.observe(candidate_id, paired_delta)
                    after = posterior.value(candidate_id)
                    observed[candidate_id] = paired_delta
                    query_rollouts += int(candidate_actor["simulator_calls"])
                    query_compute_cost += float(candidate_actor["compute_cost"] or 0.0)
                    query_rows.append(
                        {
                            "seed": seed,
                            "method": "Calibrated PIVOT-KG",
                            "budget": budget,
                            "query_index": query_index,
                            "candidate_id": candidate_id,
                            "paired_delta": paired_delta,
                            "posterior_before": before,
                            "posterior_after": after,
                            "EVSI": evsi,
                            "logical_hf_query": True,
                            "physical_pair_evaluation": True,
                            "analysis_truth_reused": False,
                            "simulator_calls": int(candidate_actor["simulator_calls"]),
                            "compute_cost": float(candidate_actor["compute_cost"] or 0.0),
                            "shared_incumbent_simulator_calls": int(incumbent_actor["simulator_calls"]) if query_index == 0 else 0,
                            "shared_incumbent_compute_cost": float(incumbent_actor["compute_cost"] or 0.0) if query_index == 0 else 0.0,
                            "paired_baseline_reused_within_budget": True,
                            "outcome_chasing": False,
                        }
                    )
                pivot_decisions[budget] = {
                    "posterior": posterior,
                    "observed": observed,
                    "selected_candidate": posterior.select(),
                    "query_rollouts": query_rollouts,
                    "query_compute_cost": query_compute_cost,
                }

            # This audit is descriptive only.  It runs after all PIVOT query
            # decisions and supplies the common actor-world truth used for the
            # exact Uniform expectation and the all-HF reference.
            truth_values: dict[str, float] = {}
            truth_actor_values: dict[str, dict[str, Any]] = {}
            incumbent_actor = evaluate_unit(
                cohort="test",
                seed=seed,
                candidate_id="incumbent",
                policy=policies["incumbent"],
                phase="truth_audit",
                method="Truth Audit",
                budget=ALL_HF_BUDGET,
                mode="actor",
                config_hash=decision_config_hash,
            )
            incumbent_value = float(incumbent_actor["value"])
            for candidate_id in discovery_panel_b4():
                result = evaluate_unit(
                    cohort="test",
                    seed=seed,
                    candidate_id=candidate_id,
                    policy=policies[candidate_id],
                    phase="truth_audit",
                    method="Truth Audit",
                    budget=ALL_HF_BUDGET,
                    mode="actor",
                    config_hash=decision_config_hash,
                )
                truth_actor_values[candidate_id] = result
                truth_values[candidate_id] = float(result["value"]) - incumbent_value
            for budget, decision in pivot_decisions.items():
                for candidate_id, observed_value in decision["observed"].items():
                    if not np.isclose(observed_value, truth_values[candidate_id], rtol=0.0, atol=1e-12):
                        raise RuntimeError(f"HF query/audit mismatch for {candidate_id} at seed {seed}")
            best = max(0.0, *truth_values.values())
            for budget in BUDGETS:
                uniform = uniform_expected_selection(rows, model, truth_values, budget)
                for subset in uniform["rows"]:
                    subset_rows.append({"seed": seed, **subset})
                decision = pivot_decisions[budget]
                selected = str(decision["selected_candidate"])
                pivot_value = 0.0 if selected == "incumbent" else truth_values[selected]
                promotion_rows.append({"seed": seed, "method": "Calibrated PIVOT-KG", "budget": budget, "selected_candidate": selected, "true_best_candidate": max({"incumbent": 0.0, **truth_values}, key=lambda key: (0.0 if key == "incumbent" else truth_values[key], 0 if key == "incumbent" else -1)), "selected_actor_delta": pivot_value, "actor_best_delta": best, "pivot_ISR": best - pivot_value, "uniform_expected_ISR": uniform["mean_ISR"], "uniform_subset_count": uniform["subset_count"], "hf_queries": budget, "physical_query_simulator_calls": decision["query_rollouts"], "query_compute_cost": decision["query_compute_cost"], "candidate_count": len(rows), "outcome_chasing": False})
                promotion_rows.append({"seed": seed, "method": "Uniform HF", "budget": budget, "selected_candidate": "expected_policy", "true_best_candidate": "expected_policy", "selected_actor_delta": best - uniform["mean_ISR"], "actor_best_delta": best, "ISR": uniform["mean_ISR"], "pivot_ISR": None, "uniform_expected_ISR": uniform["mean_ISR"], "uniform_subset_count": uniform["subset_count"], "hf_queries": budget, "physical_query_simulator_calls": 0, "candidate_count": len(rows), "outcome_chasing": False, "analysis_only_expected_allocation": True})
            best_candidate = max(
                {"incumbent": 0.0, **truth_values},
                key=lambda candidate: (
                    0.0 if candidate == "incumbent" else truth_values[candidate],
                    0 if candidate == "incumbent" else -1,
                ),
            )
            promotion_rows.append({"seed": seed, "method": "All-HF Oracle", "budget": ALL_HF_BUDGET, "selected_candidate": best_candidate, "true_best_candidate": best_candidate, "selected_actor_delta": best, "actor_best_delta": best, "ISR": 0.0, "hf_queries": ALL_HF_BUDGET, "physical_query_simulator_calls": int(incumbent_actor["simulator_calls"]) + sum(int(result["simulator_calls"]) for result in truth_actor_values.values()), "candidate_count": len(rows), "outcome_chasing": False, "analysis_only_truth_audit": True})
            for candidate_id, value in truth_values.items():
                truth_rows.append({"seed": seed, "method": "Truth Audit", "budget": ALL_HF_BUDGET, "candidate_id": candidate_id, "paired_delta": value, "audit_only": True, "logical_hf_query": False, "physical_query_simulator_calls": int(truth_actor_values[candidate_id]["simulator_calls"]), "outcome_chasing": False})
            truth_rows.append({"seed": seed, "method": "Truth Audit", "budget": ALL_HF_BUDGET, "candidate_id": "incumbent", "paired_delta": 0.0, "audit_only": True, "logical_hf_query": False, "outcome_chasing": False})
    except Exception as error:  # noqa: BLE001 - persist the gate failure
        failures.append({"error_type": type(error).__name__, "error": str(error)})
    finally:
        world.close()

    rows_by_budget: list[dict[str, Any]] = []
    for seed in test_seeds:
        for budget in BUDGETS:
            pivot = next((row for row in promotion_rows if row["seed"] == seed and row["method"] == "Calibrated PIVOT-KG" and row["budget"] == budget), None)
            uniform = next((row for row in promotion_rows if row["seed"] == seed and row["method"] == "Uniform HF" and row["budget"] == budget), None)
            if pivot and uniform:
                rows_by_budget.append({"seed": seed, "budget": budget, "pivot_ISR": pivot["pivot_ISR"], "uniform_expected_ISR": uniform["uniform_expected_ISR"]})
    contrasts = [_contrast(rows_by_budget, budget=budget) for budget in BUDGETS]
    convergence = json.loads((output / "convergence-pilot.json").read_text()) if (output / "convergence-pilot.json").exists() else {"convergence_ok": False}
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "DEV_SMOKE" if args.smoke else "DEV_EXTERNAL_REDESIGN",
        "calibration_seed_count": len(calibration_seeds),
        "test_seed_count": len(test_seeds),
        "candidate_count": 8,
        "proxy_row_count": len(proxy_rows),
        "promotion_row_count": len(promotion_rows),
        "query_row_count": len(query_rows),
        "subset_row_count": len(subset_rows),
        "truth_audit_row_count": len(truth_rows),
        "failures": failures,
        "budget_contrasts": contrasts,
        "primary_contrast": next(item for item in contrasts if item["budget"] == PRIMARY_BUDGET),
        "leave_one_out_primary": _leave_one_out(rows_by_budget, PRIMARY_BUDGET),
        "convergence_ok": bool(convergence.get("convergence_ok", False)),
        "promotion_status": "NOT_PROMOTED" if failures or not convergence.get("convergence_ok", False) else "READY_FOR_REVIEW",
        "stop_reason": "failure_or_convergence_gate" if failures or not convergence.get("convergence_ok", False) else "gate_passed_review_required",
        "analysis_note": "Uniform is exact subset expectation; truth probes are audit-only and never enter acquisition decisions.",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "rows.json", {"rows": rows_by_budget})
    atomic_json(output / "proxy-rows.json", {"rows": proxy_rows})
    atomic_json(output / "promotion-results.json", {"rows": promotion_rows})
    atomic_json(output / "query-ledger.json", {"rows": query_rows})
    atomic_json(output / "subset-ledger.json", {"rows": subset_rows})
    atomic_json(output / "truth-audit.json", {"rows": truth_rows})
    atomic_json(output / "cost-ledger.json", {"evaluation_journal": audit_evaluations(output / "evaluation-journal"), "wall_time": time.perf_counter() - started})
    _write_manifest(output)
    return {"status": summary["status"], "output": str(output), **summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
