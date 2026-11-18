#!/usr/bin/env python3
"""Run the calibrated, sequential HighwayEnv redesign.

The calibration cohort is fully audited before the held-out test cohort is
opened.  Test decisions use the same proxy panel and paired actor queries, but
the calibrated PIVOT-KG selector recomputes EVSI after every observation.
Post-decision actor audits are retained for diagnostics and never enter a
selection decision or the logical query budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.revision.evaluation_journal import EvaluationJournal, audit_evaluations
from experiments.revision.highway_calibration import (
    GaussianPosterior,
    choose_next,
    fit_calibration_model,
)
from experiments.revision.seed_registry import SeedRegistry
from pivot.cloud.archive import atomic_json
from pivot.core.policy import Policy
from pivot.environments.revision_highway import HighwayPhysicalConfig, HighwayPhysicalWorld
from pivot.runner.checkpoint import CheckpointStore

CALIBRATION_SEEDS = tuple(1300003 + 37 * index for index in range(20))
TEST_SEEDS = tuple(1400003 + 37 * index for index in range(60))
PROTOCOL_ID = "highway-physical-reactive-calibrated-v2"
CALIBRATION_NAMESPACE = "highway-calibration-20260921-v2"
TEST_NAMESPACE = "highway-test-20260921-v2"
METHODS = ("Proxy Only", "Uniform HF", "Calibrated PIVOT-KG", "All-HF Oracle")
BUDGETS = (1, 2, 4)
PRIMARY_BUDGET = 2
ALL_HF_BUDGET = 5
CALIBRATION_SHRINKAGE = 0.5
EXPECTED_ACTIONS = (0, 1, 2, 3, 4)


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
        "scripts/run_highway_calibrated.py",
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
    return f"highway-calibrated:{cohort}:{seed}:{phase}:{_slug(method)}:{budget}:{policy_id}"


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


def _initial_policy() -> Policy:
    return Policy.from_mapping({"intensity": 0.18, "bias": 0.0}, metadata={"v9": "initial"})


def discovery_panel() -> dict[str, Policy]:
    """Return the fixed five-action panel plus the explicit no-update arm."""

    return {
        "incumbent": _initial_policy(),
        "slower": Policy.from_mapping(
            {"intensity": -0.8, "bias": 0.0}, metadata={"highway_action": "SLOWER"}
        ),
        "lane_left": Policy.from_mapping(
            {"intensity": 0.0, "bias": -0.8}, metadata={"highway_action": "LANE_LEFT"}
        ),
        "idle": Policy.from_mapping(
            {"intensity": 0.0, "bias": 0.0}, metadata={"highway_action": "IDLE"}
        ),
        "lane_right": Policy.from_mapping(
            {"intensity": 0.0, "bias": 0.8}, metadata={"highway_action": "LANE_RIGHT"}
        ),
        "faster": Policy.from_mapping(
            {"intensity": 0.8, "bias": 0.0}, metadata={"highway_action": "FASTER"}
        ),
    }


def _footprint(incumbent: Policy, candidate: Policy) -> dict[str, float]:
    return {
        key: abs(float(candidate.parameters.get(key, 0.0)) - float(incumbent.parameters.get(key, 0.0)))
        for key in ("intensity", "bias")
    }


def _proxy_rows(
    values: Mapping[str, Mapping[str, Any]], policies: Mapping[str, Policy]
) -> list[dict[str, Any]]:
    incumbent_value = float(values["incumbent"]["value"])
    incumbent = policies["incumbent"]
    rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(key for key in policies if key != "incumbent"):
        candidate = policies[candidate_id]
        metadata = values[candidate_id]["metadata"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "delta_proxy": float(values[candidate_id]["value"]) - incumbent_value,
                "footprint": _footprint(incumbent, candidate),
                "policy_id": candidate.policy_id,
                "highway_action": candidate.metadata.get("highway_action"),
                "initial_available_actions": list(metadata["initial_available_actions"]),
            }
        )
    return rows


def _bootstrap_ci(values: Sequence[float], *, seed: int, draws: int = 4000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _method_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(dict(row))
    output: list[dict[str, Any]] = []
    for (method, budget), group in sorted(grouped.items()):
        values = [float(row["ISR"]) for row in group]
        low, high = _bootstrap_ci(values, seed=20260921 + budget)
        output.append(
            {
                "method": method,
                "budget": budget,
                "n_seeds": len(group),
                "mean_ISR": float(np.mean(values)),
                "ISR_ci95": [low, high],
                "mean_hf_queries": float(np.mean([row["hf_queries"] for row in group])),
                "selected_counts": {
                    candidate: sum(row["selected_candidate"] == candidate for row in group)
                    for candidate in sorted({str(row["selected_candidate"]) for row in group})
                },
            }
        )
    return output


def _contrast(
    rows: Sequence[Mapping[str, Any]], *, budget: int, comparison: str = "Calibrated PIVOT-KG"
) -> dict[str, Any]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        if int(row["budget"]) != budget or str(row["method"]) not in {"Uniform HF", comparison}:
            continue
        by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = float(row["ISR"])
    paired = [
        values["Uniform HF"] - values[comparison]
        for values in by_seed.values()
        if len(values) == 2
    ]
    low, high = _bootstrap_ci(paired, seed=20260922 + budget)
    return {
        "budget": budget,
        "estimand": f"ISR_uniform_minus_{_slug(comparison)}",
        "n_seed_pairs": len(paired),
        "mean": float(np.mean(paired)) if paired else None,
        "ci95": [low, high],
        "positive_favors": comparison,
    }


def _seed_rng(seed: int, method: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{PROTOCOL_ID}:{seed}:{method}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _select_proxy(rows: Sequence[Mapping[str, Any]], observed: Mapping[str, float]) -> str:
    options = [("incumbent", 0.0, 0)]
    options.extend(
        (
            str(row["candidate_id"]),
            float(observed.get(str(row["candidate_id"]), row["delta_proxy"])),
            -1 - int(row["candidate_index"]),
        )
        for row in rows
    )
    return max(options, key=lambda item: (item[1], item[2]))[0]


def _truth_best(actor_values: Mapping[str, float]) -> tuple[str, float]:
    options = [("incumbent", 0.0)] + [(str(key), float(value)) for key, value in actor_values.items()]
    return max(options, key=lambda item: (item[1], 0 if item[0] == "incumbent" else -1))[0], max(
        value for _, value in options
    )


def _assert_actions(values: Mapping[str, Mapping[str, Any]]) -> tuple[int, ...]:
    observed = tuple(int(value) for value in values["incumbent"]["metadata"]["initial_available_actions"])
    if set(observed) != set(EXPECTED_ACTIONS):
        raise RuntimeError(f"center-lane action panel is not fully available: {observed}")
    return observed


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite calibrated output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = _source(root, args.expected_commit)
    dependencies = _dependencies()
    config = {
        "protocol_id": PROTOCOL_ID,
        "calibration_namespace": CALIBRATION_NAMESPACE,
        "test_namespace": TEST_NAMESPACE,
        "calibration_seeds": list(CALIBRATION_SEEDS if not args.smoke else CALIBRATION_SEEDS[:2]),
        "test_seeds": list(TEST_SEEDS if not args.smoke else TEST_SEEDS[:2]),
        "candidate_actions": ["SLOWER", "LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER"],
        "candidate_count": 5,
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "all_hf_budget": ALL_HF_BUDGET,
        "horizon": args.horizon,
        "lanes_count": 4,
        "initial_lane_id": 1,
        "action_hold_steps": 1,
        "vehicles_count": 20,
        "vehicles_density": 1.0,
        "simulation_frequency": 5,
        "calibration_covariance_shrinkage": CALIBRATION_SHRINKAGE,
        "truth_audit": "post_decision_actor_evaluations_excluded_from_hf_budget",
        "claim_scope": "external_DEV_redesign_not_paper_primary_claim",
        "power_note": "60 held-out roots targets approximately 0.02 ISR contrast at observed variance",
    }
    calibration_seeds = tuple(int(seed) for seed in config["calibration_seeds"])
    test_seeds = tuple(int(seed) for seed in config["test_seeds"])
    if set(calibration_seeds).intersection(test_seeds):
        raise ValueError("calibration and test seeds overlap")
    atomic_json(output / "identity.json", {"config": config, "source": source, "dependencies": dependencies})
    base_config_hash = _digest(config)
    dependency_hash = _digest(dependencies)
    world = HighwayPhysicalWorld(
        HighwayPhysicalConfig(
            horizon=args.horizon,
            lanes_count=4,
            vehicles_count=20,
            vehicles_density=1.0,
            simulation_frequency=5,
            initial_lane_id=1,
            action_hold_steps=1,
        )
    )
    journal = EvaluationJournal(output / "evaluation-journal")
    checkpoints = CheckpointStore(output / "checkpoints", git_commit=source["git_commit"])
    registry = SeedRegistry(output / "seed-registry.sqlite")
    registry.register(
        [(seed, {"namespace": CALIBRATION_NAMESPACE, "cohort": "calibration", "seed": seed}) for seed in calibration_seeds]
        + [(seed, {"namespace": TEST_NAMESPACE, "cohort": "test", "seed": seed}) for seed in test_seeds]
    )
    policies = discovery_panel()
    atomic_json(output / "policies.json", {key: value.to_record() for key, value in policies.items()})
    proxy_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    calibration_proxy: dict[int, list[dict[str, Any]]] = {}
    calibration_truth: dict[int, dict[str, float]] = {}
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    def evaluate_unit(
        *,
        cohort: str,
        seed: int,
        policy_id: str,
        policy: Policy,
        phase: str,
        method: str,
        budget: int,
        config_hash: str,
    ) -> dict[str, Any]:
        result_path = output / "results" / cohort / str(seed) / _slug(method) / str(budget) / phase / f"{policy_id}.json"
        key = _unit_key(cohort, seed, phase, method, budget, policy_id)

        def work() -> str:
            mode = "observer" if phase.endswith("proxy") or phase == "proxy" else "actor"
            result = journal.evaluate(
                lambda: world.evaluate(policy, seed=seed, mode=mode),
                phase=phase,
                method="highway-calibrated",
                round_id=0,
                policy_id=policy.policy_id,
                seed=seed,
                mode=mode,
            )
            return json.dumps(
                {
                    "seed": seed,
                    "cohort": cohort,
                    "policy_id": policy_id,
                    "mode": mode,
                    "phase": phase,
                    "method": method,
                    "budget": budget,
                    "value": float(result.value),
                    "environment_steps": result.environment_steps,
                    "simulator_calls": result.simulator_calls,
                    "compute_cost": result.compute_cost,
                    "metadata": dict(result.metadata),
                },
                sort_keys=True,
            )

        record = checkpoints.run_unit(
            key,
            work,
            output_path=result_path,
            config_hash=config_hash,
            code_hash=source["source_hash"],
            dependency_hash=dependency_hash,
            git_commit=source["git_commit"],
        )
        if record["status"] != "completed":
            raise RuntimeError(f"calibrated unit failed: {key}")
        return json.loads(result_path.read_text())

    try:
        # Phase 1: fully sealed calibration cohort.
        for seed in calibration_seeds:
            values = {
                policy_id: evaluate_unit(
                    cohort="calibration",
                    seed=seed,
                    policy_id=policy_id,
                    policy=policy,
                    phase="calibration_proxy",
                    method="Calibration",
                    budget=0,
                    config_hash=base_config_hash,
                )
                for policy_id, policy in policies.items()
            }
            _assert_actions(values)
            rows = _proxy_rows(values, policies)
            calibration_proxy[seed] = rows
            actor_values = {
                policy_id: evaluate_unit(
                    cohort="calibration",
                    seed=seed,
                    policy_id=policy_id,
                    policy=policy,
                    phase="calibration_actor",
                    method="Calibration",
                    budget=0,
                    config_hash=base_config_hash,
                )
                for policy_id, policy in policies.items()
            }
            incumbent_value = float(actor_values["incumbent"]["value"])
            calibration_truth[seed] = {
                str(row["candidate_id"]): float(actor_values[str(row["candidate_id"])] ["value"])
                - incumbent_value
                for row in rows
            }

        model = fit_calibration_model(
            calibration_proxy,
            calibration_truth,
            covariance_shrinkage=CALIBRATION_SHRINKAGE,
        )
        atomic_json(output / "calibration-proxy-rows.json", {"rows": [
            {"seed": seed, **row} for seed, rows in calibration_proxy.items() for row in rows
        ]})
        atomic_json(
            output / "calibration-truth.json",
            {"rows": [{"seed": seed, "candidate_id": candidate, "paired_delta": value}
                      for seed, values in calibration_truth.items() for candidate, value in values.items()]},
        )
        atomic_json(output / "calibration-model.json", model.to_record())
        decision_config_hash = _digest({"config": config, "calibration_model": model.to_record()})
        atomic_json(output / "decision-identity.json", {"base_config_hash": base_config_hash, "decision_config_hash": decision_config_hash, "calibration_model": model.to_record()})

        # Phase 2: held-out test roots.  Every method sees the same proxy rows.
        for seed in test_seeds:
            values = {
                policy_id: evaluate_unit(
                    cohort="test",
                    seed=seed,
                    policy_id=policy_id,
                    policy=policy,
                    phase="proxy",
                    method="Proxy Only",
                    budget=0,
                    config_hash=decision_config_hash,
                )
                for policy_id, policy in policies.items()
            }
            available_actions = _assert_actions(values)
            rows = _proxy_rows(values, policies)
            proxy_rows.extend({"seed": seed, **row} for row in rows)
            for method in METHODS:
                method_budgets = (ALL_HF_BUDGET,) if method == "All-HF Oracle" else BUDGETS
                for budget in method_budgets:
                    posterior = GaussianPosterior(rows, model) if method == "Calibrated PIVOT-KG" else None
                    rng = _seed_rng(seed, method)
                    observed: dict[str, float] = {}
                    query_specs: list[tuple[str, float | None]] = []
                    actor_values: dict[str, float] = {}
                    incumbent_actor: float | None = None
                    query_count = ALL_HF_BUDGET if method == "All-HF Oracle" else budget
                    for query_index in range(min(query_count, len(rows))):
                        if method == "Calibrated PIVOT-KG":
                            if posterior is None:
                                raise AssertionError("calibrated method requires a posterior")
                            candidate_id, evsi = choose_next(posterior, rng, fantasies=32, posterior_samples=64)
                        elif method == "Uniform HF":
                            if query_index == 0:
                                ids = [str(row["candidate_id"]) for row in rows]
                                rng.shuffle(ids)
                                query_specs = [(candidate, None) for candidate in ids[: min(budget, len(ids))]]
                            candidate_id, evsi = query_specs[query_index]
                        else:
                            candidate_id = str(rows[query_index]["candidate_id"])
                            evsi = None
                        if incumbent_actor is None:
                            incumbent_result = evaluate_unit(
                                cohort="test", seed=seed, policy_id="incumbent", policy=policies["incumbent"],
                                phase="hf_query", method=method, budget=budget, config_hash=decision_config_hash,
                            )
                            incumbent_actor = float(incumbent_result["value"])
                        candidate_result = evaluate_unit(
                            cohort="test", seed=seed, policy_id=candidate_id, policy=policies[candidate_id],
                            phase="hf_query", method=method, budget=budget, config_hash=decision_config_hash,
                        )
                        delta = float(candidate_result["value"]) - incumbent_actor
                        observed[candidate_id] = delta
                        actor_values[candidate_id] = delta
                        if posterior is not None:
                            before = posterior.value(candidate_id)
                            posterior.observe(candidate_id, delta)
                            after = posterior.value(candidate_id)
                        else:
                            before = float(next(row["delta_proxy"] for row in rows if row["candidate_id"] == candidate_id))
                            after = delta
                        query_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "budget": budget,
                                "query_index": query_index,
                                "candidate_id": candidate_id,
                                "paired_delta": delta,
                                "posterior_before": before,
                                "posterior_after": after,
                                "EVSI": evsi,
                                "logical_hf_query": True,
                                "physical_pair_evaluation": True,
                                "physical_simulator_calls": 2,
                                "outcome_chasing": False,
                                "initial_available_actions": list(available_actions),
                            }
                        )
                    if posterior is not None:
                        selected_id = posterior.select()
                    else:
                        selected_id = _select_proxy(rows, observed)
                    # Audit unqueried actor values only after the selection is sealed.
                    for candidate_id, policy in policies.items():
                        if candidate_id == "incumbent":
                            continue
                        if incumbent_actor is None:
                            incumbent_result = evaluate_unit(
                                cohort="test", seed=seed, policy_id="incumbent", policy=policies["incumbent"],
                                phase="audit", method=method, budget=budget, config_hash=decision_config_hash,
                            )
                            incumbent_actor = float(incumbent_result["value"])
                        if candidate_id not in actor_values:
                            candidate_result = evaluate_unit(
                                cohort="test", seed=seed, policy_id=candidate_id, policy=policy,
                                phase="audit", method=method, budget=budget, config_hash=decision_config_hash,
                            )
                            actor_values[candidate_id] = float(candidate_result["value"]) - incumbent_actor
                    truth_rows.extend(
                        {
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                            "candidate_id": candidate_id,
                            "paired_delta": value,
                            "audit_only": candidate_id not in observed,
                            "logical_hf_query": candidate_id in observed,
                            "outcome_chasing": False,
                        }
                        for candidate_id, value in sorted(actor_values.items())
                    )
                    true_best_id, actor_best = _truth_best(actor_values)
                    selected_value = 0.0 if selected_id == "incumbent" else actor_values[selected_id]
                    promotion_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                            "selected_candidate": selected_id,
                            "true_best_candidate": true_best_id,
                            "selected_actor_delta": selected_value,
                            "actor_best_delta": actor_best,
                            "ISR": actor_best - selected_value,
                            "hf_queries": len(observed),
                            "physical_query_simulator_calls": 2 * len(observed),
                            "candidate_count": len(rows),
                            "initial_available_actions": list(available_actions),
                            "outcome_chasing": False,
                        }
                    )
    except Exception as error:  # noqa: BLE001 - persist the gate failure
        failures.append({"error_type": type(error).__name__, "error": str(error)})
    finally:
        world.close()

    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "DEV_SMOKE" if args.smoke else "DEV_EXTERNAL_REDESIGN",
        "calibration_seed_count": len(calibration_seeds),
        "test_seed_count": len(test_seeds),
        "proxy_row_count": len(proxy_rows),
        "promotion_row_count": len(promotion_rows),
        "query_row_count": len(query_rows),
        "truth_audit_row_count": len(truth_rows),
        "failures": failures,
        "method_summaries": _method_summary(promotion_rows),
        "primary_contrast": _contrast(promotion_rows, budget=PRIMARY_BUDGET),
        "budget_contrasts": [_contrast(promotion_rows, budget=budget) for budget in BUDGETS],
        "audit_excluded_from_query_budget": True,
        "center_lane_action_panel": list(EXPECTED_ACTIONS),
        "action_hold_steps": 1,
        "promotion_status": "NOT_PROMOTED" if failures else "DEV_EVIDENCE_COMPLETE",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "cost-ledger.json",
        {"evaluation_journal": audit_evaluations(output / "evaluation-journal"), "wall_time": time.perf_counter() - started},
    )
    atomic_json(output / "proxy-rows.json", {"rows": proxy_rows})
    atomic_json(output / "promotion-results.json", {"rows": promotion_rows})
    atomic_json(output / "query-ledger.json", {"rows": query_rows})
    atomic_json(output / "truth-audit.json", {"rows": truth_rows})
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
