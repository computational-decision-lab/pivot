"""DEV revision experiments; online decisions cannot access offline actor outcomes."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v9.common import config_hash, initial_policy, stable_seed
from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    score_pivot_voi,
    select_pivot_voi,
    should_stop,
)
from pivot.algorithms.pivot import RoundResult, run_pivot_round_sequential
from pivot.core.policy import Policy
from pivot.v9.environments import environment_for
from pivot.v9.operators import generate_candidate_batch, policy_distance

from .evaluation_journal import EvaluationJournal, audit_evaluations

METHODS = ("pivot_voi", "proxy_only", "random_hf", "all_hf")
REGISTERED_METHODS = (*METHODS, "pivot_voi_batch", "pivot_voi_no_update")


def make_world(name: str, options: dict[str, Any]) -> Any:
    if name in {"controlled", "performative_control", "congestion_resource"}:
        return environment_for(
            "performative_control" if name == "controlled" else name,
            options.get("response_strength", 0.8),
        )
    if name == "openspiel":
        from pivot.environments.revision_openspiel import OpenSpielKuhnWorld

        return OpenSpielKuhnWorld(**options)
    if name == "physical":
        from pivot.environments.revision_physical import (
            MetaDrivePhysicalConfig,
            MetaDrivePhysicalWorld,
        )

        return MetaDrivePhysicalWorld(MetaDrivePhysicalConfig(**options))
    if name == "mpe2":
        from pivot.environments.revision_mpe2 import RevisionMPE2Config, RevisionMPE2World

        return RevisionMPE2World(RevisionMPE2Config(**options))
    raise ValueError(f"unsupported revision environment: {name}")


PHASES = ("calibration", "proxy", "online_hf", "offline_oracle")


class MeasuredEvaluator:
    """Charge every actual world.evaluate call to its phase, never query counts."""

    def __init__(self, world: Any, journal: EvaluationJournal | None = None):
        self.world = world
        self.journal = journal
        self.events: list[dict[str, Any]] = []
        self.calibration_pair_costs: list[float] = []

    def query_cost(self) -> float:
        declared = getattr(self.world, "paired_query_cost", None)
        if declared is not None:
            return float(declared() if callable(declared) else declared)
        if self.calibration_pair_costs:
            return float(np.mean(self.calibration_pair_costs))
        # Calibration rows are never acquired; this estimate is replaced before online use.
        return 1.0

    def evaluate(
        self, policy: Policy, *, seed: int, mode: str, phase: str, method: str, round_id: int
    ) -> Any:
        started = time.perf_counter()
        evaluate = lambda: self.world.evaluate(policy, seed=seed, mode=mode)
        result = (
            self.journal.evaluate(
                evaluate,
                phase=phase,
                method=method,
                round_id=round_id,
                policy_id=policy.policy_id,
                seed=seed,
                mode=mode,
            )
            if self.journal is not None
            else evaluate()
        )
        self.events.append(
            {
                "phase": phase,
                "method": method,
                "round_id": round_id,
                "policy_id": policy.policy_id,
                "seed": seed,
                "mode": mode,
                "environment_steps": int(result.environment_steps),
                "simulator_calls": int(result.simulator_calls),
                "compute_cost": result.compute_cost,
                "wall_time": time.perf_counter() - started,
                "metadata": dict(result.metadata),
            }
        )
        return result

    def pair(self, incumbent: Policy, candidate: Policy, **kwargs: Any) -> dict[str, Any]:
        left = self.evaluate(incumbent, mode="actor", **kwargs)
        right = self.evaluate(candidate, mode="actor", **kwargs)
        if kwargs["phase"] == "calibration":
            self.calibration_pair_costs.append(
                float(left.environment_steps + right.environment_steps)
            )
        return {
            "delta_true": float(right.value - left.value),
            "hf_query_cost": float(left.environment_steps + right.environment_steps),
            "environment_steps": left.environment_steps + right.environment_steps,
            "simulator_calls": left.simulator_calls + right.simulator_calls,
        }


def _build_batch(
    evaluator: MeasuredEvaluator,
    incumbent: Policy,
    *,
    seed: int,
    round_id: int,
    count: int,
    phase: str,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Policy]]:
    transitions = generate_candidate_batch(
        incumbent,
        family="gradient_informed",
        shift_level=0.7,
        count=count,
        seed=seed,
        round_id=round_id,
        config_id="revision-dev-v2",
    )
    baseline = evaluator.evaluate(
        incumbent, seed=seed, mode="observer", phase=phase, method=method, round_id=round_id
    )
    rows: list[dict[str, Any]] = []
    policies: dict[str, Policy] = {}
    for transition in transitions:
        candidate = transition.candidate
        value = evaluator.evaluate(
            candidate, seed=seed, mode="observer", phase=phase, method=method, round_id=round_id
        )
        delta = float(value.value - baseline.value)
        distance = policy_distance(incumbent, candidate)
        identifier = transition.transition_id
        # Explicit allowlist: neither actor values nor oracle records enter selection.
        rows.append(
            {
                "transition_id": identifier,
                "candidate_policy_id": candidate.policy_id,
                "incumbent_policy_id": incumbent.policy_id,
                "delta_proxy": delta,
                "features": [1.0, delta, distance, float(incumbent.parameters["intensity"])],
                "query_cost": evaluator.query_cost(),
                "hf_query_cost": evaluator.query_cost(),
                "query_cost_source": "declared_environment_steps_upper_bound"
                if getattr(evaluator.world, "paired_query_cost_is_upper_bound", False)
                else "declared_environment_steps"
                if hasattr(evaluator.world, "paired_query_cost")
                else "calibration_mean_paired_environment_steps",
                "query_cost_unit": getattr(
                    evaluator.world, "paired_query_cost_unit", "environment_steps"
                ),
                "query_cost_is_upper_bound": bool(
                    getattr(evaluator.world, "paired_query_cost_is_upper_bound", False)
                ),
                "seed": seed,
            }
        )
        policies[identifier] = candidate
    return rows, policies


def _calibrate(evaluator: MeasuredEvaluator, seed: int, count: int) -> BayesianLinearDeltaPosterior:
    incumbent = initial_policy()
    rows, policies = _build_batch(
        evaluator,
        incumbent,
        seed=seed,
        round_id=-1,
        count=count,
        phase="calibration",
        method="shared",
    )
    corrections = []
    for row in rows:
        sample = evaluator.pair(
            incumbent,
            policies[row["transition_id"]],
            seed=seed,
            phase="calibration",
            method="shared",
            round_id=-1,
        )
        corrections.append(sample["delta_true"] - row["delta_proxy"])
    return BayesianLinearDeltaPosterior(
        prior_precision=1.0, noise_variance=max(float(np.var(corrections)), 1e-4)
    ).fit(np.asarray([row["features"] for row in rows]), np.asarray(corrections))


def _random_selector(seed: int) -> Callable[..., list[str]]:
    rng = np.random.default_rng(seed)

    def select(rows: list[dict[str, Any]], budget: int, **_: Any) -> list[str]:
        choices = sorted(str(row["transition_id"]) for row in rows)
        return [] if not choices or budget <= 0 else [choices[int(rng.integers(len(choices)))]]

    return select


def _canonical_selector(rows: list[dict[str, Any]], budget: int, **_: Any) -> list[str]:
    return sorted(str(row["transition_id"]) for row in rows)[:budget]


def _run_method(
    method: str,
    *,
    candidates: list[dict[str, Any]],
    incumbent: Policy,
    posterior: BayesianLinearDeltaPosterior,
    hf: Callable[..., Any],
    seed: int,
    budget: int,
) -> RoundResult:
    if method == "pivot_voi_batch":
        from .controls import run_batch

        return run_batch(
            incumbent=incumbent,
            candidates=candidates,
            posterior=posterior,
            hf=hf,
            seed=seed,
            budget=budget,
        )
    is_voi = method in {"pivot_voi", "pivot_voi_no_update"}
    model = posterior if is_voi else None
    acquisition = (
        select_pivot_voi
        if is_voi
        else (_random_selector(seed) if method == "random_hf" else _canonical_selector)
    )
    limit = 0 if method == "proxy_only" else len(candidates) if method == "all_hf" else budget

    def voi_stop(rows: list[dict[str, Any]], model: Any, **_: Any) -> tuple[bool, str | None]:
        available = [
            row for row in rows if not row.get("is_incumbent") and not row.get("hf_queried")
        ]
        if not available:
            return True, "candidate_exhausted"
        scores = score_pivot_voi(
            available, model, seed=seed, fantasies=8, posterior_samples=32, decision_candidates=rows
        )
        return should_stop(
            selection_probability=float(scores[0]["selection_probability_lower"]),
            max_acquisition=max(float(item["acquisition_upper"]) for item in scores),
            delta=0.05,
            eta=0.0,
        )

    return run_pivot_round_sequential(
        incumbent=incumbent,
        candidates=copy.deepcopy(candidates),
        proxy=None,
        hf=hf,
        acquisition=acquisition,
        budget=limit,
        model=model,
        acquisition_kwargs={"seed": seed, "fantasies": 8, "posterior_samples": 32}
        if is_voi
        else {},
        acquisition_method=method,
        posterior_version="calibrated-linear" if model is not None else "proxy",
        score=score_pivot_voi if is_voi else None,
        stop=voi_stop if is_voi else None,
        update_posterior=method != "pivot_voi_no_update",
    )


def _ledger(
    events: list[dict[str, Any]], results: list[dict[str, Any]], elapsed: float
) -> dict[str, Any]:
    phases = {}
    for phase in PHASES:
        selected = [event for event in events if event["phase"] == phase]
        phases[phase] = {
            "environment_steps": sum(e["environment_steps"] for e in selected),
            "simulator_calls": sum(e["simulator_calls"] for e in selected),
            "wall_time": sum(e["wall_time"] for e in selected),
            "compute_cost_complete": all(
                e["compute_cost"] is not None and e["metadata"].get("compute_cost_complete", True)
                for e in selected
            ),
            "compute_cost": sum(e["compute_cost"] for e in selected)
            if all(e["compute_cost"] is not None for e in selected)
            else None,
        }
    return {
        "phases": phases,
        "evaluations": events,
        "wall_time": elapsed,
        "hf_query_count": sum(r["query_count"] for r in results),
        "hf_cost": sum(r["hf_cost"] for r in results),
        "cost_unit": "environment_steps",
        "environment_steps": sum(e["environment_steps"] for e in events),
        "simulator_calls": sum(e["simulator_calls"] for e in events),
        "failed_queries": 0,
        "retried_queries": 0,
    }


def _execute(config: dict[str, Any], *, journal_root: Path | None = None) -> dict[str, Any]:
    evaluator = MeasuredEvaluator(
        make_world(config["environment"], config["environment_options"]),
        EvaluationJournal(journal_root) if journal_root is not None else None,
    )
    try:
        return _execute_world(config, evaluator)
    finally:
        close = getattr(evaluator.world, "close", None)
        if close is not None:
            close()


def _execute_world(config: dict[str, Any], evaluator: MeasuredEvaluator) -> dict[str, Any]:
    started = time.perf_counter()
    experiment, seed = config["experiment"], config["seed"]
    # Domain-separated seeds isolate calibration from evaluation and retain CRN fairness.
    calibration_seed = stable_seed("revision", seed, "calibration")
    evaluation_seeds = [
        stable_seed("revision", seed, "evaluation", r) for r in range(config["rounds"])
    ]
    if calibration_seed in evaluation_seeds:
        raise RuntimeError("calibration/evaluation seed collision")
    initial_posterior = _calibrate(evaluator, calibration_seed, config["candidate_count"])
    results, traces, oracle_records = [], [], []
    shared_batch = None
    if experiment == "e5c":
        shared_batch = _build_batch(
            evaluator,
            initial_policy(),
            seed=evaluation_seeds[0],
            round_id=0,
            count=config["candidate_count"],
            phase="proxy",
            method="shared",
        )
    for method in config["methods"]:
        incumbent = initial_policy()
        posterior = copy.deepcopy(initial_posterior)
        for round_id, evaluation_seed in enumerate(evaluation_seeds):
            rows, policies = (
                shared_batch
                if shared_batch is not None
                else _build_batch(
                    evaluator,
                    incumbent,
                    seed=evaluation_seed,
                    round_id=round_id,
                    count=config["candidate_count"],
                    phase="proxy",
                    method=method,
                )
            )

            def hf(
                row: dict[str, Any],
                incumbent: Policy = incumbent,
                policies: dict[str, Policy] = policies,
                evaluation_seed: int = evaluation_seed,
                method: str = method,
                round_id: int = round_id,
            ) -> dict[str, Any]:
                return evaluator.pair(
                    incumbent,
                    policies[str(row["transition_id"])],
                    seed=evaluation_seed,
                    phase="online_hf",
                    method=method,
                    round_id=round_id,
                )

            result = _run_method(
                method,
                candidates=rows,
                incumbent=incumbent,
                posterior=posterior,
                hf=hf,
                seed=stable_seed(
                    seed, "voi_controls" if method.startswith("pivot_voi") else method, round_id
                ),
                budget=config["budget"],
            )
            # Offline audit is constructed only AFTER the online decision has completed.
            oracle = {"incumbent": 0.0}
            for identifier, policy in policies.items():
                oracle[identifier] = evaluator.pair(
                    incumbent,
                    policy,
                    seed=evaluation_seed,
                    phase="offline_oracle",
                    method=method,
                    round_id=round_id,
                )["delta_true"]
            selected_id = result.selected_candidate_id
            selected = incumbent if selected_id == "incumbent" else policies[selected_id]
            actual_trace = [dict(event) for event in result.audit_trace]
            if not actual_trace:
                raise RuntimeError("sequential runner did not provide an actual audit_trace")
            record = {
                "method": method,
                "round_id": round_id,
                "evaluation_seed": evaluation_seed,
                "candidate_set_hash": config_hash(rows),
                "observable_candidates": copy.deepcopy(rows),
                "incumbent_policy": incumbent.to_record(),
                "selected_policy": selected.to_record(),
                "selected_candidate_id": selected_id,
                "selected_delta_estimate": result.selected_delta_estimate,
                "selected_delta_true_observed": result.selected_delta_true,
                "selected_delta_true_offline": oracle[selected_id],
                "update_selection_regret_offline": max(oracle.values()) - oracle[selected_id],
                "queried_ids": list(result.queried_ids),
                "query_count": result.query_count,
                "hf_budget": 0
                if method == "proxy_only"
                else len(rows)
                if method == "all_hf"
                else config["budget"],
                "hf_cost": result.hf_cost,
                "rows": [dict(row) for row in result.rows],
                "audit_trace": actual_trace,
                "stop_reason": result.stop_reason,
                "posterior_observations_before": posterior.n_observations
                if method.startswith("pivot_voi")
                else None,
                "posterior_observations_after": result.posterior.n_observations
                if method.startswith("pivot_voi")
                else None,
            }
            results.append(record)
            traces.extend(
                {"method": method, "round_id": round_id, **event} for event in actual_trace
            )
            oracle_records.append({"method": method, "round_id": round_id, "outcomes": oracle})
            incumbent = selected
            if method.startswith("pivot_voi"):
                posterior = result.posterior
    return {
        "experiment": f"{experiment.upper()}-revision",
        "status": "DEV",
        "protocol_id": config["protocol_id"],
        "config_hash": config_hash(config),
        "environment": config["environment"],
        "methods": config["methods"],
        "results": results,
        "audit_trace": traces,
        "offline_oracle": oracle_records,
        "seed_contract": {
            "calibration_seed": calibration_seed,
            "evaluation_seeds": evaluation_seeds,
        },
        "cost_ledger": _ledger(evaluator.events, results, time.perf_counter() - started),
    }


def _content_hash(paths: Sequence[Path], root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(str(path.resolve().relative_to(root.resolve())).encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and never replaces a concurrent writer.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
    finally:
        os.unlink(temporary)


def _recover_corrupt_publication(output: Path, *, raw_valid: bool) -> None:
    manifest_path = output / "manifest.json"
    names = [
        "manifest.json",
        "raw_results.json",
        "audit_trace.json",
        "cost_ledger.json",
        "DEV_SUMMARY.md",
        "config.json",
        "dependencies.json",
        "evaluation_audit.json",
    ]
    try:
        existing = json.loads(manifest_path.read_text())
        intact = all(
            (output / name).is_file()
            and hashlib.sha256((output / name).read_bytes()).hexdigest() == artifact["sha256"]
            for name, artifact in existing["files"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        intact = False
    if intact:
        return
    if not any((output / name).exists() for name in names):
        return
    recovery = output / "recovery" / uuid.uuid4().hex
    recovery.mkdir(parents=True)
    # Preserve the entire old publication, including corrupt bytes, before retry.
    for name in names:
        if name == "raw_results.json" and raw_valid:
            continue
        path = output / name
        if path.exists():
            os.replace(path, recovery / name)
    _write_new_or_identical(
        recovery / "reason.json",
        json.dumps({"reason": "same_identity_corrupt_publication"}).encode(),
    )


def run_revision_smoke(
    experiment: str,
    *,
    root: Path,
    output: Path,
    seed: int = 1,
    methods: Sequence[str] = METHODS,
    rounds: int | None = None,
    resume: bool = False,
    environment: str = "controlled",
    environment_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a measured DEV seed unit. Resume only a checksum-valid checkpoint."""
    if experiment not in {"e3c", "e5c"}:
        raise ValueError("experiment must be e3c or e5c")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(REGISTERED_METHODS):
        raise ValueError("methods must be unique registered revision methods")
    count_rounds = (2 if experiment == "e3c" else 1) if rounds is None else rounds
    if (experiment == "e3c" and count_rounds < 2) or (experiment == "e5c" and count_rounds != 1):
        raise ValueError("E3C requires >=2 rounds; E5C requires exactly one fixed-batch round")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite revision run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": f"pivot-revision-{experiment}-dev-v2",
        "experiment": experiment,
        "seed": seed,
        "methods": list(methods),
        "rounds": count_rounds,
        "candidate_count": 4,
        "budget": 2,
        "environment": environment,
        "environment_options": dict(environment_options or {}),
        "response_strength": 0.8,
        "operator_family": "gradient_informed",
        "operator_shift": 0.7,
        "fantasies": 8,
        "posterior_samples": 32,
        "delta": 0.05,
        "eta": 0.0,
    }
    from pivot.runner.checkpoint import CheckpointStore

    code_files = [
        *sorted((root / "src/pivot").rglob("*.py")),
        *sorted((root / "experiments/revision").glob("*.py")),
        root / "experiments/v9/common.py",
    ]
    dependency_files = [p for p in (root / "uv.lock", root / "pyproject.toml") if p.exists()]
    dependencies = {
        "python": platform.python_version(),
        "installed": dict(
            sorted(
                (dist.metadata["Name"].lower(), dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
        "declaration_hash": _content_hash(dependency_files, root),
    }
    identity = {
        "config_hash": config_hash(config),
        "code_hash": _content_hash(code_files, root),
        "dependency_hash": config_hash(dependencies),
        "git_commit": _git_commit(root),
    }
    identity_path = output / "run_identity.json"
    # Seal identity before executing; changed inputs require a new output directory.
    _write_new_or_identical(identity_path, json.dumps(identity, sort_keys=True, indent=2).encode())
    store = CheckpointStore(output / "checkpoints", git_commit=_git_commit(root))
    raw_valid = store.is_valid(
        f"{experiment}:seed:{seed}", output_path=output / "raw_results.json", **identity
    )
    _recover_corrupt_publication(output, raw_valid=raw_valid)
    record = store.run_unit(
        f"{experiment}:seed:{seed}",
        lambda: json.dumps(
            _execute(config, journal_root=output / "evaluation_journal"), sort_keys=True, indent=2
        ),
        output_path=output / "raw_results.json",
        config_hash=config_hash(config),
        code_hash=identity["code_hash"],
        dependency_hash=identity["dependency_hash"],
        git_commit=_git_commit(root),
    )
    if record["status"] != "completed":
        raise RuntimeError(f"revision unit failed; see checkpoint manifest: {record}")
    raw = json.loads((output / "raw_results.json").read_text())
    summary = (
        f"# {experiment.upper()} revision DEV\n\n"
        f"{len(methods)} methods; {count_rounds} rounds per method; seed {seed}.\n\n"
        "Implementation smoke only; no powered scientific claim. E3C feeds the selected Policy "
        "and PIVOT posterior into the next round. E5C shares one fixed observable candidate set. "
        "Online HF uses paired actor rollouts; offline oracle results are computed after selection.\n\n"
        "Costs separately record calibration, proxy, online HF, and offline oracle calls and steps. "
        "Wall time is measured locally; monetary compute cost is unavailable. Random HF and all-HF "
        "use observed candidate deltas plus proxy predictions for unqueried candidates. No LUCB claim.\n"
    )
    artifacts = {
        "audit_trace.json": json.dumps(raw["audit_trace"], sort_keys=True, indent=2).encode(),
        "cost_ledger.json": json.dumps(raw["cost_ledger"], sort_keys=True, indent=2).encode(),
        "DEV_SUMMARY.md": summary.encode(),
        "config.json": json.dumps(config, sort_keys=True, indent=2).encode(),
        "dependencies.json": json.dumps(dependencies, sort_keys=True, indent=2).encode(),
        "evaluation_audit.json": json.dumps(
            audit_evaluations(output / "evaluation_journal"), sort_keys=True, indent=2
        ).encode(),
    }
    for name, payload in artifacts.items():
        _write_new_or_identical(output / name, payload)
    manifest = {
        "protocol_id": config["protocol_id"],
        "status": "DEV",
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(root),
        "config_hash": config_hash(config),
        "code_hash": record["code_hash"],
        "dependency_hash": record["dependency_hash"],
        "files": {},
    }
    for name in ["raw_results.json", "run_identity.json", *artifacts]:
        payload = (output / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    _write_new_or_identical(
        output / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode()
    )
    return {**raw, "manifest": manifest}


def _git_commit(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
