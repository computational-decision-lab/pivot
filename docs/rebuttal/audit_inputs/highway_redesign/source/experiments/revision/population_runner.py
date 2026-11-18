"""Recoverable DEV population experiments, with paid actor data as the only HF input.

Calibration and observer preparation are external, frozen inputs. Native rewards
are never clipped. Gaussian stopping, when requested, is model-conditional only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from pivot.core.policy import Policy
from pivot.runner.checkpoint import CheckpointStore

from .absolute_features import AbsoluteFeaturePrior
from .evaluation_journal import audit_evaluations
from .lucb import LUCBConfig, PairedLUCB1
from .observations import CallerContext, PairedObservationService, QueryKey, _hash, _json, _seal
from .population import GaussianPopulation, PopulationObservation, QueryAction, score_queries
from .population_features import DeltaFeaturePrior, ObservableTransition
from .population_stopping import PosteriorStopRule, decision_bounds

METHODS = (
    "PIVOT",
    "GlobalPairedKG",
    "RandomHF",
    "ProxyOnly",
    "UniformHF",
    "AllHF",
    "PIVOTBatch",
    "PIVOTNoUpdate",
    "PairedLUCB",
)


def _installed_dependencies() -> dict[str, str]:
    """Record the effective version for each normalized distribution name.

    ``distributions()`` may expose duplicate system/user installations in an
    order that differs from ``version()`` resolution. Resolve every discovered
    name through the same API used by verification so capture cannot freeze an
    inactive duplicate.
    """
    names = {
        re.sub(r"[-_.]+", "-", name).lower()
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata["Name"])
    }
    return {name: importlib.metadata.version(name) for name in sorted(names)}


@dataclass(frozen=True)
class FrozenInputs:
    root: Path
    git_commit: str
    files: Mapping[str, str]
    dependencies: Mapping[str, str]

    @classmethod
    def capture(cls, root: Path) -> FrozenInputs:
        """Snapshot actual bytes/installed versions; callers use detached code for runs.

        Capturing a snapshot is not a calibration or formal-run readiness gate.
        Tests may capture uncommitted source; every use still verifies the bytes.
        """
        root = Path(root).resolve()
        files = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("src", "experiments", "scripts", "tests")
            for path in sorted((root / directory).rglob("*.py"))
        }
        files["pyproject.toml"] = hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
        dependencies = _installed_dependencies()
        dependencies["__python__"] = platform.python_version()
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        return cls(root, commit, files, dependencies)

    @property
    def source_hash(self) -> str:
        return _hash(dict(self.files))

    @property
    def dependency_hash(self) -> str:
        return _hash(dict(self.dependencies))

    def verify(self, *callables: Any) -> None:
        current = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != self.git_commit:
            raise ValueError("source HEAD changed from frozen identity")
        for relative, digest in self.files.items():
            path = (self.root / relative).resolve()
            if (
                not path.is_relative_to(self.root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"source bytes changed: {relative}")
        for name, version in self.dependencies.items():
            actual = (
                platform.python_version()
                if name == "__python__"
                else importlib.metadata.version(name)
            )
            if actual != version:
                raise ValueError(f"installed dependency changed: {name}")
        paths = [inspect.getsourcefile(value) for value in callables]
        paths.extend(
            getattr(module, "__file__", None)
            for name, module in tuple(sys.modules.items())
            if name.startswith(("pivot.", "experiments.revision."))
        )
        for filename in paths:
            if filename is None:
                raise ValueError("callable source is unavailable for identity verification")
            path = Path(filename).resolve()
            if (
                not path.is_relative_to(self.root)
                or str(path.relative_to(self.root)) not in self.files
            ):
                raise ValueError(f"imported/callback source is outside frozen source map: {path}")


@dataclass(frozen=True)
class CandidatePool:
    rows: tuple[ObservableTransition, ...]
    policies: Mapping[str, Policy]

    def __post_init__(self) -> None:
        if not self.rows or any(not isinstance(row, ObservableTransition) for row in self.rows):
            raise TypeError("only typed observable transitions are accepted")
        ids = [row.candidate_id for row in self.rows]
        if len(set(ids)) != len(ids) or set(ids) != set(self.policies):
            raise ValueError("candidate rows and policy mapping must match uniquely")
        if any(not isinstance(policy, Policy) for policy in self.policies.values()):
            raise TypeError("candidate mapping must contain Policies")

    def record(self) -> dict[str, Any]:
        return {
            "rows": [asdict(row) for row in self.rows],
            "policies": {key: policy.to_record() for key, policy in self.policies.items()},
        }


@dataclass(frozen=True)
class RoundCalibration:
    absolute_prior: GaussianPopulation
    paired_noise: Mapping[str, Any]
    persistent_variance: float
    evidence: Mapping[str, Any]
    preparation_costs: Mapping[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        if not self.evidence:
            raise ValueError("calibration provenance/evidence is required")
        return {
            "absolute_prior": _model(self.absolute_prior),
            "paired_noise": {
                key: np.asarray(value).tolist() for key, value in self.paired_noise.items()
            },
            "persistent_variance": self.persistent_variance,
            "evidence": dict(self.evidence),
            "preparation_costs": self.preparation_costs,
        }


@dataclass(frozen=True)
class SourcedDeltaBounds:
    lower: float
    upper: float
    source: str
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("LUCB requires a source for true native paired-delta bounds")
        LUCBConfig(("validation",), self.epsilon, self.delta, self.lower, self.upper, 0, 1)


@dataclass(frozen=True)
class PopulationProtocol:
    protocol_id: str
    instance_id: str
    experiment: str
    methods: tuple[str, ...]
    rounds: int
    max_queries: int
    round_cost_budget: float
    offline_replicates: int = 0
    seed: int = 11
    quadrature_order: int = 24
    lucb_bounds: SourcedDeltaBounds | None = None
    stopping: PosteriorStopRule | None = None

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.instance_id:
            raise ValueError("protocol and instance identifiers required")
        if self.experiment not in {"E3C", "E5C"}:
            raise ValueError("experiment must be E3C or E5C")
        if (self.experiment == "E5C" and self.rounds != 1) or (
            self.experiment == "E3C" and self.rounds < 2
        ):
            raise ValueError("E5C has one fixed round; E3C requires at least two rounds")
        for name in ("rounds", "max_queries", "offline_replicates", "quadrature_order", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid integer: {name}")
        if (
            self.quadrature_order < 2
            or not math.isfinite(self.round_cost_budget)
            or self.round_cost_budget < 0
        ):
            raise ValueError("invalid quadrature or cost budget")
        if (
            not self.methods
            or len(set(self.methods)) != len(self.methods)
            or set(self.methods) - set(METHODS)
        ):
            raise ValueError("unknown or duplicate method")


def _model(model: GaussianPopulation) -> dict[str, Any]:
    return {
        "candidate_ids": list(model.candidate_ids),
        "mean": model.mean.tolist(),
        "covariance": model.covariance.tolist(),
        "decision_matrix": model.decision_matrix.tolist(),
        "representation": model.representation,
        "model_spec": model.model_spec,
        "version": model.version,
        "observation_ids": list(model.observation_ids),
        "queried_candidates": list(model.queried_candidates),
        "lineage_id": model.lineage_id,
    }


def _feature(prior: DeltaFeaturePrior) -> dict[str, Any]:
    return {
        "mean": prior.mean.tolist(),
        "covariance": prior.covariance.tolist(),
        "model_spec": prior.model_spec,
        "version": prior.version,
        "observation_ids": list(prior.observation_ids),
    }


def _absolute_feature(prior: AbsoluteFeaturePrior | None) -> dict[str, Any] | None:
    if prior is None:
        return None
    return {
        "mean": prior.mean.tolist(),
        "covariance": prior.covariance.tolist(),
        "model_spec": prior.model_spec,
        "version": prior.version,
        "observation_ids": list(prior.observation_ids),
        "lineage_id": prior.lineage_id,
    }


def _policy_parameter_hash(policy: Policy) -> str:
    """Canonical behavior fingerprint for worlds parameterized by Policy.parameters."""
    return _hash(dict(policy.parameters))


def _exact_selected_value(
    selected_candidate_id: str,
    purchases: list[dict[str, Any]],
    prior: AbsoluteFeaturePrior,
) -> tuple[float | None, str | None]:
    """Return an exact selected value only from explicit deterministic evidence."""
    index = 0 if selected_candidate_id == "incumbent" else 1
    values = [
        float(purchase["raw"]["absolute_mean"][index])
        for purchase in purchases
        if purchase["raw"]["is_exact"] is True
        and (
            selected_candidate_id == "incumbent"
            or purchase["candidate_id"] == selected_candidate_id
        )
    ]
    if values:
        if any(value != values[0] for value in values[1:]):
            raise ValueError("conflicting exact selected values in paired purchases")
        return values[0], "current_round_exact_purchase"
    if (
        selected_candidate_id == "incumbent"
        and prior.covariance[-1, -1] == 0
        and np.all(prior.covariance[-1, :] == 0)
        and np.all(prior.covariance[:, -1] == 0)
    ):
        return float(prior.mean[-1]), "carried_exact_incumbent_state"
    return None, None


def _costs(raws: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [raw["costs"] for raw in raws]
    return {
        "environment_steps": sum(cost["environment_steps"] for cost in costs),
        "simulator_calls": sum(cost["simulator_calls"] for cost in costs),
        "wall_time": sum(cost["wall_time"] for cost in costs),
        "compute_cost": sum(cost["compute_cost"] for cost in costs)
        if all(cost["compute_cost"] is not None for cost in costs)
        else None,
        "compute_cost_complete": all(cost["compute_cost_complete"] for cost in costs),
        "scope": "successful purchases; durable journals separately retain failed/unknown costs",
    }


def _unit(
    store: CheckpointStore,
    key: str,
    path: Path,
    worker: Callable[[], str],
    frozen: FrozenInputs,
    config: Any,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal(path.parent / (path.stem + ".identity.json"), config)
    record = store.run_unit(
        key,
        worker,
        output_path=path,
        config_hash=_hash(config),
        code_hash=frozen.source_hash,
        dependency_hash=frozen.dependency_hash,
        git_commit=frozen.git_commit,
    )
    if record["status"] != "completed":
        raise RuntimeError(f"population unit failed: {record.get('error_message')}")
    return json.loads(path.read_text())


def _round(
    *,
    service: PairedObservationService,
    protocol: PopulationProtocol,
    method: str,
    incumbent: Policy,
    round_id: int,
    pool: CandidatePool,
    calibration: RoundCalibration,
    feature_prior: DeltaFeaturePrior,
    absolute_feature_prior: AbsoluteFeaturePrior | None = None,
) -> dict[str, Any]:
    feature_round = feature_prior.for_round(
        pool.rows, persistent_variance=calibration.persistent_variance
    )
    ids = tuple(row.candidate_id for row in pool.rows)
    if (
        calibration.absolute_prior.candidate_ids != ids
        or calibration.absolute_prior.representation != "absolute"
    ):
        raise ValueError("absolute calibration prior differs from ordered candidate pool")
    if set(calibration.paired_noise) != set(ids):
        raise ValueError("per-arm paired likelihood is required for the entire candidate set")
    # QueryAction validates the full 2x2 likelihood even for delta methods.
    absolute_actions = [
        calibration.absolute_prior.absolute_pair_query(
            row.candidate_id,
            noise_covariance=calibration.paired_noise[row.candidate_id],
            cost=row.query_cost,
        )
        for row in pool.rows
    ]
    contrast = np.array([-1.0, 1.0])
    absolute_round = (
        absolute_feature_prior.for_round(
            pool.rows, persistent_variance=calibration.persistent_variance
        )
        if method == "GlobalPairedKG" and absolute_feature_prior is not None
        else None
    )
    actions = (
        [
            absolute_round.query(
                row.candidate_id,
                noise_covariance=calibration.paired_noise[row.candidate_id],
            )
            for row in pool.rows
        ]
        if absolute_round is not None
        else absolute_actions
        if method == "GlobalPairedKG"
        else [
            feature_round.query(
                action.candidate_id,
                noise_variance=float(contrast @ action.noise_covariance @ contrast),
            )
            for action in absolute_actions
        ]
    )
    model = (
        absolute_round.model
        if absolute_round is not None
        else calibration.absolute_prior
        if method == "GlobalPairedKG"
        else feature_round.model
    )
    initial_model = model
    lookup = {action.candidate_id: action for action in actions}
    trace: list[dict[str, Any]] = []
    purchases: list[dict[str, Any]] = []
    indices: Counter[str] = Counter()
    exact: set[str] = set()
    spent = 0.0
    condition_epoch = 0

    def record(event: str, **details: Any) -> None:
        trace.append(
            {
                "event": event,
                "query_count": len(purchases),
                "posterior_version": model.version,
                "conditioning_epoch": condition_epoch,
                "cumulative_hf_cost": spent,
                "confidence_certified": False,
                **details,
            }
        )

    def buy(candidate: str, replicate: int | None = None) -> PopulationObservation:
        nonlocal spent
        action = lookup[candidate]
        replicate = indices[candidate] if replicate is None else replicate
        if replicate != indices[candidate]:
            raise ValueError("nonmonotone repeat purchase index")
        record(
            "query", candidate_id=candidate, replicate_index=replicate, declared_cost=action.cost
        )
        expected_key = QueryKey(
            protocol.protocol_id, round_id, candidate, replicate, instance_id=protocol.instance_id
        )
        expected_context = CallerContext("online", method)
        raw = service.query(
            expected_key, incumbent, pool.policies[candidate], context=expected_context
        ).to_dict()
        # A resumed callback may only replay this exact durable purchase. Do not
        # allow a caller to substitute another candidate/replicate/method record.
        if raw["query_key"] != asdict(expected_key) or raw["context"] != asdict(expected_context):
            raise ValueError("observation callback returned the wrong query identity/context")
        actual = float(raw["costs"]["environment_steps"])
        if not math.isfinite(actual) or actual < 0 or actual > action.cost + 1e-12:
            raise ValueError("actual environment-step cost exceeds declared ceiling")
        if spent + actual > protocol.round_cost_budget + 1e-12:
            raise ValueError("online budget exceeded")
        values = tuple(raw["absolute_mean"]) if method == "GlobalPairedKG" else (raw["mean_delta"],)
        observation_id = _hash(
            {"method": method, "key": raw["query_key"], "identity": raw["identity"]}
        )
        observation = PopulationObservation(observation_id, values, actual)
        indices[candidate] += 1
        spent += actual
        purchases.append(
            {
                "observation_id": observation_id,
                "candidate_id": candidate,
                "replicate_index": replicate,
                "raw": raw,
            }
        )
        record(
            "observe",
            candidate_id=candidate,
            observation_id=observation_id,
            values=list(values),
            actual_cost=actual,
            is_exact=raw["is_exact"],
        )
        if raw["is_exact"]:
            exact.add(candidate)
            if np.any(action.noise_covariance):
                raise ValueError("exact adapter requires a zero-noise calibration likelihood")
        return observation

    def condition(observations: list[tuple[QueryAction, PopulationObservation]]) -> None:
        nonlocal model, condition_epoch
        before = model.version
        if method != "PIVOTNoUpdate":
            for action, observation in observations:
                model = model.condition(action, observation)
            condition_epoch += 1
        record(
            "condition",
            updated=method != "PIVOTNoUpdate",
            posterior_version_before=before,
            posterior_version_after=model.version,
            observation_ids=[o.observation_id for _, o in observations],
            scalar_updates=len(observations) if method != "PIVOTNoUpdate" else 0,
        )

    def rescore() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        available = [action for action in actions if action.candidate_id not in exact]
        scores = score_queries(model, available, quadrature_order=protocol.quadrature_order)
        bounds = decision_bounds(model)
        record("rescore", scores=scores, **bounds)
        return scores, bounds

    scores, bounds = rescore()
    reason = "query_budget"
    status = "completed"
    lucb_result = None
    selected: str | None = None
    if method == "ProxyOnly":
        selected = max(
            ("incumbent", *ids),
            key=lambda candidate: (
                0.0
                if candidate == "incumbent"
                else next(row.delta_proxy for row in pool.rows if row.candidate_id == candidate)
            ),
        )
        reason = "proxy_only"
    elif method == "PairedLUCB":
        source = protocol.lucb_bounds
        if source is None:
            raise ValueError("PairedLUCB bounds missing")
        if len({action.cost for action in actions}) != 1:
            raise ValueError("current LUCB1 contract requires equal per-arm query cost ceilings")
        state = PairedLUCB1(
            LUCBConfig(
                ids,
                source.epsilon,
                source.delta,
                source.lower,
                source.upper,
                protocol.round_cost_budget,
                actions[0].cost,
            )
        )
        while (reservation := state.ask()) is not None:
            if len(purchases) + len(reservation.requests) > protocol.max_queries:
                reason = "query_budget_before_complete_lucb_reservation"
                status = "budget_censored"
                break
            record("reserve", reservation=asdict(reservation))
            for request in reservation.requests:
                observation = buy(request.candidate_id, request.replicate_index)
                state.observe(
                    request,
                    delta=observation.values[0],
                    is_exact=purchases[-1]["raw"]["is_exact"],
                    cost=observation.actual_cost,
                )
            record("lucb_update", state=asdict(state.result()))
        lucb_result = asdict(state.result())
        selected = state.result().recommendation or "incumbent"
        if status != "budget_censored":
            status = state.result().status
            reason = state.result().reason or status
    elif method == "PIVOTBatch":
        # Static initial ranking, cycling it if repeated sampling budget remains.
        menu = []
        reserved = 0.0
        while len(menu) < protocol.max_queries:
            affordable = [
                row
                for row in scores
                if row["cost"] <= protocol.round_cost_budget - reserved + 1e-12
            ]
            if not affordable:
                break
            candidate = affordable[len(menu) % len(affordable)]["candidate_id"]
            menu.append(candidate)
            reserved += lookup[candidate].cost
        record("batch_plan", menu=menu, reserved_cost=reserved)
        pending = []
        for candidate in menu:
            if candidate not in exact:
                pending.append((lookup[candidate], buy(candidate)))
        if pending:
            condition(pending)
        scores, bounds = rescore()
        reason = "batch_menu_complete"
    else:
        rng = np.random.default_rng(protocol.seed + round_id)
        while len(purchases) < protocol.max_queries:
            available = [
                row for row in scores if row["cost"] <= protocol.round_cost_budget - spent + 1e-12
            ]
            if not available:
                reason = "cost_budget" if scores else "all_candidates_exact"
                break
            if (
                method in {"PIVOT", "GlobalPairedKG"}
                and protocol.stopping is not None
                and protocol.stopping.satisfied(bounds, actions)
            ):
                reason = "model_conditional_probability_and_voi"
                break
            if method == "RandomHF":
                candidate = available[int(rng.integers(len(available)))]["candidate_id"]
            elif method in {"UniformHF", "AllHF"}:
                candidate = min(
                    (row["candidate_id"] for row in available),
                    key=lambda key: (indices[key], ids.index(key)),
                )
            else:
                candidate = available[0]["candidate_id"]
            condition([(lookup[candidate], buy(candidate))])
            scores, bounds = rescore()
    selected = selected or model.selected_candidate_id
    record("stop", stop_reason=reason, status=status, **decision_bounds(model))
    record("select", candidate_id=selected)
    next_prior = (
        feature_prior
        if method in {"GlobalPairedKG", "ProxyOnly", "PairedLUCB"}
        else feature_round.next_prior(model)
    )
    exact_selected_value: float | None = None
    exact_selected_value_source: str | None = None
    if method == "GlobalPairedKG" and absolute_round is not None:
        exact_selected_value, exact_selected_value_source = _exact_selected_value(
            selected, purchases, absolute_feature_prior
        )
        next_absolute_prior = absolute_round.next_prior(
            model,
            selected,
            exact_selected_value=(
                exact_selected_value
                if exact_selected_value_source == "current_round_exact_purchase"
                else None
            ),
        )
    else:
        next_absolute_prior = absolute_feature_prior
    return {
        "status": status,
        "method": method,
        "round_id": round_id,
        "incumbent_policy": incumbent.to_record(),
        "selected_candidate_id": selected,
        "selected_policy": (
            incumbent if selected == "incumbent" else pool.policies[selected]
        ).to_record(),
        "candidate_pool": pool.record(),
        "candidate_pool_hash": _hash(pool.record()),
        "calibration": calibration.record(),
        "feature_prior_before": _feature(feature_prior),
        "feature_prior_after": _feature(next_prior),
        **(
            {
                "absolute_prior_before": _absolute_feature(absolute_feature_prior),
                "absolute_prior_after": _absolute_feature(next_absolute_prior),
                "exact_selected_value": exact_selected_value,
                "exact_selected_value_source": exact_selected_value_source,
            }
            if absolute_feature_prior is not None
            else {}
        ),
        "model_before": _model(initial_model),
        "model_after": _model(model),
        "query_count": len(purchases),
        "hf_cost": spent,
        "costs": _costs([row["raw"] for row in purchases]),
        "purchases": purchases,
        "audit_trace": trace,
        "stop_reason": reason,
        "lucb_result": lucb_result,
        "confidence_certified": False,
    }


def _decision_worker(
    *,
    service: PairedObservationService,
    protocol: PopulationProtocol,
    method: str,
    incumbent: Policy,
    round_id: int,
    pool: CandidatePool,
    calibration: RoundCalibration,
    feature_prior: DeltaFeaturePrior,
    absolute_feature_prior: AbsoluteFeaturePrior | None = None,
) -> str:
    """Build one decision payload for CheckpointStore without loop-bound closures."""
    return _json(
        _round(
            service=service,
            protocol=protocol,
            method=method,
            incumbent=incumbent,
            round_id=round_id,
            pool=pool,
            calibration=calibration,
            feature_prior=feature_prior,
            absolute_feature_prior=absolute_feature_prior,
        )
    )


def _audit_worker(
    *,
    service: PairedObservationService,
    protocol: PopulationProtocol,
    method: str,
    round_id: int,
    incumbent: Policy,
    policies: Mapping[str, Policy],
) -> str:
    """Replay the independent offline phase as a durable checkpoint unit."""
    raws = [
        service.query(
            QueryKey(
                protocol.protocol_id,
                round_id,
                candidate,
                replicate,
                instance_id=protocol.instance_id,
            ),
            incumbent,
            policy,
            context=CallerContext("offline", method),
        ).to_dict()
        for candidate, policy in policies.items()
        for replicate in range(protocol.offline_replicates)
    ]
    return _json(
        {
            "round_id": round_id,
            "observations": raws,
            "costs": _costs(raws),
            "scope": "independent offline contexts; never exposed to decisions",
        }
    )


def run_population_experiment(
    *,
    output: Path,
    service: PairedObservationService,
    protocol: PopulationProtocol,
    initial_policy: Policy,
    feature_prior: DeltaFeaturePrior,
    candidate_factory: Callable[[Policy, int], CandidatePool],
    calibration_factory: Callable[[Policy, int, CandidatePool], RoundCalibration],
    frozen: FrozenInputs,
    absolute_feature_prior: AbsoluteFeaturePrior | None = None,
) -> dict[str, Any]:
    """Run all decisions first, then separate paid offline audit, with lossless replay.

    Factories are trusted frozen preparation code: they receive no observation
    service or offline data. Persist/cache expensive observer preparation outside
    this runner and include its provenance and costs in RoundCalibration.
    Cost caps are per method per round in actual environment steps; all methods
    use identical ceilings and the same pool for identical incumbent/round inputs.
    """
    frozen.verify(candidate_factory, calibration_factory, type(service.world))
    if (service.source_hash, service.dependency_hash, service.git_commit) != (
        frozen.source_hash,
        frozen.dependency_hash,
        frozen.git_commit,
    ):
        raise ValueError("observation service differs from frozen source/dependency identity")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": asdict(protocol),
        "source_hash": frozen.source_hash,
        "dependency_hash": frozen.dependency_hash,
        "git_commit": frozen.git_commit,
        "initial_policy": initial_policy.to_record(),
        "feature_prior": _feature(feature_prior),
        **(
            {"absolute_feature_prior": _absolute_feature(absolute_feature_prior)}
            if absolute_feature_prior is not None
            else {}
        ),
        "world_config": json.loads(service.world_config),
        "seed_namespace": service.seed_namespace,
        "preparation_functions": [
            f"{fn.__module__}.{fn.__qualname__}" for fn in (candidate_factory, calibration_factory)
        ],
    }
    _seal(output / "identity.json", identity)
    _seal(
        output / "source.json",
        {"files": dict(frozen.files), "dependencies": dict(frozen.dependencies)},
    )
    store = CheckpointStore(output / "checkpoint", git_commit=frozen.git_commit)
    prepared: dict[tuple[str, int], tuple[CandidatePool, RoundCalibration]] = {}
    methods: dict[str, Any] = {}
    for method in protocol.methods:
        if method == "PairedLUCB" and protocol.lucb_bounds is None:
            methods[method] = {
                "status": "not_ready",
                "reason": "true native paired-delta bounds with a source are required",
                "rounds": [],
                "offline_audit": [],
            }
            continue
        if (
            method == "GlobalPairedKG"
            and protocol.experiment == "E3C"
            and absolute_feature_prior is None
        ):
            methods[method] = {
                "status": "not_ready",
                "reason": "E3C absolute posterior transfer requires an explicit absolute_transfer preparation callback",
                "rounds": [],
                "offline_audit": [],
            }
            continue
        incumbent, current_prior = initial_policy, feature_prior
        current_absolute_prior = (
            absolute_feature_prior if method == "GlobalPairedKG" else None
        )
        seen_absolute_candidate_policies = {initial_policy.policy_id}
        seen_absolute_candidate_parameters = {_policy_parameter_hash(initial_policy)}
        records = []
        for round_id in range(protocol.rounds):
            preparation_key = (incumbent.policy_id, round_id)
            if preparation_key not in prepared:
                candidates = candidate_factory(incumbent, round_id)
                calibration = calibration_factory(incumbent, round_id, candidates)
                if not isinstance(candidates, CandidatePool) or not isinstance(
                    calibration, RoundCalibration
                ):
                    raise TypeError("factories must return typed observable/prepared inputs")
                prepared[preparation_key] = candidates, calibration
            candidates, calibration = prepared[preparation_key]
            candidate_policy_ids = [policy.policy_id for policy in candidates.policies.values()]
            candidate_parameter_hashes = [
                _policy_parameter_hash(policy) for policy in candidates.policies.values()
            ]
            if method == "GlobalPairedKG" and current_absolute_prior is not None and (
                incumbent.policy_id in candidate_policy_ids
                or len(set(candidate_policy_ids)) != len(candidate_policy_ids)
                or seen_absolute_candidate_policies.intersection(candidate_policy_ids)
                or _policy_parameter_hash(incumbent) in candidate_parameter_hashes
                or len(set(candidate_parameter_hashes)) != len(candidate_parameter_hashes)
                or seen_absolute_candidate_parameters.intersection(candidate_parameter_hashes)
            ):
                raise ValueError(
                    "absolute feature transfer requires fresh candidate policies each round"
                )
            config = {
                **identity,
                "method": method,
                "round_id": round_id,
                "incumbent": incumbent.to_record(),
                "pool": candidates.record(),
                "calibration": calibration.record(),
                "feature_prior": _feature(current_prior),
                **(
                    {
                        "absolute_feature_prior": _absolute_feature(current_absolute_prior),
                        "seen_absolute_candidate_policy_ids": sorted(
                            seen_absolute_candidate_policies
                        ),
                        "candidate_parameter_hashes": candidate_parameter_hashes,
                        "incumbent_parameter_hash": _policy_parameter_hash(incumbent),
                        "seen_absolute_candidate_parameter_hashes": sorted(
                            seen_absolute_candidate_parameters
                        ),
                    }
                    if current_absolute_prior is not None
                    else {}
                ),
            }
            worker = partial(
                _decision_worker,
                service=service,
                protocol=protocol,
                method=method,
                incumbent=incumbent,
                round_id=round_id,
                pool=candidates,
                calibration=calibration,
                feature_prior=current_prior,
                absolute_feature_prior=current_absolute_prior,
            )
            record = _unit(
                store,
                f"decision:{method}:{round_id}",
                output / method / f"round-{round_id}.json",
                worker,
                frozen,
                config,
            )
            records.append(record)
            incumbent = Policy.from_record(record["selected_policy"])
            current_prior = DeltaFeaturePrior(**record["feature_prior_after"])
            if record.get("absolute_prior_after") is not None:
                current_absolute_prior = AbsoluteFeaturePrior(**record["absolute_prior_after"])
            seen_absolute_candidate_policies.update(candidate_policy_ids)
            seen_absolute_candidate_parameters.update(candidate_parameter_hashes)
        methods[method] = {"status": "completed", "rounds": records, "offline_audit": []}
    # No offline purchase is issued until every method's decisions are durable.
    for method, data in methods.items():
        for record in data["rounds"]:
            round_id = record["round_id"]
            incumbent = Policy.from_record(record["incumbent_policy"])
            policies = {
                key: Policy.from_record(value)
                for key, value in record["candidate_pool"]["policies"].items()
            }
            audit = partial(
                _audit_worker,
                service=service,
                protocol=protocol,
                method=method,
                round_id=round_id,
                incumbent=incumbent,
                policies=policies,
            )
            audit_record = _unit(
                store,
                f"audit:{method}:{round_id}",
                output / method / f"audit-{round_id}.json",
                audit,
                frozen,
                {**identity, "decision_hash": _hash(record), "method": method, "round": round_id},
            )
            data["offline_audit"].append(audit_record)
    frozen.verify(candidate_factory, calibration_factory, type(service.world))
    result = {
        "status": "DEV",
        "calibration_passed": False,
        "identity": identity,
        "methods": methods,
        "cost_journals": {
            str(path.relative_to(service.root)): audit_evaluations(path)
            for path in sorted(service.root.rglob("journal"))
            if path.is_dir()
        },
        "limitations": [
            "Gaussian priors/likelihoods are supplied, not certified calibrated.",
            "Nominal independent context count requires adapter validation.",
            "Calibration/observer preparation costs are external inputs; missing costs remain unknown.",
        ],
    }
    _unit(store, "experiment", output / "result.json", lambda: _json(result), frozen, identity)
    return json.loads((output / "result.json").read_text())
