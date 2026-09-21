"""Repeatable paid paired observations, isolated by method and acquisition phase.

This service is independent of the finite-realization DEV runner. A new replicate
is a new observation; exact solvers explicitly do not create independent samples.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult
from pivot.runner.checkpoint import CheckpointStore, ProtocolMismatchError

from .evaluation_journal import EvaluationJournal, audit_evaluations
from .seed_registry import SeedRegistry


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class QueryKey:
    protocol_id: str
    round_id: int
    candidate_id: str
    replicate_index: int
    bundle_size: int = 1
    instance_id: str = "default"

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v
            for v in (self.protocol_id, self.candidate_id, self.instance_id)
        ):
            raise ValueError("query identifiers must be nonempty strings")
        for name in ("round_id", "replicate_index", "bundle_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "bundle_size" else 0)
            ):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class CallerContext:
    phase: str
    method: str

    def __post_init__(self) -> None:
        if self.phase not in {"calibration", "online", "offline"} or not self.method:
            raise ValueError("phase must be calibration/online/offline and method nonempty")


@dataclass(frozen=True)
class Observation:
    """Immutable serialized raw result; callers receive independent dictionary copies."""

    payload: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)


def _seal(path: Path, value: dict[str, Any]) -> None:
    data = _json(value).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ProtocolMismatchError(f"observation identity conflict: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".identity-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ProtocolMismatchError(f"observation identity conflict: {path}") from None
        parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        os.unlink(temporary)


class PairedObservationService:
    """Buy observable bundles; cache ownership never crosses method or phase.

    The source/dependency hashes and explicit world configuration are caller-owned
    frozen identities. They must describe the actual adapter and installed code.
    Methods share context seeds, not observations or accounting files.
    """

    def __init__(
        self,
        root: Path,
        *,
        world: Any,
        world_config: dict[str, Any],
        source_hash: str,
        dependency_hash: str,
        seed_namespace: str = "paired-observation-v1",
        git_commit: str = "unspecified",
    ) -> None:
        for value in (source_hash, dependency_hash):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("source/dependency identities must be lowercase SHA-256")
        self.root = Path(root)
        self.world = world
        self.world_config = _json(world_config)  # Freeze caller mutations.
        self.source_hash = source_hash
        self.dependency_hash = dependency_hash
        self.seed_namespace = seed_namespace
        self.git_commit = git_commit

    def query(
        self,
        key: QueryKey,
        incumbent: Policy,
        candidate: Policy,
        *,
        context: CallerContext,
        paired: bool = True,
    ) -> Observation:
        if not isinstance(paired, bool):
            raise TypeError("paired must be boolean")
        owner = {"context": asdict(context), "key": asdict(key)}
        directory = self.root / context.phase / _hash(context.method) / _hash(asdict(key))
        directory.mkdir(parents=True, exist_ok=True)
        policy_records = {"incumbent": incumbent.to_record(), "candidate": candidate.to_record()}
        identity = {
            **owner,
            "policy_hashes": {role: _hash(record) for role, record in policy_records.items()},
            "world_class": f"{type(self.world).__module__}.{type(self.world).__qualname__}",
            "world_config": json.loads(self.world_config),
            "source_hash": self.source_hash,
            "dependency_hash": self.dependency_hash,
            "git_commit": self.git_commit,
            "seed_namespace": self.seed_namespace,
            "paired": paired,
        }
        # Sealed independently of result integrity: even a corrupt output cannot be
        # replaced with observations from a different protocol/policy/source.
        _seal(directory / "identity.json", identity)
        store = CheckpointStore(directory / "checkpoint", git_commit=self.git_commit)
        identity_args = {
            "config_hash": _hash(identity),
            "code_hash": self.source_hash,
            "dependency_hash": self.dependency_hash,
            "git_commit": self.git_commit,
        }
        output = directory / "observation.json"
        unit_key = _hash(owner)

        def worker() -> str:
            samples = []
            pairs = self._seeds(key, context, paired)
            for index, (left_seed, right_seed) in enumerate(pairs):
                pair = {}
                for role, policy, seed in (
                    ("incumbent", incumbent, left_seed),
                    ("candidate", candidate, right_seed),
                ):
                    pair[role] = self._evaluate_leg(
                        store, directory, key, context, index, role, policy, seed, identity_args
                    )
                samples.append(pair)
            return _json(self._summarize(key, context, identity, pairs, samples, output, directory))

        record = store.run_unit(unit_key, worker, output_path=output, **identity_args)
        if record["status"] != "completed":
            raise RuntimeError(
                f"observation query failed: {record.get('error_type')}: {record.get('error_message')}; {directory}"
            )
        return Observation(output.read_text())

    def _seeds(self, key: QueryKey, context: CallerContext, paired: bool) -> list[list[int]]:
        """Derive reproducible 32-bit context seeds, not an independence proof.

        The registry rejects collisions across methods, phases, candidates and
        replicates sharing this service root. Distributed workers require the
        same coordinator registry or a frozen prevalidated seed plan. Adapter
        contracts must separately verify genuine stochastic contexts.
        """
        pairs = []
        claims = []
        for offset in range(key.bundle_size):
            common = {
                "namespace": self.seed_namespace,
                "protocol": key.protocol_id,
                "instance": key.instance_id,
                "phase": context.phase,
                "round": key.round_id,
                "candidate": key.candidate_id,
                "replicate": key.replicate_index,
                "bundle_size": key.bundle_size,
                "offset": offset,
            }
            # Preserve incumbent draws between paired/unpaired ablations; only
            # candidate context independence changes. Method never enters seed ID.
            left = int(_hash({**common, "stream": "pair"})[:8], 16)
            right = (
                left if paired else int(_hash({**common, "stream": "candidate-unpaired"})[:8], 16)
            )
            if not paired and left == right:
                raise RuntimeError("seed collision in unpaired observation")
            claims.append((left, {**common, "stream": "pair"}))
            if not paired:
                claims.append((right, {**common, "stream": "candidate-unpaired"}))
            pairs.append([left, right])
        expected = key.bundle_size if paired else 2 * key.bundle_size
        if len({seed for pair in pairs for seed in pair}) != expected:
            raise RuntimeError("seed collision inside observation bundle")
        SeedRegistry(self.root / "seed_registry.sqlite").register(claims)
        return pairs

    def _evaluate_leg(
        self,
        store: CheckpointStore,
        directory: Path,
        key: QueryKey,
        context: CallerContext,
        index: int,
        role: str,
        policy: Policy,
        seed: int,
        identity_args: dict[str, str],
    ) -> dict[str, Any]:
        path = directory / "samples" / f"{index:06d}" / f"{role}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        def work() -> str:
            journal = EvaluationJournal(directory / "journal")
            started = time.perf_counter()

            def evaluate() -> RolloutResult:
                result = self.world.evaluate(policy, seed=seed, mode="actor")
                _validate_result(result)
                return result

            result = journal.evaluate(
                evaluate,
                phase=context.phase,
                method=context.method,
                round_id=key.round_id,
                policy_id=policy.policy_id,
                seed=seed,
                mode="actor",
            )
            return _json(
                {
                    "policy_id": policy.policy_id,
                    "seed": seed,
                    "value": float(result.value),
                    "environment_steps": result.environment_steps,
                    "simulator_calls": result.simulator_calls,
                    "compute_cost": result.compute_cost,
                    "wall_time": time.perf_counter() - started,
                    "metadata": dict(result.metadata),
                }
            )

        record = store.run_unit(f"leg:{index}:{role}", work, output_path=path, **identity_args)
        if record["status"] != "completed":
            raise RuntimeError(f"{role} evaluation failed: {record.get('error_message')}")
        return json.loads(path.read_text())

    @staticmethod
    def _summarize(
        key: QueryKey,
        context: CallerContext,
        identity: dict[str, Any],
        pairs: list[list[int]],
        samples: list[dict[str, Any]],
        output: Path,
        directory: Path,
    ) -> dict[str, Any]:
        values = np.asarray(
            [[sample["incumbent"]["value"], sample["candidate"]["value"]] for sample in samples]
        )
        deltas = values[:, 1] - values[:, 0]
        legs = [sample[role] for sample in samples for role in ("incumbent", "candidate")]
        flags = [_is_exact(leg["metadata"]) for leg in legs]
        if len(set(flags)) != 1:
            raise ValueError("inconsistent exact/stochastic evaluation semantics")
        exact = all(flags)
        if exact and not np.all(values == values[0]):
            raise ValueError("declared exact seed-independent evaluation changed across bundle")
        complete_compute = all(
            leg["compute_cost"] is not None and leg["metadata"].get("compute_cost_complete", True)
            for leg in legs
        )
        covariance = np.cov(values, rowvar=False, ddof=1).tolist() if len(samples) > 1 else None
        return {
            "query_key": asdict(key),
            "context": asdict(context),
            "identity": identity,
            "paired": identity["paired"],
            "seed_pairs": pairs,
            "samples": samples,
            "absolute_values": values.tolist(),
            "deltas": deltas.tolist(),
            "absolute_mean": values.mean(axis=0).tolist(),
            "mean_delta": float(deltas.mean()),
            "delta_sample_variance": float(np.var(deltas, ddof=1)) if len(samples) > 1 else None,
            "absolute_sample_covariance": covariance,
            "sample_count": len(samples),
            "is_exact": exact,
            "independent_stochastic_sample_count": 0 if exact else len(samples),
            "independence_verified": False,
            "sample_count_interpretation": "nominal_context_draws_not_verified_independent_samples",
            "replicate_adds_independent_information": not exact,
            "sample_independence_contract": "adapter_exact_seed_independent"
            if exact
            else "arm_specific_pseudorandom_contexts_adapter_validation_required",
            "costs": {
                "environment_steps": sum(leg["environment_steps"] for leg in legs),
                "simulator_calls": sum(leg["simulator_calls"] for leg in legs),
                "compute_cost": sum(leg["compute_cost"] for leg in legs)
                if all(leg["compute_cost"] is not None for leg in legs)
                else None,
                "compute_cost_complete": complete_compute,
                "wall_time": sum(leg["wall_time"] for leg in legs),
            },
            "provenance": {
                "output_path": str(output),
                "journal_path": str(directory / "journal"),
                "journal_audit": audit_evaluations(directory / "journal"),
            },
        }


def _validate_result(result: RolloutResult) -> None:
    if not math.isfinite(float(result.value)):
        raise ValueError("nonfinite observation value")
    for name in ("environment_steps", "simulator_calls"):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid measured {name}")
    if result.compute_cost is not None and (
        not math.isfinite(float(result.compute_cost)) or result.compute_cost < 0
    ):
        raise ValueError("invalid compute cost")
    _json(dict(result.metadata))


def _is_exact(metadata: dict[str, Any]) -> bool:
    exact = (
        metadata.get("is_exact") is True or metadata.get("evaluation_method") == "exact_game_tree"
    )
    return exact and metadata.get("seed_affects_value") is False
