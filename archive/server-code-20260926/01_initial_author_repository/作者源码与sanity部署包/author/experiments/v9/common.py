from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from pivot.core.policy import Policy
from pivot.v9.artifacts import current_commit, write_json
from pivot.v9.environments import environment_for
from pivot.v9.operators import (
    action_distribution_distance,
    chi_square_shift,
    generate_candidate_batch,
    policy_distance,
)
from pivot.v9.schema import make_transition_row, numeric_features, stable_id


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping in {path}")
    return payload


def load_profile(root: Path, name: str) -> dict[str, Any]:
    payload = load_yaml(root / "configs/v9/profiles.yaml")
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, Mapping) or name not in profiles:
        raise ValueError(f"unknown V9 profile: {name}")
    value = profiles[name]
    if not isinstance(value, Mapping):
        raise TypeError("profile must be a mapping")
    return {str(key): item for key, item in value.items()}


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def stable_seed(*parts: object) -> int:
    """Derive a process-independent 32-bit seed from semantic identifiers."""

    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def seed_list(config: Mapping[str, Any], profile: Mapping[str, Any]) -> list[int]:
    start = int(config.get("seed_start", 1))
    count = int(profile["seeds"])
    if count <= 0:
        raise ValueError("profile seed count must be positive")
    return list(range(start, start + count))


def setup_output(output: Path, *, resume: bool) -> None:
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite V9 run: {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)


def append_failure(output: Path, *, experiment: str, seed: int | None, error: BaseException, context: Mapping[str, Any]) -> None:
    record = {
        "experiment": experiment,
        "seed": seed,
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "failure_type": type(error).__name__,
        "exception": str(error),
        "root_cause": "unclassified",
        "fix": None,
        "rerun_status": "not_rerun",
        "result_included": False,
        "context": dict(context),
    }
    with (output / "failure_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def initial_policy() -> Policy:
    return Policy.from_mapping({"intensity": 0.18, "bias": 0.0}, metadata={"v9": "initial"})


def paired_transition_rows(
    *,
    experiment_id: str,
    environment_id: str,
    response_strength: float,
    incumbent: Policy,
    operator_family: str,
    operator_shift: float,
    candidate_count: int,
    seed: int,
    trajectory_id: str,
    round_id: int,
    root: Path,
    config_digest: str,
    method: str | None = None,
    include_strategic: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a common-random-number batch without exposing outcomes to selection."""

    world: Any = environment_for(environment_id, response_strength)
    transitions = generate_candidate_batch(
        incumbent,
        family=operator_family,
        shift_level=operator_shift,
        count=candidate_count,
        seed=seed + round_id * 100003,
        round_id=round_id,
        config_id=f"{experiment_id}:{environment_id}",
    )
    direct_incumbent = world.evaluate(incumbent, seed=seed + round_id * 991, mode="observer")
    actor_incumbent = world.evaluate(incumbent, seed=seed + round_id * 991, mode="actor")
    strategic_incumbent = (
        world.evaluate(incumbent, seed=seed + round_id * 991, mode="strategic")
        if include_strategic
        else None
    )
    output: list[dict[str, Any]] = []
    for candidate_index, transition in enumerate(transitions):
        context_seed = seed + round_id * 991
        direct_candidate = world.evaluate(transition.candidate, seed=context_seed, mode="observer")
        actor_candidate = world.evaluate(transition.candidate, seed=context_seed, mode="actor")
        strategic_candidate = (
            world.evaluate(transition.candidate, seed=context_seed, mode="strategic")
            if include_strategic
            else None
        )
        delta_proxy = float(direct_candidate.value - direct_incumbent.value)
        delta_actor = float(actor_candidate.value - actor_incumbent.value)
        delta_strategic = (
            None
            if strategic_candidate is None or strategic_incumbent is None
            else float(strategic_candidate.value - strategic_incumbent.value)
        )
        candidate_key = stable_id(experiment_id, environment_id, seed, round_id, transition.transition_id)
        row = make_transition_row(
            experiment_id=experiment_id,
            environment_id=environment_id,
            environment_family=str(getattr(world, "environment_family", environment_id)),
            trajectory_id=trajectory_id,
            round_id=round_id,
            seed=seed,
            operator_id=f"{operator_family}:{operator_shift:g}",
            operator_family=operator_family,
            candidate_id=candidate_index,
            incumbent_policy_id=incumbent.policy_id,
            candidate_policy_id=transition.candidate.policy_id,
            delta_proxy=delta_proxy,
            delta_direct=delta_proxy,
            delta_actor=delta_actor,
            delta_strategic=delta_strategic,
            delta_true=delta_actor,
            policy_distance=policy_distance(incumbent, transition.candidate),
            action_distribution_distance=action_distribution_distance(incumbent, transition.candidate),
            response_strength=response_strength,
            operator_shift=operator_shift,
            chi_square_shift=chi_square_shift(operator_shift),
            method=method,
            hf_queried=False,
            hf_cost=float(2 * actor_candidate.environment_steps),
            config_hash=config_digest,
            source_commit=current_commit(root),
            transition_id=candidate_key,
            candidate_template_id=stable_id(
                "candidate-template", environment_id, seed, round_id, operator_family, operator_shift, candidate_index
            ),
            candidate_parameters=dict(transition.candidate.parameters),
            incumbent_parameters=dict(incumbent.parameters),
            proxy_incumbent_value=float(direct_incumbent.value),
            proxy_candidate_value=float(direct_candidate.value),
            actor_incumbent_value=float(actor_incumbent.value),
            actor_candidate_value=float(actor_candidate.value),
            strategic_incumbent_value=None if strategic_incumbent is None else float(strategic_incumbent.value),
            strategic_candidate_value=None if strategic_candidate is None else float(strategic_candidate.value),
            outcome_available_for_offline_audit=True,
        )
        row["features"] = numeric_features(row)
        output.append(row)
    return output


def write_decision(output: Path, *, experiment: str, status: str, reason: str, design_valid: bool, powered: bool, allowed_claim: str, forbidden_claim: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if status not in {
        "IMPLEMENTATION_FAILURE",
        "DESIGN_INVALID",
        "UNDERPOWERED",
        "HYPOTHESIS_SUPPORTED",
        "HYPOTHESIS_NOT_SUPPORTED",
    }:
        raise ValueError(f"invalid V9 status: {status}")
    payload = {
        "experiment": experiment,
        "design_valid": bool(design_valid),
        "powered": bool(powered),
        "status": status,
        "reason": reason,
        "allowed_claim": allowed_claim,
        "forbidden_claim": forbidden_claim,
        "metrics": dict(metrics),
    }
    write_json(output / "scientific_decision.json", payload)
    return payload


def normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))
