from __future__ import annotations

from typing import Any

import numpy as np

from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition

from .schema import stable_id

OPERATOR_FAMILIES = ("local_random", "gradient_informed", "evolutionary_population")


def chi_square_shift(shift_level: float) -> float:
    if shift_level < 0 or not np.isfinite(shift_level):
        raise ValueError("shift_level must be finite and non-negative")
    return float(np.expm1(shift_level * shift_level))


def generate_candidate_batch(
    incumbent: Policy,
    *,
    family: str,
    shift_level: float,
    count: int,
    seed: int,
    round_id: int = 0,
    config_id: str = "v9",
) -> list[PolicyTransition]:
    """Generate a fixed candidate batch shared by every verifier method."""

    if family not in OPERATOR_FAMILIES:
        raise ValueError(f"unknown operator family: {family}")
    if count <= 0 or shift_level < 0 or not np.isfinite(shift_level):
        raise ValueError("count and shift_level are invalid")
    rng = np.random.default_rng(int(seed))
    current = float(incumbent.parameters.get("intensity", 0.2))
    bias = float(incumbent.parameters.get("bias", 0.0))
    radius = 0.035 + 0.045 * shift_level
    candidates: list[PolicyTransition] = []
    for index in range(count):
        if family == "local_random":
            delta = float(rng.normal(0.0, radius))
        elif family == "gradient_informed":
            direction = 1.0 if current < 0.52 else -1.0
            delta = direction * radius * (0.75 + 0.1 * rng.random()) + float(rng.normal(0.0, radius * 0.08))
        else:
            elite_direction = 1.0 if (seed + index) % 3 else -1.0
            delta = elite_direction * radius * (0.55 + 0.1 * index / max(count, 1))
            delta += float(rng.normal(0.0, radius * 0.15))
        value = float(np.clip(current + delta, -0.85, 0.85))
        candidate = Policy.from_mapping(
            {"intensity": value, "bias": bias + float(rng.normal(0.0, radius * 0.15))},
            metadata={"v9_operator_family": family, "shift_level": f"{shift_level:.8g}"},
        )
        candidates.append(
            PolicyTransition(
                incumbent=incumbent,
                candidate=candidate,
                round_id=round_id,
                candidate_index=index,
                improvement_operator=f"v9-{family}",
                edit_type="intensity_bias",
                seed=seed,
                config_id=config_id,
            )
        )
    return candidates


def policy_distance(incumbent: Policy, candidate: Policy) -> float:
    keys = sorted(set(incumbent.parameters) | set(candidate.parameters))
    return float(np.linalg.norm([candidate.parameters.get(key, 0.0) - incumbent.parameters.get(key, 0.0) for key in keys]))


def action_distribution_distance(incumbent: Policy, candidate: Policy) -> float:
    states = np.linspace(-1.0, 1.0, 21)
    left = np.asarray([incumbent.action(float(state)) for state in states])
    right = np.asarray([candidate.action(float(state)) for state in states])
    return float(np.mean(np.abs(left - right)))


def transition_metadata(transition: PolicyTransition, family: str, shift_level: float) -> dict[str, Any]:
    return {
        "operator_id": f"{family}:{shift_level:g}",
        "operator_family": family,
        "operator_shift": float(shift_level),
        "chi_square_shift": chi_square_shift(shift_level),
        "candidate_key": stable_id(transition.transition_id, family, shift_level),
        "policy_distance": policy_distance(transition.incumbent, transition.candidate),
        "action_distribution_distance": action_distribution_distance(transition.incumbent, transition.candidate),
    }
