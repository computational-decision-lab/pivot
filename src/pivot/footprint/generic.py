from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pivot.core.policy import Policy


@dataclass(frozen=True)
class Footprint:
    distance: float
    components: dict[str, float]


def _entropy(probability: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, probability))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def compute_update_footprint(
    pi: Policy, pi_prime: Policy, evaluation_states: Sequence[float]
) -> Footprint:
    keys = sorted(set(pi.parameters) | set(pi_prime.parameters))
    differences = [pi_prime.parameters.get(key, 0.0) - pi.parameters.get(key, 0.0) for key in keys]
    parameter_l2 = float(np.linalg.norm(np.asarray(differences, dtype=float)))
    states = list(evaluation_states)
    old_actions = np.asarray([pi.action(state) for state in states], dtype=float)
    new_actions = np.asarray([pi_prime.action(state) for state in states], dtype=float)
    shifts = np.abs(new_actions - old_actions)
    if len(states):
        action_shift = float(shifts.mean())
        max_shift = float(shifts.max())
        old_entropy = float(np.mean([_entropy((value + 1.0) / 2.0) for value in old_actions]))
        new_entropy = float(np.mean([_entropy((value + 1.0) / 2.0) for value in new_actions]))
        support_expansion = float(
            np.mean((new_actions < old_actions.min()) | (new_actions > old_actions.max()))
        )
    else:
        action_shift = max_shift = old_entropy = new_entropy = support_expansion = 0.0
    components = {
        "mean_kl": 0.5 * parameter_l2**2,
        "max_kl": 0.5 * parameter_l2**2,
        "action_shift": action_shift,
        "entropy_change": new_entropy - old_entropy,
        "occupancy_divergence": action_shift,
        "support_expansion": support_expansion,
        "trajectory_divergence": action_shift,
        "episode_length_change": 0.0,
        "max_action_shift": max_shift,
    }
    distance = parameter_l2 + action_shift
    return Footprint(distance=distance, components=components)
