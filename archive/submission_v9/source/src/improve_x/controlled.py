from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.performative.world import PerformativeWorld
from pivot.evaluation.decomposition import decompose_effects
from pivot.evaluation.paired import PairedEvaluator
from pivot.footprint.generic import compute_update_footprint


def evaluate_transition(
    world: PerformativeWorld,
    transition: PolicyTransition,
    contexts: Sequence[RolloutContext],
) -> dict[str, object]:
    """Evaluate one transition in observer, actor, and strategic worlds."""

    observer = PairedEvaluator(world, mode="observer").evaluate(transition, contexts)
    actor = PairedEvaluator(world, mode="actor").evaluate(transition, contexts)
    strategic = PairedEvaluator(world, mode="strategic").evaluate(transition, contexts)
    footprint = compute_update_footprint(transition.incumbent, transition.candidate, (-1.0, 0.0, 1.0))
    effects = decompose_effects(observer.delta, actor.delta, strategic.delta)
    return {
        "proxy_world_id": str(observer.paired_seed_ids),
        "high_fidelity_world_id": world.config.config_id,
        "proxy_incumbent_value": observer.incumbent_value,
        "proxy_candidate_value": observer.candidate_value,
        "delta_proxy": observer.delta,
        "actor_incumbent_value": actor.incumbent_value,
        "actor_candidate_value": actor.candidate_value,
        "delta_actor": actor.delta,
        "true_incumbent_value": actor.incumbent_value,
        "true_candidate_value": actor.candidate_value,
        "delta_true": actor.delta,
        "strategic_incumbent_value": strategic.incumbent_value,
        "strategic_candidate_value": strategic.candidate_value,
        "delta_strategic": strategic.delta,
        "mechanical_effect": effects.mechanical_effect,
        "competition_effect": effects.competition_effect,
        "improvement_reversal": observer.delta > 0 and actor.delta < 0,
        "strategic_improvement_reversal": actor.delta > 0 and strategic.delta < 0,
        "update_footprint": footprint.distance,
        "footprint_components": dict(footprint.components),
        "response_strength": world.config.response_strength,
        "competition_strength": world.config.competition_strength,
        "paired_seed_ids": list(observer.paired_seed_ids),
        "hf_queried": True,
        "hf_query_reason": "controlled_three_world_evaluation",
        "hf_query_cost": float(actor.environment_steps + strategic.environment_steps),
    }


def make_contexts(seed: int, count: int, scenario_prefix: str) -> tuple[RolloutContext, ...]:
    if count <= 0:
        raise ValueError("contexts_per_transition must be positive")
    return tuple(
        RolloutContext(seed=seed + offset, scenario_id=f"{scenario_prefix}-{seed + offset}")
        for offset in range(count)
    )


def policy_from_config(payload: Mapping[str, Any]) -> Policy:
    values = payload.get("initial_policy", {"intensity": 0.2})
    if not isinstance(values, Mapping):
        raise TypeError("initial_policy must be a mapping")
    return Policy.from_mapping({str(key): float(value) for key, value in values.items()})
