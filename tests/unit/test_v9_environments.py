from __future__ import annotations

from pivot.core.policy import Policy
from pivot.v9.environments import CongestionResourceWorld, PerformativeControlWorld


def test_performative_world_is_reproducible_and_response_changes_value() -> None:
    policy = Policy.from_mapping({"intensity": 0.55, "bias": 0.02})
    world = PerformativeControlWorld()
    first = world.evaluate(policy, seed=7, mode="actor")
    second = world.evaluate(policy, seed=7, mode="actor")
    assert first.value == second.value
    assert first.value != world.evaluate(policy, seed=7, mode="observer").value


def test_congestion_world_has_endogenous_queue_signal() -> None:
    policy = Policy.from_mapping({"intensity": 0.7})
    world = CongestionResourceWorld()
    observer = world.evaluate(policy, seed=4, mode="observer")
    actor = world.evaluate(policy, seed=4, mode="actor")
    assert observer.value != actor.value
    assert actor.metadata["final_queue"] >= 0.0


def test_world_rewards_are_not_hard_ceiling_values() -> None:
    world = PerformativeControlWorld()
    values = [world.evaluate(Policy.from_mapping({"intensity": 0.2 + 0.01 * i}), seed=i, mode="actor").value for i in range(10)]
    assert len(set(values)) > 5
