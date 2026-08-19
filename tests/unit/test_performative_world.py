from pivot.core.policy import Policy
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld


def test_zero_response_actor_equals_observer() -> None:
    world = PerformativeWorld(PerformativeConfig(response_strength=0.0, competition_strength=0.0))
    policy = Policy.from_mapping({"intensity": 0.4})
    observer = world.evaluate(policy, seed=3, mode="observer")
    actor = world.evaluate(policy, seed=3, mode="actor")
    assert observer.value == actor.value


def test_response_changes_actor_value_for_same_update() -> None:
    low = PerformativeWorld(PerformativeConfig(response_strength=0.1, competition_strength=0.0))
    high = PerformativeWorld(PerformativeConfig(response_strength=0.9, competition_strength=0.0))
    policy = Policy.from_mapping({"intensity": 0.7})
    assert low.evaluate(policy, seed=4, mode="actor").value != high.evaluate(policy, seed=4, mode="actor").value


def test_rollout_value_respects_bound() -> None:
    world = PerformativeWorld(PerformativeConfig(response_strength=0.7, reward_bound=2.0))
    result = world.evaluate(Policy.from_mapping({"intensity": 0.9}), seed=4, mode="actor")
    assert -2.0 <= result.value <= 2.0


def test_seed_changes_stochastic_return_but_is_reproducible() -> None:
    world = PerformativeWorld(PerformativeConfig(response_strength=0.2, noise_scale=0.1))
    policy = Policy.from_mapping({"intensity": 0.4})
    assert world.evaluate(policy, seed=1, mode="actor").value == world.evaluate(
        policy, seed=1, mode="actor"
    ).value
    assert world.evaluate(policy, seed=1, mode="actor").value != world.evaluate(
        policy, seed=2, mode="actor"
    ).value


def test_controlled_world_can_expose_proxy_improvement_reversal() -> None:
    world = PerformativeWorld(PerformativeConfig(response_strength=0.7, competition_strength=0.0))
    incumbent = Policy.from_mapping({"intensity": 0.2})
    candidate = Policy.from_mapping({"intensity": 0.4})
    proxy_delta = world.evaluate(candidate, seed=5, mode="observer").value - world.evaluate(incumbent, seed=5, mode="observer").value
    actor_delta = world.evaluate(candidate, seed=5, mode="actor").value - world.evaluate(incumbent, seed=5, mode="actor").value
    assert proxy_delta > 0
    assert actor_delta < 0
