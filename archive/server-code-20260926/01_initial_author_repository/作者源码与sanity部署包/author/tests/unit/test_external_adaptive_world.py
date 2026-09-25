from __future__ import annotations

import numpy as np

from pivot.environments.external_adaptive.mpe2_world import (
    MPE2Config,
    MPE2Policy,
    MPE2World,
    generate_mpe2_candidates,
)


def test_mpe2_policy_is_deterministic_and_candidates_are_distinct() -> None:
    policy = MPE2Policy.random(seed=11, observation_dim=10, action_dim=5)
    observation = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    assert policy.action(observation) == policy.action(observation)
    candidates = generate_mpe2_candidates(policy, count=8, seed=12, scale=0.2)
    assert len(candidates) == 8
    assert len({candidate.policy_id for candidate in candidates}) == 8


def test_mpe2_world_runs_paired_actor_rollouts_without_graphics() -> None:
    config = MPE2Config(max_cycles=8, environment_version="mpe2-simple-adversary-v1")
    world = MPE2World(config)
    policy = MPE2Policy.random(seed=3, observation_dim=10, action_dim=5)
    first = world.evaluate_actor(policy, seed=44)
    second = world.evaluate_actor(policy, seed=44)

    assert first.value == second.value
    assert first.metadata["environment_source"] == "Farama-MPE2-simple_adversary_v3"
    assert first.environment_steps == 8


def test_mpe2_observer_proxy_is_distinct_from_actor_value() -> None:
    config = MPE2Config(max_cycles=8)
    world = MPE2World(config)
    policy = MPE2Policy.random(seed=3, observation_dim=10, action_dim=5)
    proxy = world.evaluate_observer(policy, seed=44)
    actor = world.evaluate_actor(policy, seed=44)

    assert proxy.metadata["mode"] == "observer"
    assert actor.metadata["mode"] == "actor"
    assert proxy.metadata["environment_source"] == actor.metadata["environment_source"]


def test_mpe2_explicit_best_response_records_an_independent_rule() -> None:
    world = MPE2World(MPE2Config(max_cycles=4))
    policy = MPE2Policy.random(seed=7, observation_dim=10, action_dim=5)
    result = world.evaluate_actor(policy, seed=9, opponent_bias=1.0, opponent_family="explicit-best-response")

    assert result.metadata["opponent_family"] == "explicit-best-response"
