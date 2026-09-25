from __future__ import annotations

import pytest

from pivot.core.policy import Policy
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.strategic_market.config import StrategicMarketConfig
from pivot.environments.strategic_market.world import StrategicMarketWorld


def test_s0_fixed_opponent_equals_actor_world() -> None:
    policy = Policy.from_mapping({"intensity": 0.6, "position_size": 0.5})
    interactive = InteractiveMarketConfig(participation_rate=0.1)
    actor = StrategicMarketWorld(StrategicMarketConfig(interactive=interactive, opponent_mode="fixed")).evaluate(policy, seed=1)
    baseline = StrategicMarketWorld(StrategicMarketConfig(interactive=interactive, opponent_mode="fixed")).actor_world.evaluate(policy, seed=1, mode="actor")
    assert actor.value == pytest.approx(baseline.value)


def test_reactive_and_adaptive_opponents_are_deterministic_and_logged() -> None:
    policy = Policy.from_mapping({"intensity": 0.8, "position_size": 0.8})
    reactive = StrategicMarketWorld(StrategicMarketConfig(opponent_mode="reactive", market_share_sensitivity=0.05))
    adaptive = StrategicMarketWorld(StrategicMarketConfig(opponent_mode="adaptive", market_share_sensitivity=0.05, adaptation_steps=4, learning_rate=0.2))
    first = reactive.evaluate(policy, seed=2)
    second = reactive.evaluate(policy, seed=2)
    assert first.value == second.value
    assert first.metadata["opponent_response"] > 0
    assert adaptive.evaluate(policy, seed=2).metadata["opponent_skill"] > 0
