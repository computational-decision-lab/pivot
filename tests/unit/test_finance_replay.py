from __future__ import annotations

import pytest

from pivot.core.policy import Policy
from pivot.environments.execution_replay import ExecutionReplayWorld
from pivot.environments.finance_backtest.config import FinanceConfig
from pivot.environments.finance_backtest.world import HistoricalBacktestWorld
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.improvers.typed_finance import TypedFinanceEdit


def test_f0_has_no_execution_cost_and_f1_has_virtual_fills() -> None:
    policy = Policy.from_mapping({"intensity": 0.6, "position_size": 0.5})
    config = FinanceConfig(partial_fill_ratio=0.5)
    f0 = HistoricalBacktestWorld(config).evaluate(policy, seed=1)
    f1 = ExecutionReplayWorld(config).evaluate(policy, seed=1)
    assert f0.metadata["virtual_fills"] is True
    assert f1.metadata["virtual_fills"] is True
    assert f0.value != f1.value


def test_zero_participation_f2_equals_f1() -> None:
    policy = Policy.from_mapping({"intensity": 0.6, "position_size": 0.5})
    config = FinanceConfig()
    f1 = ExecutionReplayWorld(config).evaluate(policy, seed=2)
    f2 = InteractiveMarketWorld(InteractiveMarketConfig(finance=config, participation_rate=0.0)).evaluate(policy, seed=2)
    assert f2.value == pytest.approx(f1.value)


def test_participation_changes_actor_value_and_typed_edit_is_executable() -> None:
    policy = Policy.from_mapping({"intensity": 0.2})
    low = InteractiveMarketWorld(InteractiveMarketConfig(participation_rate=0.01)).evaluate(policy, seed=3)
    high = InteractiveMarketWorld(InteractiveMarketConfig(participation_rate=0.25)).evaluate(policy, seed=3)
    assert low.value != high.value
    transition = TypedFinanceEdit("participation", 0.1).propose(policy, seed=4)
    assert transition.edit_type == "participation"
    assert transition.candidate.parameters["participation"] == pytest.approx(0.1)
