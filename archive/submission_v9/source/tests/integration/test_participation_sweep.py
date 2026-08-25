from __future__ import annotations

from pivot.core.policy import Policy
from pivot.environments.finance_backtest.config import FinanceConfig
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld


def test_participation_metadata_is_monotone_for_fixed_policy() -> None:
    policy = Policy.from_mapping({"intensity": 0.8, "position_size": 0.8})
    values = []
    for participation in (0.0, 0.05, 0.1, 0.2):
        result = InteractiveMarketWorld(
            InteractiveMarketConfig(finance=FinanceConfig(), participation_rate=participation)
        ).evaluate(policy, seed=1)
        values.append(float(result.metadata["liquidity_depletion"]))
    assert values == sorted(values)
