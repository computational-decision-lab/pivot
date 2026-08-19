from __future__ import annotations

from typing import Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult
from pivot.environments.finance_backtest.world import ExecutionReplayWorld, _context, _position

from .config import InteractiveMarketConfig


class InteractiveMarketWorld:
    """F2 replay with explicit impact, liquidity depletion and recovery."""

    world_id = "F2-interactive-actor"

    def __init__(self, config: InteractiveMarketConfig) -> None:
        self.config = config
        self.replay = ExecutionReplayWorld(config.finance)

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "actor",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        context = _context(context_or_seed, seed)
        if mode == "observer" or self.config.participation_rate == 0.0:
            result = self.replay.evaluate(policy, context, mode="observer")
            metadata = dict(result.metadata)
            metadata.update({"world_id": self.world_id, "participation": self.config.participation_rate, "impact": 0.0, "liquidity_depletion": 0.0})
            return RolloutResult(result.value, result.environment_steps, result.simulator_calls, result.compute_cost, metadata)
        prices = np.asarray(self.config.finance.prices, dtype=float)
        returns = np.diff(prices) / prices[:-1]
        total = 0.0
        previous_position = 0.0
        impact_stock = 0.0
        impact_cost_total = 0.0
        turnover = 0.0
        for index, market_return in enumerate(returns):
            state = float(returns[index - 1]) if index else context.exogenous_value
            target = _position(policy, state)
            change = target - previous_position
            fill = min(abs(change) * self.config.finance.partial_fill_ratio, self.config.finance.queue_depth)
            executed_position = previous_position + np.sign(change) * fill
            terminal_impact = (
                self.config.participation_rate * self.config.impact_coefficient * fill
            )
            impact_cost = 0.5 * terminal_impact * fill
            cost_rate = (self.config.finance.spread_bps / 2 + self.config.finance.fee_bps + self.config.finance.slippage_bps) / 10_000
            total += executed_position * float(market_return) - fill * cost_rate - impact_cost
            turnover += fill
            impact_cost_total += impact_cost
            impact_stock = (
                (1.0 - self.config.liquidity_recovery) * impact_stock + terminal_impact
            )
            previous_position = executed_position
        return RolloutResult(
            value=float(total),
            environment_steps=len(returns),
            simulator_calls=1,
            metadata={
                "world_id": self.world_id,
                "virtual_fills": True,
                "participation": self.config.participation_rate,
                "impact": impact_stock,
                "impact_cost": impact_cost_total,
                "liquidity_depletion": min(1.0, turnover * self.config.participation_rate),
                "recovery": self.config.liquidity_recovery,
                "turnover": turnover,
                "fixture_version": self.config.finance.fixture_version,
                "data_source": self.config.finance.data_source,
                "data_sha256": self.config.finance.data_sha256,
                "source_url": self.config.finance.source_url,
                "ground_truth_for_endogenous_response": False,
            },
        )

    def response_distance(self, incumbent: Policy, candidate: Policy) -> float:
        return abs(_position(candidate, 0.0) - _position(incumbent, 0.0))
