from __future__ import annotations

from typing import Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult

from .config import FinanceConfig


class HistoricalBacktestWorld:
    """F0 fixed-price-path evaluator with no execution or endogenous response."""

    world_id = "F0-backtest"

    def __init__(self, config: FinanceConfig) -> None:
        self.config = config

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "observer",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        _ = mode
        context = _context(context_or_seed, seed)
        rng = np.random.default_rng(context.seed)
        prices = np.asarray(self.config.prices, dtype=float)
        returns = np.diff(prices) / prices[:-1]
        total = 0.0
        turnover = 0.0
        previous_position = 0.0
        for index, market_return in enumerate(returns):
            state = float(returns[index - 1]) if index else context.exogenous_value
            position = _position(policy, state)
            # Consume a deterministic random draw so F0/F1/F2 share the same
            # context protocol even though the fixture path is fixed.
            _ = rng.random()
            total += position * float(market_return)
            turnover += abs(position - previous_position)
            previous_position = position
        return RolloutResult(
            value=float(total),
            environment_steps=len(returns),
            simulator_calls=1,
            metadata={
                "world_id": self.world_id,
                "virtual_fills": True,
                "turnover": turnover,
                "participation": 0.0,
                "liquidity_consumption": 0.0,
                "strategy_frequency": self.config.strategy_frequency,
                "simulation_frequency": self.config.simulation_frequency,
                "fixture_version": self.config.fixture_version,
            },
        )

    def response_distance(self, incumbent: Policy, candidate: Policy) -> float:
        return abs(_position(candidate, 0.0) - _position(incumbent, 0.0))


class ExecutionReplayWorld(HistoricalBacktestWorld):
    """F1 fixed path plus spread, fee, slippage, queue and partial fills."""

    world_id = "F1-execution-replay"

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "observer",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        _ = mode
        context = _context(context_or_seed, seed)
        prices = np.asarray(self.config.prices, dtype=float)
        returns = np.diff(prices) / prices[:-1]
        total = 0.0
        turnover = 0.0
        fills = 0.0
        previous_position = 0.0
        cost_rate = (self.config.spread_bps / 2 + self.config.fee_bps + self.config.slippage_bps) / 10_000
        for index, market_return in enumerate(returns):
            state = float(returns[index - 1]) if index else context.exogenous_value
            target = _position(policy, state)
            requested = abs(target - previous_position)
            fill = min(requested * self.config.partial_fill_ratio, self.config.queue_depth)
            executed_position = previous_position + np.sign(target - previous_position) * fill
            total += executed_position * float(market_return) - fill * cost_rate
            turnover += fill
            fills += 1.0 if fill > 0 else 0.0
            previous_position = executed_position
        return RolloutResult(
            value=float(total),
            environment_steps=len(returns),
            simulator_calls=1,
            metadata={
                "world_id": self.world_id,
                "virtual_fills": True,
                "turnover": turnover,
                "fills": fills,
                "partial_fill_ratio": self.config.partial_fill_ratio,
                "queue_depth": self.config.queue_depth,
                "spread_crossing": turnover,
                "liquidity_consumption": turnover / self.config.queue_depth,
                "participation": 0.0,
                "strategy_frequency": self.config.strategy_frequency,
                "simulation_frequency": self.config.simulation_frequency,
                "fixture_version": self.config.fixture_version,
            },
        )


def _position(policy: Policy, state: float) -> float:
    action = policy.action(state)
    size = policy.parameters.get("position_size", 1.0)
    return float(np.clip(action * size, -1.0, 1.0))


def _context(context_or_seed: RolloutContext | int | None, seed: int | None) -> RolloutContext:
    if context_or_seed is None:
        if seed is None:
            raise TypeError("either context_or_seed or seed is required")
        context_or_seed = seed
    if isinstance(context_or_seed, RolloutContext):
        return context_or_seed
    return RolloutContext(seed=int(context_or_seed), scenario_id=f"finance-{context_or_seed}")
