from __future__ import annotations

from bisect import bisect_right
from itertools import pairwise
from typing import Literal

import numpy as np

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext, RolloutResult

from .config import FinanceConfig
from .public_data import PublicBookDepthDataset, PublicKlineDataset


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
                "data_source": self.config.data_source,
                "data_sha256": self.config.data_sha256,
                "source_url": self.config.source_url,
                "ground_truth_for_endogenous_response": (
                    self.config.ground_truth_for_endogenous_response
                ),
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
                "data_source": self.config.data_source,
                "data_sha256": self.config.data_sha256,
                "source_url": self.config.source_url,
                "ground_truth_for_endogenous_response": (
                    self.config.ground_truth_for_endogenous_response
                ),
            },
        )


class PublicDepthExecutionWorld:
    """Depth-aware virtual execution using observed percentage-depth snapshots.

    This adapter models liquidity consumption against an observed order-book
    curve. It is an execution/impact proxy, not a counterfactual market or
    endogenous-response ground truth.
    """

    world_id = "F2-public-depth-execution-proxy"

    def __init__(
        self,
        klines: PublicKlineDataset,
        depth: PublicBookDepthDataset,
        finance: FinanceConfig,
        *,
        participation_rate: float,
        liquidity_recovery: float = 0.25,
    ) -> None:
        if klines.dataset_id != depth.dataset_id:
            raise ValueError("kline and depth datasets must identify the same session")
        if klines.symbol != depth.symbol:
            raise ValueError("kline and depth symbols must match")
        if not 0 <= participation_rate <= 1:
            raise ValueError("participation_rate must be in [0, 1]")
        if not 0 < liquidity_recovery <= 1:
            raise ValueError("liquidity_recovery must be in (0, 1]")
        if len(finance.prices) != len(klines.bars):
            raise ValueError("finance price path must align one-to-one with kline bars")
        if not depth.rows:
            raise ValueError("book-depth dataset must not be empty")
        self.klines = klines
        self.depth = depth
        self.finance = finance
        self.participation_rate = participation_rate
        self.liquidity_recovery = liquidity_recovery
        self._snapshot_times = sorted({row.timestamp_us for row in depth.rows})
        self._levels: dict[tuple[int, int], tuple[tuple[float, float], ...]] = {}
        grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for row in depth.rows:
            key = (row.timestamp_us, int(np.sign(row.percentage)))
            grouped.setdefault(key, []).append((abs(row.percentage), row.notional))
        for key, levels in grouped.items():
            ordered = tuple(sorted(levels, key=lambda item: item[0]))
            if any(
                right[0] <= left[0] or right[1] < left[1]
                for left, right in pairwise(ordered)
            ):
                raise ValueError("depth levels must have increasing distance and cumulative notional")
            self._levels[key] = ordered

    def evaluate(
        self,
        policy: Policy,
        context_or_seed: RolloutContext | int | None = None,
        mode: Literal["observer", "actor", "strategic"] = "actor",
        *,
        seed: int | None = None,
    ) -> RolloutResult:
        context = _context(context_or_seed, seed)
        if mode == "observer" or self.participation_rate == 0.0:
            replay = ExecutionReplayWorld(self.finance).evaluate(policy, context, mode="observer")
            metadata = dict(replay.metadata)
            metadata.update(
                {
                    "world_id": self.world_id,
                    "participation": self.participation_rate,
                    "depth_tail_clips": 0,
                    "missing_depth_bars": 0,
                    "impact_cost": 0.0,
                    "ground_truth_for_endogenous_response": False,
                }
            )
            return RolloutResult(
                replay.value,
                replay.environment_steps,
                replay.simulator_calls,
                replay.compute_cost,
                metadata,
            )

        returns = np.diff(np.asarray(self.finance.prices, dtype=float)) / np.asarray(
            self.finance.prices[:-1], dtype=float
        )
        total = 0.0
        previous_position = 0.0
        impact_stock = 0.0
        impact_cost_total = 0.0
        turnover = 0.0
        tail_clips = 0
        missing_depth = 0
        cost_rate = (
            self.finance.spread_bps / 2 + self.finance.fee_bps + self.finance.slippage_bps
        ) / 10_000
        for index, market_return in enumerate(returns):
            state = float(returns[index - 1]) if index else context.exogenous_value
            target = _position(policy, state)
            change = target - previous_position
            fill = min(abs(change) * self.finance.partial_fill_ratio, self.finance.queue_depth)
            executed_position = previous_position + np.sign(change) * fill
            timestamp = self._snapshot_time_for(self.klines.bars[index].open_time_us)
            levels = None if timestamp is None else self._levels.get((timestamp, int(np.sign(change))))
            order_notional = self.participation_rate * self.klines.bars[index].quote_volume * fill
            terminal_impact = 0.0
            clipped = False
            if fill > 0 and levels:
                terminal_impact, clipped = _depth_impact(levels, order_notional)
            elif fill > 0:
                missing_depth += 1
            tail_clips += int(clipped)
            impact_cost = 0.5 * terminal_impact * fill
            total += executed_position * float(market_return) - fill * cost_rate - impact_cost
            impact_cost_total += impact_cost
            turnover += fill
            impact_stock = (1.0 - self.liquidity_recovery) * impact_stock + terminal_impact
            previous_position = executed_position
        return RolloutResult(
            value=float(total),
            environment_steps=len(returns),
            simulator_calls=1,
            metadata={
                "world_id": self.world_id,
                "virtual_fills": True,
                "participation": self.participation_rate,
                "impact": impact_stock,
                "impact_cost": impact_cost_total,
                "turnover": turnover,
                "depth_tail_clips": tail_clips,
                "missing_depth_bars": missing_depth,
                "recovery": self.liquidity_recovery,
                "fixture_version": self.finance.fixture_version,
                "data_source": self.finance.data_source,
                "data_sha256": self.finance.data_sha256,
                "source_url": self.finance.source_url,
                "ground_truth_for_endogenous_response": False,
            },
        )

    def response_distance(self, incumbent: Policy, candidate: Policy) -> float:
        return abs(_position(candidate, 0.0) - _position(incumbent, 0.0))

    def _snapshot_time_for(self, open_time_us: int) -> int | None:
        position = bisect_right(self._snapshot_times, open_time_us) - 1
        return self._snapshot_times[position] if position >= 0 else None


def _depth_impact(levels: tuple[tuple[float, float], ...], order_notional: float) -> tuple[float, bool]:
    if order_notional <= 0:
        return 0.0, False
    previous_percentage = 0.0
    previous_notional = 0.0
    for percentage, notional in levels:
        if order_notional <= notional:
            if notional == previous_notional:
                return percentage / 100.0, False
            fraction = (order_notional - previous_notional) / (notional - previous_notional)
            return (previous_percentage + fraction * (percentage - previous_percentage)) / 100.0, False
        previous_percentage, previous_notional = percentage, notional
    return levels[-1][0] / 100.0, True


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
