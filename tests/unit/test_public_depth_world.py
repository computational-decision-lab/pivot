from __future__ import annotations

import pytest

from pivot.core.policy import Policy
from pivot.environments.execution_replay import ExecutionReplayWorld
from pivot.environments.finance_backtest.config import FinanceConfig
from pivot.environments.finance_backtest.public_data import (
    BookDepthRow,
    KlineBar,
    PublicBookDepthDataset,
    PublicKlineDataset,
)
from pivot.environments.finance_backtest.world import PublicDepthExecutionWorld


def _datasets() -> tuple[PublicKlineDataset, PublicBookDepthDataset, FinanceConfig]:
    bars = tuple(
        KlineBar(
            open_time_us=1_672_531_200_000_000 + index * 60_000_000,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=100.0,
            close_time_us=1_672_531_259_999_000 + index * 60_000_000,
            quote_volume=10_000.0,
            trade_count=10,
            taker_buy_volume=50.0,
            taker_buy_quote_volume=5_000.0,
        )
        for index in range(4)
    )
    rows = tuple(
        BookDepthRow(
            timestamp_us=bar.open_time_us,
            percentage=percentage,
            depth=100.0,
            notional=notional,
        )
        for bar in bars
        for percentage, notional in ((-1.0, 1_000.0), (-2.0, 2_000.0), (1.0, 1_000.0), (2.0, 2_000.0))
    )
    klines = PublicKlineDataset(
        dataset_id="depth-test",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/klines.zip",
        archive_sha256="a" * 64,
        archive_member="klines.csv",
        timestamp_unit="milliseconds",
        bars=bars,
    )
    depth = PublicBookDepthDataset(
        dataset_id="depth-test",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/depth.zip",
        archive_sha256="b" * 64,
        archive_member="depth.csv",
        rows=rows,
    )
    finance = FinanceConfig(
        prices=tuple(bar.close for bar in bars),
        spread_bps=0.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        partial_fill_ratio=1.0,
        queue_depth=1.0,
        fixture_version="depth-test",
        data_source="binance-public-data",
        data_sha256="a" * 64,
    )
    return klines, depth, finance


def test_depth_world_zero_participation_matches_replay() -> None:
    klines, depth, finance = _datasets()
    policy = Policy.from_mapping({"intensity": 0.5, "position_size": 1.0})
    replay = ExecutionReplayWorld(finance).evaluate(policy, seed=1)
    depth_world = PublicDepthExecutionWorld(klines, depth, finance, participation_rate=0.0)
    result = depth_world.evaluate(policy, seed=1)
    assert result.value == pytest.approx(replay.value)
    assert result.metadata["ground_truth_for_endogenous_response"] is False


def test_depth_world_charges_more_for_larger_participation() -> None:
    klines, depth, finance = _datasets()
    policy = Policy.from_mapping({"intensity": 0.5, "position_size": 1.0})
    low = PublicDepthExecutionWorld(klines, depth, finance, participation_rate=0.01).evaluate(
        policy, seed=1
    )
    high = PublicDepthExecutionWorld(klines, depth, finance, participation_rate=0.05).evaluate(
        policy, seed=1
    )
    assert high.value < low.value
    assert high.metadata["depth_tail_clips"] == 0
