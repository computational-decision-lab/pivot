from __future__ import annotations

import string
from dataclasses import dataclass


@dataclass(frozen=True)
class FinanceConfig:
    """Versioned local fixture; all fills are virtual and non-authorizing."""

    prices: tuple[float, ...] = (100.0, 100.4, 100.1, 100.8, 100.6, 101.0, 100.7)
    spread_bps: float = 2.0
    fee_bps: float = 1.0
    slippage_bps: float = 3.0
    partial_fill_ratio: float = 1.0
    queue_depth: float = 1.0
    strategy_frequency: str = "bar"
    simulation_frequency: str = "event"
    fixture_version: str = "finance-fixture-v1"
    data_source: str = "synthetic_fixture"
    data_sha256: str | None = None
    source_url: str | None = None
    ground_truth_for_endogenous_response: bool = False
    virtual_fills: bool = True

    def __post_init__(self) -> None:
        if len(self.prices) < 2 or any(price <= 0 for price in self.prices):
            raise ValueError("prices must contain at least two positive values")
        if any(value < 0 for value in (self.spread_bps, self.fee_bps, self.slippage_bps)):
            raise ValueError("execution costs must be non-negative")
        if not 0 < self.partial_fill_ratio <= 1:
            raise ValueError("partial_fill_ratio must be in (0, 1]")
        if self.queue_depth <= 0 or not self.virtual_fills:
            raise ValueError("queue_depth must be positive and fills must remain virtual")
        if not self.fixture_version or not self.data_source:
            raise ValueError("fixture_version and data_source must be non-empty")
        if self.data_sha256 is not None and (
            len(self.data_sha256) != 64
            or any(character not in string.hexdigits for character in self.data_sha256)
        ):
            raise ValueError("data_sha256 must be a 64-character hexadecimal digest")
        if self.ground_truth_for_endogenous_response:
            raise ValueError("observational finance replay cannot be endogenous-response ground truth")
