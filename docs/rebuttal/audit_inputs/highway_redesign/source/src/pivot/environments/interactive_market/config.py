from __future__ import annotations

from dataclasses import dataclass, field

from pivot.environments.finance_backtest.config import FinanceConfig


@dataclass(frozen=True)
class InteractiveMarketConfig:
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    participation_rate: float = 0.0
    impact_coefficient: float = 0.12
    liquidity_recovery: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.participation_rate <= 1:
            raise ValueError("participation_rate must be in [0, 1]")
        if self.impact_coefficient < 0 or not 0 < self.liquidity_recovery <= 1:
            raise ValueError("impact and recovery parameters are invalid")
