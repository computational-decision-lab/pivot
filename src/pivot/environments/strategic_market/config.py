from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pivot.environments.interactive_market.config import InteractiveMarketConfig

OpponentMode = Literal["fixed", "reactive", "adaptive"]


@dataclass(frozen=True)
class StrategicMarketConfig:
    interactive: InteractiveMarketConfig = field(default_factory=InteractiveMarketConfig)
    opponent_mode: OpponentMode = "fixed"
    opponent_count: int = 1
    adaptation_steps: int = 0
    learning_rate: float = 0.1
    market_share_sensitivity: float = 0.0
    opponent_id: str = "opponent-v1"

    def __post_init__(self) -> None:
        if self.opponent_count <= 0 or self.adaptation_steps < 0:
            raise ValueError("opponent count and adaptation steps are invalid")
        if self.learning_rate < 0 or self.market_share_sensitivity < 0:
            raise ValueError("opponent response parameters must be non-negative")
        if self.opponent_mode == "adaptive" and self.adaptation_steps <= 0:
            raise ValueError("adaptive opponents require adaptation_steps > 0")
