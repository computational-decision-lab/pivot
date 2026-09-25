from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformativeConfig:
    response_strength: float = 0.0
    competition_strength: float = 0.0
    noise_scale: float = 0.05
    horizon: int = 12
    reward_bound: float = 10.0
    decay: float = 0.85
    optimization_strength: float = 1.0
    config_id: str = "controlled-v1"

    def __post_init__(self) -> None:
        if self.response_strength < 0 or self.competition_strength < 0:
            raise ValueError("response strengths must be non-negative")
        if self.noise_scale < 0 or self.horizon <= 0 or self.reward_bound <= 0:
            raise ValueError("noise, horizon, and reward bound are invalid")
        if self.decay < 0 or self.decay >= 1:
            raise ValueError("decay must be in [0, 1)")
        if self.optimization_strength < 0:
            raise ValueError("optimization_strength must be non-negative")
        if not self.config_id:
            raise ValueError("config_id must not be empty")
