from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RolloutContext:
    seed: int
    scenario_id: str
    exogenous_value: float = 0.0
    initial_state: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutResult:
    value: float
    environment_steps: int = 0
    simulator_calls: int = 1
    compute_cost: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedEvaluation:
    incumbent_value: float
    candidate_value: float
    delta: float
    standard_error: float
    confidence_interval: tuple[float, float]
    num_rollouts: int
    paired_seed_ids: tuple[int, ...]
    deltas: tuple[float, ...]
    environment_steps: int
    simulator_calls: int
    compute_cost: float | None = None
