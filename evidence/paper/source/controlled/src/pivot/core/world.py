from __future__ import annotations

from typing import Literal, Protocol

from .policy import Policy
from .result import RolloutContext, RolloutResult

WorldMode = Literal["observer", "actor", "strategic"]


class World(Protocol):
    def evaluate(
        self, policy: Policy, context: RolloutContext, mode: WorldMode = "observer"
    ) -> RolloutResult:
        ...
