from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .common import candidate_id, validate_budget


class UncertaintyModel(Protocol):
    def uncertainty(self, candidate: Any) -> float:
        ...


def select_uncertainty(candidates: Sequence[Any], model: UncertaintyModel, budget: int) -> list[str]:
    validate_budget(candidates, budget)
    ordered = sorted(
        candidates,
        key=lambda candidate: (-float(model.uncertainty(candidate)), candidate_id(candidate)),
    )
    return [candidate_id(candidate) for candidate in ordered[:budget]]
