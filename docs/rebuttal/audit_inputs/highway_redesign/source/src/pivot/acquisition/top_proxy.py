from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .common import candidate_id, candidate_value, validate_budget


def select_top_proxy(candidates: Sequence[Any], budget: int) -> list[str]:
    validate_budget(candidates, budget)
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate_value(candidate, "delta_proxy"), candidate_id(candidate)),
    )
    return [candidate_id(candidate) for candidate in ordered[:budget]]
