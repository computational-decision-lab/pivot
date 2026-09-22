from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pivot.core.transition import PolicyTransition


def candidate_id(candidate: Mapping[str, Any] | PolicyTransition) -> str:
    if isinstance(candidate, PolicyTransition):
        return candidate.transition_id
    value = candidate.get("transition_id")
    if value is None:
        raise ValueError("candidate is missing transition_id")
    return str(value)


def candidate_value(candidate: Mapping[str, Any] | PolicyTransition, key: str, default: float = 0.0) -> float:
    value = getattr(candidate, key, default) if isinstance(candidate, PolicyTransition) else candidate.get(key, default)
    return default if value is None else float(value)


def validate_budget(candidates: Sequence[Any], budget: int) -> None:
    if budget < 0 or budget > len(candidates):
        raise ValueError("budget must be between zero and the number of candidates")
    ids = [candidate_id(candidate) for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate transition IDs must be unique")
