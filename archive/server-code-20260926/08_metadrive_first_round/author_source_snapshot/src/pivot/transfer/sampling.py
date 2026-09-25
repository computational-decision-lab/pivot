from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def stratified_transition_sample(
    rows: Sequence[Mapping[str, Any]],
    budget: int,
    keys: Sequence[str] = ("response_strength", "optimization_strength"),
) -> list[Mapping[str, Any]]:
    """Deterministic round-robin sample over registered transition strata."""

    if budget <= 0 or budget > len(rows):
        raise ValueError("budget must be within the available rows")
    strata: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[tuple(row.get(key) for key in keys)].append(row)
    selected: list[Mapping[str, Any]] = []
    positions = {key: 0 for key in strata}
    while len(selected) < budget:
        advanced = False
        for key in sorted(strata, key=str):
            position = positions[key]
            if position < len(strata[key]) and len(selected) < budget:
                selected.append(strata[key][position])
                positions[key] += 1
                advanced = True
        if not advanced:
            break
    return selected
