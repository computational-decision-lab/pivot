from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _sign(value: object, tolerance: float) -> int:
    if value is None:
        return 0
    if not isinstance(value, (int, float)):
        raise TypeError("fidelity values must be numeric or None")
    numeric = float(value)
    if abs(numeric) <= tolerance:
        return 0
    return 1 if numeric > 0 else -1


def _agreement(rows: Sequence[Mapping[str, Any]], left: str, right: str, tolerance: float) -> tuple[float | None, int]:
    comparable = [row for row in rows if row.get(left) is not None and row.get(right) is not None]
    comparable = [row for row in comparable if _sign(row[left], tolerance) and _sign(row[right], tolerance)]
    if not comparable:
        return None, 0
    agreement = sum(_sign(row[left], tolerance) == _sign(row[right], tolerance) for row in comparable)
    return agreement / len(comparable), len(comparable)


def compute_layer_fidelity(
    rows: Sequence[Mapping[str, Any]], tolerance: float = 1e-9
) -> dict[str, float | int | None]:
    """Compare adjacent world deltas while preserving unavailable layers."""

    observer, n_observer = _agreement(rows, "delta_proxy", "delta_actor", tolerance)
    actor, n_actor = _agreement(rows, "delta_actor", "delta_strategic", tolerance)
    strategic, n_strategic = _agreement(rows, "delta_proxy", "delta_strategic", tolerance)
    return {
        "observer_fidelity": observer,
        "actor_fidelity": actor,
        "strategic_fidelity": strategic,
        "n_observer_actor": n_observer,
        "n_actor_strategic": n_actor,
        "n_observer_strategic": n_strategic,
    }
