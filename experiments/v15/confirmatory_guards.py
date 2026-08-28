"""Shared guards for immutable V15 confirmatory execution.

The DEV runners accept bounded overrides so the protocol can be smoke-tested
cheaply.  Confirmatory runners must use the values in the frozen protocol and
must never overwrite an existing result directory.  Keeping those checks in a
small dependency-free module makes it harder for one phase to accidentally
develop a more permissive execution path than the others.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def reject_confirmatory_overrides(confirmatory: bool, **overrides: Any) -> None:
    """Reject bounded DEV controls when a run is labelled confirmatory.

    ``None`` means that the caller did not request an override.  False-y values
    such as ``0`` are still rejected because they would alter the registered
    protocol just as surely as a positive value.
    """

    if not confirmatory:
        return
    changed = sorted(name for name, value in overrides.items() if value is not None)
    if changed:
        raise ValueError(
            "confirmatory execution must use the frozen protocol; overrides are not allowed: "
            + ", ".join(changed)
        )


def require_registered_budgets(
    confirmatory: bool, requested: Sequence[int], registered: Sequence[int]
) -> None:
    """Require exact registered HF budgets for confirmatory replay."""

    requested_values = tuple(sorted({int(value) for value in requested}))
    registered_values = tuple(sorted({int(value) for value in registered}))
    if any(value < 0 for value in requested_values):
        raise ValueError("HF budgets must be non-negative")
    if confirmatory and requested_values != registered_values:
        raise ValueError(
            "confirmatory execution must use the registered HF budgets: "
            f"requested={requested_values}, registered={registered_values}"
        )


def reject_existing_confirmatory_output(output: Path, confirmatory: bool) -> None:
    """Prevent a confirmatory phase from silently replacing prior results."""

    if confirmatory and Path(output).exists():
        raise FileExistsError(
            f"confirmatory output is immutable and already exists: {Path(output).resolve()}"
        )


def registered_counts(config: Mapping[str, Any], *, operator_count: int) -> dict[str, int]:
    """Return protocol counts after validating their integer shape."""

    if operator_count <= 0:
        raise ValueError("operator_count must be positive")
    try:
        rounds = int(config["rounds"])
        candidates = int(config["candidates_per_round"])
        registry = config["seed_registry"]
        trajectories_per_operator = int(registry["trajectory_count_per_operator_family"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("protocol counts are incomplete or non-integer") from exc
    if rounds <= 0 or candidates <= 0 or trajectories_per_operator <= 0:
        raise ValueError("protocol counts must be positive")
    return {
        "trajectories": trajectories_per_operator * operator_count,
        "rounds": rounds,
        "candidates": candidates,
    }


def require_registered_count(confirmatory: bool, actual: int, expected: int, label: str) -> None:
    """Check an observed count against the frozen protocol when confirming."""

    if confirmatory and int(actual) != int(expected):
        raise ValueError(
            f"confirmatory {label} count must match the frozen protocol: "
            f"actual={actual}, expected={expected}"
        )


__all__ = [
    "registered_counts",
    "reject_confirmatory_overrides",
    "reject_existing_confirmatory_output",
    "require_registered_budgets",
    "require_registered_count",
]
