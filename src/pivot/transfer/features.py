from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pivot.core.transition import PolicyTransition

FloatArray = NDArray[np.float64]


def _row_value(row: Mapping[str, Any] | PolicyTransition, key: str, default: float = 0.0) -> float:
    if isinstance(row, PolicyTransition):
        value = getattr(row, key, default)
    else:
        value = row.get(key, default)
    return default if value is None else float(value)


def transition_feature_vector(
    row: Mapping[str, Any] | PolicyTransition,
    *,
    include_footprint: bool = True,
) -> FloatArray:
    """Return the fixed first-version feature vector for a policy transition.

    The vector intentionally exposes only information available before an HF
    query.  It therefore cannot leak `delta_true`/`delta_actor` into the
    differential model.
    """

    footprint_components: Mapping[str, Any]
    if isinstance(row, PolicyTransition):
        footprint_components = row.footprint_components
    else:
        value = row.get("footprint_components", {})
        footprint_components = value if isinstance(value, Mapping) else {}
    footprint_values = (
        _row_value(row, "update_footprint"),
        float(footprint_components.get("mean_kl", 0.0) or 0.0),
        float(footprint_components.get("action_shift", 0.0) or 0.0),
        float(footprint_components.get("entropy_change", 0.0) or 0.0),
        float(footprint_components.get("support_expansion", 0.0) or 0.0),
    )
    if not include_footprint:
        footprint_values = (0.0, 0.0, 0.0, 0.0, 0.0)
    return np.asarray(
        [
            _row_value(row, "delta_proxy"),
            footprint_values[0],
            _row_value(row, "response_strength"),
            _row_value(row, "competition_strength"),
            _row_value(row, "candidate_index"),
            footprint_values[1],
            footprint_values[2],
            footprint_values[3],
            footprint_values[4],
        ],
        dtype=np.float64,
    )


def policy_feature_vector(policy_or_row: Any, role: str = "candidate") -> FloatArray:
    """Encode policy parameters for the global value baseline."""

    if hasattr(policy_or_row, "parameters"):
        parameters = policy_or_row.parameters
    elif isinstance(policy_or_row, Mapping):
        if role == "candidate":
            parameters = policy_or_row.get("candidate_parameters", policy_or_row)
        else:
            parameters = policy_or_row.get("incumbent_parameters", policy_or_row)
        if not isinstance(parameters, Mapping):
            parameters = {}
    else:
        parameters = {}
    keys = sorted(str(key) for key in parameters)
    # The controlled baseline uses an explicit stable feature order.  Unknown
    # parameters are retained in sorted order by the model's fit encoder.
    return np.asarray([float(parameters[key]) for key in keys], dtype=np.float64)


def policy_parameter_mapping(policy_or_row: Any, role: str = "candidate") -> dict[str, float]:
    if hasattr(policy_or_row, "parameters"):
        parameters = policy_or_row.parameters
    elif isinstance(policy_or_row, Mapping):
        parameters = policy_or_row.get(f"{role}_parameters", policy_or_row)
    else:
        parameters = {}
    if not isinstance(parameters, Mapping):
        return {}
    return {str(key): float(value) for key, value in parameters.items()}


def aligned_policy_features(items: Sequence[Any], role: str = "candidate") -> tuple[FloatArray, tuple[str, ...]]:
    mappings = [policy_parameter_mapping(item, role=role) for item in items]
    keys = tuple(sorted({key for mapping in mappings for key in mapping}))
    matrix = np.asarray(
        [[mapping.get(key, 0.0) for key in keys] for mapping in mappings],
        dtype=np.float64,
    )
    return matrix, keys
