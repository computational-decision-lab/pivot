from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    NONE = "none"
    OBSERVER_FAILURE = "observer_failure"
    ENVIRONMENT_RESPONSE_FAILURE = "environment_response_failure"
    STRATEGIC_FAILURE = "strategic_failure"
    OPTIMIZATION_DRIFT = "optimization_drift"
    UNKNOWN = "unknown"


def classify_failure(
    *,
    delta_proxy: float | None,
    delta_actor: float | None,
    delta_strategic: float | None = None,
    trajectory_proxy_increasing: bool | None = None,
    trajectory_true_decreasing: bool | None = None,
    tolerance: float = 1e-9,
) -> FailureType:
    """Classify the strongest observed failure without imputing missing worlds."""

    def positive(value: float | None) -> bool:
        return value is not None and value > tolerance

    def negative(value: float | None) -> bool:
        return value is not None and value < -tolerance

    if positive(delta_actor) and negative(delta_strategic):
        return FailureType.STRATEGIC_FAILURE
    if positive(delta_proxy) and negative(delta_actor):
        return FailureType.ENVIRONMENT_RESPONSE_FAILURE
    if negative(delta_proxy) and positive(delta_actor):
        return FailureType.OBSERVER_FAILURE
    if trajectory_proxy_increasing is True and trajectory_true_decreasing is True:
        return FailureType.OPTIMIZATION_DRIFT
    if delta_proxy is None or (delta_actor is None and delta_strategic is None):
        return FailureType.UNKNOWN
    return FailureType.NONE
