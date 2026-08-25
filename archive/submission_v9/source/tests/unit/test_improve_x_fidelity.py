from __future__ import annotations

from improve_x.failures.taxonomy import FailureType, classify_failure
from improve_x.metrics.fidelity import compute_layer_fidelity


def test_failure_taxonomy_prioritizes_strategic_reversal() -> None:
    assert (
        classify_failure(delta_proxy=1.0, delta_actor=0.5, delta_strategic=-0.2)
        == FailureType.STRATEGIC_FAILURE
    )


def test_failure_taxonomy_distinguishes_observer_and_actor_failures() -> None:
    assert (
        classify_failure(delta_proxy=1.0, delta_actor=-0.2)
        == FailureType.ENVIRONMENT_RESPONSE_FAILURE
    )
    assert (
        classify_failure(delta_proxy=-1.0, delta_actor=0.2)
        == FailureType.OBSERVER_FAILURE
    )
    assert classify_failure(delta_proxy=0.0, delta_actor=None) == FailureType.UNKNOWN


def test_layer_fidelity_reports_null_aware_sign_agreement() -> None:
    rows = [
        {"delta_proxy": 1.0, "delta_actor": 0.5, "delta_strategic": 0.2},
        {"delta_proxy": 1.0, "delta_actor": -0.5, "delta_strategic": -0.2},
        {"delta_proxy": 1.0, "delta_actor": 0.5, "delta_strategic": -0.2},
        {"delta_proxy": 1.0, "delta_actor": None, "delta_strategic": None},
    ]
    metrics = compute_layer_fidelity(rows)
    assert metrics["observer_fidelity"] == 2 / 3
    assert metrics["actor_fidelity"] == 2 / 3
    assert metrics["n_observer_actor"] == 3
    assert metrics["n_actor_strategic"] == 3
