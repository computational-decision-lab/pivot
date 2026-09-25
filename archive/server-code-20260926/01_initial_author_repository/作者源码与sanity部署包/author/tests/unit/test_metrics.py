import pytest

from pivot.metrics.improvement import compute_improvement_fidelity, compute_improvement_metrics


def test_metrics_detect_reversal_and_ties() -> None:
    rows = [
        {"delta_proxy": 1.0, "delta_true": 0.5, "round_id": 0},
        {"delta_proxy": 2.0, "delta_true": -0.2, "round_id": 0},
        {"delta_proxy": -1.0, "delta_true": -0.4, "round_id": 1},
        {"delta_proxy": 0.0, "delta_true": 0.0, "round_id": 2},
    ]
    metrics = compute_improvement_metrics(rows, tau_sign=1e-9, tau_mtr=1e-8)
    assert metrics["ide"] == 0.8250000000000001
    assert metrics["isc"] == 2 / 3
    assert metrics["irr"] == 0.5
    assert metrics["n_ties"] == 1
    assert metrics["n_reversals"] == 1


def test_strategic_reversal_rate_is_separate_from_irr() -> None:
    rows = [
        {"delta_proxy": 1.0, "delta_true": 0.8, "delta_actor": 0.1, "delta_strategic": 0.1},
        {"delta_proxy": 1.0, "delta_true": 0.4, "delta_actor": 0.2, "delta_strategic": -0.3},
    ]
    metrics = compute_improvement_metrics(rows, tau_sign=1e-9, tau_mtr=1e-8)
    assert metrics["sirr"] == 0.5


def test_improvement_fidelity_is_operator_weighted_loss() -> None:
    rows = [
        {"delta_proxy": 1.0, "delta_true": 0.5},
        {"delta_proxy": 2.0, "delta_true": -0.2},
        {"delta_proxy": -1.0, "delta_true": -0.4},
    ]
    assert compute_improvement_fidelity(rows, loss="absolute_delta_error") == pytest.approx(1.1)
    assert compute_improvement_fidelity(rows, loss="sign_error") == pytest.approx(1 / 3)


def test_improvement_fidelity_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError, match="loss"):
        compute_improvement_fidelity([], loss="unknown")
