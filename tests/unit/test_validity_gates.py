from __future__ import annotations

from pivot.research.validity import evaluate_e3b_gates


def test_e3b_validity_gates_accept_non_degenerate_development_fixture() -> None:
    report = evaluate_e3b_gates(
        rewards=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_possible_reward=1.0,
        response_differences=[-0.2, 0.1, 0.3, -0.1],
        candidate_true_deltas=[-0.3, -0.05, 0.1, 0.25],
        proxy_deltas=[-0.2, 0.0, 0.2, 0.3],
        paired_deltas=[0.2, 0.25, 0.15, 0.3],
    )
    assert report.valid is True
    assert all(gate.passed for gate in report.gates)


def test_e3b_validity_gates_explain_reward_ceiling_failure() -> None:
    report = evaluate_e3b_gates(
        rewards=[0.99, 1.0, 1.0, 0.98],
        max_possible_reward=1.0,
        response_differences=[0.1, 0.1],
        candidate_true_deltas=[0.1, 0.2],
        proxy_deltas=[0.1, 0.2],
        paired_deltas=[0.1, 0.2],
    )
    assert report.valid is False
    assert any(gate.name == "no_premature_ceiling" and not gate.passed for gate in report.gates)
