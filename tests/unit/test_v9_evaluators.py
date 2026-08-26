from __future__ import annotations

from pivot.v9.evaluators import BootstrapEnsemble, evaluate_ood


def _rows() -> list[dict[str, object]]:
    result = []
    for index in range(12):
        incumbent = 0.15 + 0.01 * index
        candidate = incumbent + (-0.2 if index % 4 == 0 else 0.1)
        result.append(
            {
                "transition_id": f"t{index}",
                "candidate_parameters": {"intensity": candidate, "bias": 0.0},
                "incumbent_parameters": {"intensity": incumbent, "bias": 0.0},
                "response_strength": 0.5 + 0.01 * index,
                "operator_shift": 0.2 + 0.02 * index,
                "delta_proxy": candidate - incumbent + 0.03,
                "delta_true": candidate - incumbent,
                "actor_candidate_value": candidate,
                "actor_incumbent_value": incumbent,
                "policy_distance": abs(candidate - incumbent),
                "action_distribution_distance": abs(candidate - incumbent),
                "chi_square_shift": 0.1,
                "candidate_id": index,
            }
        )
    return result


def test_learned_ensemble_is_deterministic() -> None:
    rows = _rows()
    first = BootstrapEnsemble("differential", members=4, seed=8).fit(rows).predict_row(rows[0])
    second = BootstrapEnsemble("differential", members=4, seed=8).fit(rows).predict_row(rows[0])
    assert first == second


def test_ood_evaluator_reports_global_and_transition_metrics() -> None:
    rows = _rows()
    report = evaluate_ood(rows[:8], rows[8:], family="bayesian_linear")
    assert report["train_n"] == 8
    assert report["test_n"] == 4
    assert "transition_ISC" in report
    assert 0.0 <= float(report["Brier"]) <= 1.0
