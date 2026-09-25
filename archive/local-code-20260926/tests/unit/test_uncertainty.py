from pivot.evaluation.uncertainty import bootstrap_mean_ci


def test_bootstrap_ci_is_deterministic_and_contains_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    first = bootstrap_mean_ci(values, seed=7, n_bootstrap=500)
    second = bootstrap_mean_ci(values, seed=7, n_bootstrap=500)
    assert first == second
    assert first[0] <= 2.5 <= first[1]
