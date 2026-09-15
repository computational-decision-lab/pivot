from __future__ import annotations

from experiments.v9.e4c_learned_ood import _build_splits


def test_e4c_splits_are_group_disjoint_and_cover_required_axes() -> None:
    rows = []
    for seed in range(6):
        for environment in ("a", "b", "c"):
            for operator in ("local_random", "gradient_informed", "evolutionary_population"):
                rows.append(
                    {
                        "transition_id": f"{seed}-{environment}-{operator}",
                        "seed": seed,
                        "environment_id": environment,
                        "operator_family": operator,
                        "response_strength": 0.3 if seed < 3 else 0.8,
                    }
                )
    splits = _build_splits(rows)
    assert {name for name, _, _ in splits} == {"trajectory", "environment", "operator", "response_regime"}
    for _, train, test in splits:
        assert {row["transition_id"] for row in train}.isdisjoint({row["transition_id"] for row in test})
