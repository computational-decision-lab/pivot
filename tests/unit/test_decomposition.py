from pivot.evaluation.decomposition import decompose_effects


def test_known_decomposition_is_exact() -> None:
    values = decompose_effects(direct=2.0, actor=1.25, strategic=0.5)
    assert values.mechanical_effect == -0.75
    assert values.competition_effect == -0.75


def test_missing_counterfactual_is_preserved() -> None:
    values = decompose_effects(direct=None, actor=1.25, strategic=None)
    assert values.mechanical_effect is None
    assert values.competition_effect is None
