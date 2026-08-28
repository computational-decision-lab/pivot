from __future__ import annotations

from pathlib import Path

import pytest


def test_confirmatory_overrides_are_rejected_but_dev_overrides_are_allowed() -> None:
    from experiments.v15.confirmatory_guards import reject_confirmatory_overrides

    with pytest.raises(ValueError, match="frozen protocol"):
        reject_confirmatory_overrides(True, task_limit=1)
    with pytest.raises(ValueError, match="frozen protocol"):
        reject_confirmatory_overrides(True, agent_steps=6)

    reject_confirmatory_overrides(False, task_limit=1, agent_steps=6)


def test_confirmatory_budgets_must_match_registered_values() -> None:
    from experiments.v15.confirmatory_guards import require_registered_budgets

    require_registered_budgets(True, (1, 2, 4), (1, 2, 4))
    with pytest.raises(ValueError, match="registered HF budgets"):
        require_registered_budgets(True, (1, 2), (1, 2, 4))
    require_registered_budgets(False, (1,), (1, 2, 4))


def test_confirmatory_output_is_not_overwritten(tmp_path: Path) -> None:
    from experiments.v15.confirmatory_guards import reject_existing_confirmatory_output

    reject_existing_confirmatory_output(tmp_path, False)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        reject_existing_confirmatory_output(tmp_path, True)


def test_confirmatory_empty_output_directory_is_reserved(tmp_path: Path) -> None:
    from experiments.v15.confirmatory_guards import reject_existing_confirmatory_output

    output = tmp_path / "confirmatory"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        reject_existing_confirmatory_output(output, True)


def test_registered_counts_are_derived_from_protocol() -> None:
    from experiments.v15.confirmatory_guards import registered_counts

    counts = registered_counts({
        "rounds": 30,
        "candidates_per_round": 4,
        "seed_registry": {"trajectory_count_per_operator_family": 30},
    }, operator_count=2)
    assert counts == {"trajectories": 60, "rounds": 30, "candidates": 4}
