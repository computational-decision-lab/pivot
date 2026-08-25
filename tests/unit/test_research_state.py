from __future__ import annotations

import pytest

from pivot.research.state import ExperimentState, ExperimentStateMachine, classify_experiment


def test_classification_has_one_explicit_terminal_state() -> None:
    assert classify_experiment(design_invalid=True).state is ExperimentState.DESIGN_INVALID
    assert classify_experiment(underpowered=True).state is ExperimentState.UNDERPOWERED
    assert classify_experiment(hypothesis_supported=True).state is ExperimentState.HYPOTHESIS_SUPPORTED
    assert classify_experiment(hypothesis_supported=False, confirmatory=True).state is ExperimentState.HYPOTHESIS_NOT_SUPPORTED


def test_state_machine_rejects_terminal_state_rewrites() -> None:
    machine = ExperimentStateMachine("e3b_closed_loop")
    machine.transition(ExperimentState.RUNNING, "development started")
    machine.transition(ExperimentState.DESIGN_INVALID, "ceiling gate failed")
    with pytest.raises(ValueError, match="terminal"):
        machine.transition(ExperimentState.RUNNING, "cannot reopen confirmatory result")


def test_classification_rejects_conflicting_flags() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        classify_experiment(design_invalid=True, underpowered=True)
