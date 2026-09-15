from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExperimentState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
    DESIGN_INVALID = "DESIGN_INVALID"
    UNDERPOWERED = "UNDERPOWERED"
    HYPOTHESIS_SUPPORTED = "HYPOTHESIS_SUPPORTED"
    HYPOTHESIS_NOT_SUPPORTED = "HYPOTHESIS_NOT_SUPPORTED"


TERMINAL_STATES = frozenset(
    {
        ExperimentState.IMPLEMENTATION_FAILURE,
        ExperimentState.DESIGN_INVALID,
        ExperimentState.UNDERPOWERED,
        ExperimentState.HYPOTHESIS_SUPPORTED,
        ExperimentState.HYPOTHESIS_NOT_SUPPORTED,
    }
)


@dataclass(frozen=True)
class ExperimentClassification:
    state: ExperimentState
    reason: str


def classify_experiment(
    *,
    implementation_failure: bool = False,
    design_invalid: bool = False,
    underpowered: bool = False,
    hypothesis_supported: bool | None = None,
    confirmatory: bool = False,
    reason: str | None = None,
) -> ExperimentClassification:
    """Choose exactly one V7 scientific outcome from explicit evidence flags."""

    choices: list[ExperimentState] = []
    if implementation_failure:
        choices.append(ExperimentState.IMPLEMENTATION_FAILURE)
    if design_invalid:
        choices.append(ExperimentState.DESIGN_INVALID)
    if underpowered:
        choices.append(ExperimentState.UNDERPOWERED)
    if hypothesis_supported is True:
        choices.append(ExperimentState.HYPOTHESIS_SUPPORTED)
    elif hypothesis_supported is False and confirmatory:
        choices.append(ExperimentState.HYPOTHESIS_NOT_SUPPORTED)
    if len(choices) != 1:
        raise ValueError("exactly one experiment outcome must be selected")
    state = choices[0]
    return ExperimentClassification(state, reason or state.value.lower().replace("_", " "))


@dataclass
class ExperimentStateMachine:
    experiment_id: str
    state: ExperimentState = ExperimentState.PENDING
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")

    def transition(self, state: ExperimentState, reason: str) -> None:
        if not reason.strip():
            raise ValueError("state transition reason must not be empty")
        if self.state in TERMINAL_STATES:
            raise ValueError("terminal experiment state cannot be rewritten")
        if self.state is ExperimentState.PENDING and state is not ExperimentState.RUNNING:
            raise ValueError("pending experiment must enter RUNNING first")
        if self.state is ExperimentState.RUNNING and state not in TERMINAL_STATES:
            raise ValueError("running experiment must end in one scientific outcome")
        self.state = state
        self.history.append({"state": state.value, "reason": reason})
