from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


class ImprovementOperator(Protocol):
    """Operator that proposes candidate policy transitions."""

    @property
    def operator_name(self) -> str:
        ...

    def propose(
        self,
        incumbent: Policy,
        round_id: int | str,
        seed: int,
        **kwargs: Any,
    ) -> CandidateBatch:
        """
        Generate a batch of candidates from one incumbent.

        Concrete legacy PIVOT operators can be adapted by
        :func:`adapt_legacy_operator` without changing their implementation.
        """


@dataclass(frozen=True)
class CandidateBatch:
    """Immutable candidate set for one self-improvement round."""

    incumbent: Policy
    candidates: tuple[PolicyTransition, ...]
    operator: str
    round_id: int | str
    seed: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operator:
            raise ValueError("operator must not be empty")
        if not self.candidates:
            raise ValueError("candidate batch must contain at least one candidate")
        if any(transition.incumbent.policy_id != self.incumbent.policy_id for transition in self.candidates):
            raise ValueError("all candidates must share the batch incumbent")
        if any(transition.round_id != self.round_id for transition in self.candidates):
            raise ValueError("all candidates must share the batch round_id")
        if any(transition.improvement_operator != self.operator for transition in self.candidates):
            raise ValueError("all candidates must share the batch operator")
        if len({transition.candidate.policy_id for transition in self.candidates}) != len(self.candidates):
            raise ValueError("candidate policy IDs must be unique within a batch")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        metadata = dict(self.metadata) if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(transition.candidate.policy_id for transition in self.candidates)

    def to_records(self) -> tuple[dict[str, object], ...]:
        return tuple(transition.to_record() for transition in self.candidates)


def adapt_legacy_operator(operator: Any) -> ImprovementOperator:
    """Return an adapter for a legacy PIVOT operator.

    The adapter intentionally keeps keyword arguments explicit at the call
    site; it only standardizes the result shape and does not alter candidate
    generation semantics.
    """

    return _LegacyOperatorAdapter(operator)


@dataclass(frozen=True)
class _LegacyOperatorAdapter:
    legacy: Any

    @property
    def operator_name(self) -> str:
        return str(getattr(self.legacy, "operator_name", self.legacy.__class__.__name__))

    def propose(
        self,
        incumbent: Policy,
        round_id: int | str,
        seed: int,
        **kwargs: Any,
    ) -> CandidateBatch:
        transitions = self.legacy.propose(
            incumbent,
            seed=seed,
            round_id=round_id,
            **kwargs,
        )
        return CandidateBatch(incumbent, tuple(transitions), self.operator_name, round_id, seed)
