from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pivot.core.policy import Policy

from .operator import CandidateBatch


@dataclass(frozen=True)
class _Round:
    batch: CandidateBatch
    selected_index: int
    evaluations: tuple[Mapping[str, object], ...]
    query_cost: float


@dataclass
class ImprovementTrajectory:
    """Append-only multi-round self-improvement trajectory."""

    initial_policy: Policy
    rounds: list[_Round] = field(default_factory=list)

    @property
    def current_policy(self) -> Policy:
        return self.initial_policy if not self.rounds else self.rounds[-1].batch.candidates[self.rounds[-1].selected_index].candidate

    @property
    def cumulative_true_improvement(self) -> float:
        return self._cumulative_value("delta_true", "true_delta")

    @property
    def cumulative_actor_improvement(self) -> float:
        """Cumulative selected change in the endogenous actor world."""

        return self._cumulative_value("delta_actor")

    @property
    def cumulative_strategic_improvement(self) -> float:
        """Cumulative selected change after the strategic response."""

        return self._cumulative_value("delta_strategic")

    @property
    def proxy_curve(self) -> tuple[float, ...]:
        return (0.0, *self._curve_value("delta_proxy"))

    @property
    def true_curve(self) -> tuple[float, ...]:
        return (0.0, *self._curve_value("delta_true", "true_delta"))

    @property
    def actor_curve(self) -> tuple[float, ...]:
        return (0.0, *self._curve_value("delta_actor"))

    @property
    def strategic_curve(self) -> tuple[float, ...]:
        return (0.0, *self._curve_value("delta_strategic"))

    @staticmethod
    def _evaluation_value(round_data: _Round, field: str, fallback: str | None = None) -> object:
        evaluation = round_data.evaluations[round_data.selected_index]
        value = evaluation.get(field)
        return evaluation.get(fallback) if value is None and fallback is not None else value

    def _curve_value(self, field: str, fallback: str | None = None) -> tuple[float, ...]:
        total = 0.0
        values: list[float] = []
        for round_data in self.rounds:
            value = self._evaluation_value(round_data, field, fallback)
            if isinstance(value, (int, float)):
                total += float(value)
            values.append(total)
        return tuple(values)

    def _cumulative_value(self, field: str, fallback: str | None = None) -> float:
        values = self._curve_value(field, fallback)
        return values[-1] if values else 0.0

    def append_round(
        self,
        batch: CandidateBatch,
        selected_index: int,
        evaluations: Sequence[Mapping[str, object]],
        query_cost: float = 0.0,
    ) -> Policy:
        if batch.incumbent.policy_id != self.current_policy.policy_id:
            raise ValueError("batch incumbent must equal the current trajectory policy")
        if not 0 <= selected_index < len(batch.candidates):
            raise IndexError("selected_index is outside the candidate batch")
        if len(evaluations) != len(batch.candidates):
            raise ValueError("one evaluation is required for every candidate")
        if query_cost < 0:
            raise ValueError("query_cost must be non-negative")
        normalized = tuple(dict(evaluation) for evaluation in evaluations)
        self.rounds.append(_Round(batch, selected_index, normalized, float(query_cost)))
        return self.current_policy

    def to_records(self) -> tuple[dict[str, object], ...]:
        records: list[dict[str, object]] = []
        for round_data in self.rounds:
            for index, (transition, evaluation) in enumerate(
                zip(round_data.batch.candidates, round_data.evaluations, strict=True)
            ):
                record: dict[str, object] = transition.to_record()
                record.update(evaluation)
                record["selected"] = index == round_data.selected_index
                record["query_cost"] = round_data.query_cost
                records.append(record)
        return tuple(records)
