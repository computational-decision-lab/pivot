from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from .policy import Policy

TRANSITION_COLUMNS = (
    "transition_id",
    "round_id",
    "incumbent_policy_id",
    "candidate_policy_id",
    "candidate_index",
    "improvement_operator",
    "edit_type",
    "proxy_world_id",
    "high_fidelity_world_id",
    "proxy_incumbent_value",
    "proxy_candidate_value",
    "delta_proxy",
    "actor_incumbent_value",
    "actor_candidate_value",
    "delta_actor",
    "true_incumbent_value",
    "true_candidate_value",
    "delta_true",
    "strategic_incumbent_value",
    "strategic_candidate_value",
    "delta_strategic",
    "mechanical_effect",
    "competition_effect",
    "improvement_reversal",
    "strategic_improvement_reversal",
    "update_footprint",
    "footprint_components",
    "response_strength",
    "competition_strength",
    "optimization_strength",
    "opponent_context",
    "hf_queried",
    "hf_query_reason",
    "hf_query_cost",
    "seed",
    "paired_seed_ids",
    "config_id",
    "git_commit",
    "timestamp",
    "selected",
)


@dataclass(frozen=True)
class PolicyTransition:
    incumbent: Policy
    candidate: Policy
    round_id: int | str
    candidate_index: int
    improvement_operator: str
    edit_type: str | None = None
    proxy_world_id: str | None = None
    high_fidelity_world_id: str | None = None
    proxy_incumbent_value: float | None = None
    proxy_candidate_value: float | None = None
    delta_proxy: float | None = None
    actor_incumbent_value: float | None = None
    actor_candidate_value: float | None = None
    delta_actor: float | None = None
    true_incumbent_value: float | None = None
    true_candidate_value: float | None = None
    delta_true: float | None = None
    strategic_incumbent_value: float | None = None
    strategic_candidate_value: float | None = None
    delta_strategic: float | None = None
    mechanical_effect: float | None = None
    competition_effect: float | None = None
    improvement_reversal: bool | None = None
    strategic_improvement_reversal: bool | None = None
    update_footprint: float | None = None
    footprint_components: Mapping[str, float] = field(default_factory=dict)
    response_strength: float | None = None
    competition_strength: float | None = None
    optimization_strength: float | None = None
    opponent_context: Mapping[str, object] = field(default_factory=dict)
    hf_queried: bool = False
    hf_query_reason: str | None = None
    hf_query_cost: float | None = None
    seed: int | None = None
    paired_seed_ids: tuple[int, ...] = ()
    config_id: str | None = None
    git_commit: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    selected: bool = False
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.improvement_operator:
            raise ValueError("improvement_operator must not be empty")
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        object.__setattr__(self, "footprint_components", MappingProxyType(dict(self.footprint_components)))
        object.__setattr__(self, "opponent_context", MappingProxyType(dict(self.opponent_context)))
        key = {
            "round_id": self.round_id,
            "incumbent": self.incumbent.policy_id,
            "candidate": self.candidate.policy_id,
            "operator": self.improvement_operator,
            "index": self.candidate_index,
            "seed": self.seed,
            "config_id": self.config_id,
        }
        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        object.__setattr__(self, "transition_id", hashlib.sha256(canonical.encode()).hexdigest()[:20])

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "transition_id": self.transition_id,
            "round_id": self.round_id,
            "incumbent_policy_id": self.incumbent.policy_id,
            "candidate_policy_id": self.candidate.policy_id,
            "incumbent_parameters": dict(self.incumbent.parameters),
            "candidate_parameters": dict(self.candidate.parameters),
            "candidate_index": self.candidate_index,
            "improvement_operator": self.improvement_operator,
            "edit_type": self.edit_type,
            "proxy_world_id": self.proxy_world_id,
            "high_fidelity_world_id": self.high_fidelity_world_id,
            "proxy_incumbent_value": self.proxy_incumbent_value,
            "proxy_candidate_value": self.proxy_candidate_value,
            "delta_proxy": self.delta_proxy,
            "actor_incumbent_value": self.actor_incumbent_value,
            "actor_candidate_value": self.actor_candidate_value,
            "delta_actor": self.delta_actor,
            "true_incumbent_value": self.true_incumbent_value,
            "true_candidate_value": self.true_candidate_value,
            "delta_true": self.delta_true,
            "strategic_incumbent_value": self.strategic_incumbent_value,
            "strategic_candidate_value": self.strategic_candidate_value,
            "delta_strategic": self.delta_strategic,
            "mechanical_effect": self.mechanical_effect,
            "competition_effect": self.competition_effect,
            "improvement_reversal": self.improvement_reversal,
            "strategic_improvement_reversal": self.strategic_improvement_reversal,
            "update_footprint": self.update_footprint,
            "footprint_components": dict(self.footprint_components),
            "response_strength": self.response_strength,
            "competition_strength": self.competition_strength,
            "optimization_strength": self.optimization_strength,
            "opponent_context": dict(self.opponent_context),
            "hf_queried": self.hf_queried,
            "hf_query_reason": self.hf_query_reason,
            "hf_query_cost": self.hf_query_cost,
            "seed": self.seed,
            "paired_seed_ids": list(self.paired_seed_ids),
            "config_id": self.config_id,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "selected": self.selected,
        }
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PolicyTransition:
        incumbent = Policy.from_mapping(
            record.get("incumbent_parameters", {}),
        )
        candidate = Policy.from_mapping(record.get("candidate_parameters", {}))
        paired = record.get("paired_seed_ids", ())
        transition = cls(
            incumbent=incumbent,
            candidate=candidate,
            round_id=record["round_id"],
            candidate_index=int(record["candidate_index"]),
            improvement_operator=str(record["improvement_operator"]),
            edit_type=record.get("edit_type"),
            proxy_world_id=record.get("proxy_world_id"),
            high_fidelity_world_id=record.get("high_fidelity_world_id"),
            proxy_incumbent_value=record.get("proxy_incumbent_value"),
            proxy_candidate_value=record.get("proxy_candidate_value"),
            delta_proxy=record.get("delta_proxy"),
            actor_incumbent_value=record.get("actor_incumbent_value"),
            actor_candidate_value=record.get("actor_candidate_value"),
            delta_actor=record.get("delta_actor"),
            true_incumbent_value=record.get("true_incumbent_value"),
            true_candidate_value=record.get("true_candidate_value"),
            delta_true=record.get("delta_true"),
            strategic_incumbent_value=record.get("strategic_incumbent_value"),
            strategic_candidate_value=record.get("strategic_candidate_value"),
            delta_strategic=record.get("delta_strategic"),
            mechanical_effect=record.get("mechanical_effect"),
            competition_effect=record.get("competition_effect"),
            improvement_reversal=record.get("improvement_reversal"),
            strategic_improvement_reversal=record.get("strategic_improvement_reversal"),
            update_footprint=record.get("update_footprint"),
            footprint_components=record.get("footprint_components", {}),
            response_strength=record.get("response_strength"),
            competition_strength=record.get("competition_strength"),
            optimization_strength=record.get("optimization_strength"),
            opponent_context=record.get("opponent_context", {}),
            hf_queried=bool(record.get("hf_queried", False)),
            hf_query_reason=record.get("hf_query_reason"),
            hf_query_cost=record.get("hf_query_cost"),
            seed=None if record.get("seed") is None else int(record["seed"]),
            paired_seed_ids=tuple(int(value) for value in paired),
            config_id=record.get("config_id"),
            git_commit=record.get("git_commit"),
            timestamp=str(record.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            selected=bool(record.get("selected", False)),
        )
        expected_incumbent = record.get("incumbent_policy_id")
        expected_candidate = record.get("candidate_policy_id")
        if expected_incumbent is not None and str(expected_incumbent) != incumbent.policy_id:
            raise ValueError("incumbent_policy_id does not match policy content")
        if expected_candidate is not None and str(expected_candidate) != candidate.policy_id:
            raise ValueError("candidate_policy_id does not match policy content")
        expected = record.get("transition_id")
        if expected is not None and str(expected) != transition.transition_id:
            raise ValueError("transition_id does not match transition content")
        return transition
