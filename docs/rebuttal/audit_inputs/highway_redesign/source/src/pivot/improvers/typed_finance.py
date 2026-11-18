from __future__ import annotations

from dataclasses import dataclass

from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition

FINANCE_EDIT_TYPES = (
    "signal", "entry", "exit", "threshold", "position_size", "risk_size",
    "holding_horizon", "rebalance_frequency", "urgency", "participation",
)


@dataclass(frozen=True)
class TypedFinanceEdit:
    edit_type: str
    delta: float

    def __post_init__(self) -> None:
        if self.edit_type not in FINANCE_EDIT_TYPES:
            raise ValueError(f"unsupported finance edit type: {self.edit_type}")

    def propose(
        self,
        incumbent: Policy,
        *,
        round_id: int | str = 0,
        candidate_index: int = 0,
        seed: int | None = None,
        config_id: str | None = None,
    ) -> PolicyTransition:
        value = incumbent.parameters.get(self.edit_type, 0.0) + self.delta
        candidate = Policy.from_mapping(
            {**dict(incumbent.parameters), self.edit_type: value},
            metadata={"edit_type": self.edit_type, "executable": "true"},
        )
        return PolicyTransition(
            incumbent=incumbent,
            candidate=candidate,
            round_id=round_id,
            candidate_index=candidate_index,
            improvement_operator="typed-finance",
            edit_type=self.edit_type,
            seed=seed,
            config_id=config_id,
        )
