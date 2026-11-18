from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Policy:
    """Immutable policy parameters with a content-derived identity."""

    parameters: Mapping[str, float]
    metadata: Mapping[str, str] = field(default_factory=dict)
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        copied = {str(key): float(value) for key, value in self.parameters.items()}
        if not copied:
            raise ValueError("policy parameters must not be empty")
        if any(not math.isfinite(value) for value in copied.values()):
            raise ValueError("policy parameters must be finite")
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        object.__setattr__(self, "parameters", MappingProxyType(copied))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        payload = {"parameters": copied, "metadata": metadata}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "policy_id", hashlib.sha256(canonical.encode()).hexdigest()[:16])

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, float], metadata: Mapping[str, str] | None = None
    ) -> Policy:
        return cls(values, {} if metadata is None else metadata)

    def action(self, state: float) -> float:
        """A small generic policy used by the controlled environment."""
        intensity = self.parameters.get("intensity", 0.0)
        bias = self.parameters.get("bias", 0.0)
        value = intensity + bias * state
        return max(-1.0, min(1.0, value))

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Policy:
        raw_parameters = record.get("parameters", {})
        raw_metadata = record.get("metadata", {})
        parameters = (
            {str(key): float(value) for key, value in raw_parameters.items()}
            if isinstance(raw_parameters, Mapping)
            else {}
        )
        metadata = (
            {str(key): str(value) for key, value in raw_metadata.items()}
            if isinstance(raw_metadata, Mapping)
            else {}
        )
        policy = cls.from_mapping(
            parameters,
            metadata,
        )
        expected = record.get("policy_id")
        if expected is not None and str(expected) != policy.policy_id:
            raise ValueError("policy_id does not match policy content")
        return policy
