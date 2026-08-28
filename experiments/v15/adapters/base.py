"""Common, side-effect-free contracts for external coding-agent scaffolds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AdapterRequest:
    """A fully identified execution request with no task contents embedded."""

    task_id: str
    policy_hash: str
    seed: int
    resource_limits: dict[str, Any] = field(default_factory=dict)
    mode: str = "confirmatory"

    def __post_init__(self) -> None:
        if not self.task_id or not self.policy_hash:
            raise ValueError("task_id and policy_hash are required")
        if self.mode not in {"DEV", "confirmatory", "assessment"}:
            raise ValueError("mode must be DEV, confirmatory, or assessment")

    @property
    def request_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "policy_hash": self.policy_hash,
            "seed": self.seed,
            "resource_limits": self.resource_limits,
            "mode": self.mode,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class AdapterResult:
    """Machine-readable outcome; ``NOT_RUN`` is an explicit non-result."""

    status: str
    scaffold: str
    request_hash: str
    reason: str
    model_calls_performed: int = 0
    container_executions: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {
            "NOT_RUN",
            "COMPLETED",
            "IMPLEMENTATION_FAILURE",
            "DESIGN_INVALID",
            "UNDERPOWERED",
        }:
            raise ValueError(f"unknown adapter status: {self.status}")
        if self.model_calls_performed < 0 or self.container_executions < 0:
            raise ValueError("execution counters must be non-negative")


class ExternalAdapter(Protocol):
    """Minimal interface consumed by the control plane."""

    scaffold: str

    def status(self) -> dict[str, Any]:
        """Report availability without running the scaffold."""

    def command_preview(self, task_id: str, policy_hash: str) -> dict[str, Any]:
        """Return a redacted, dry-run command description."""

    def execute(self, request: AdapterRequest) -> AdapterResult:
        """Execute only when a concrete implementation is explicitly wired."""
