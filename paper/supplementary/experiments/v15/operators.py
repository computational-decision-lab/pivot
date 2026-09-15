"""Two registered proposal mechanisms that consume proxy feedback only."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from .protocol import AgentPolicy


@dataclass(frozen=True)
class ProposalContext:
    """Information available to the self-improvement operator."""

    proxy_score: float
    proxy_feedback: dict[str, Any]
    round_index: int
    seed: int
    resource_budget: int = 8


class ProposalOperator(Protocol):
    """Protocol for an operator that knows nothing about sealed deployment outcomes."""

    @property
    def name(self) -> str:
        """Stable public operator identifier."""
        ...

    def propose(self, incumbent: AgentPolicy, context: ProposalContext, *, count: int) -> list[AgentPolicy]:
        """Produce a candidate batch from the public observer feedback."""


def _rng(context: ProposalContext, salt: str) -> random.Random:
    return random.Random(f"{salt}:{context.seed}:{context.round_index}:{context.proxy_score:.12f}")


@dataclass(frozen=True)
class HarnessSkillEvolution:
    """Instruction/skill/search workflow evolution."""

    name: str = "harness_skill_evolution"

    def propose(self, incumbent: AgentPolicy, context: ProposalContext, *, count: int) -> list[AgentPolicy]:
        if count <= 0:
            raise ValueError("count must be positive")
        generator = _rng(context, self.name)
        candidates: list[AgentPolicy] = []
        failed = int(context.proxy_feedback.get("failed_tests", 0))
        for index in range(count):
            depth = max(1, int(incumbent.search_policy.get("depth", 2)) + 1 + (index % 2))
            max_files = int(incumbent.search_policy.get("max_files", 12)) + 2 + generator.randrange(3)
            prompt = (
                f"{incumbent.system_prompt} Before editing, inspect failing tests and relevant callers "
                f"(diagnostic failures: {failed}); summarize evidence before the patch."
            )
            candidates.append(
                incumbent.with_updates(
                    system_prompt=prompt,
                    search_policy={"depth": depth, "max_files": max_files},
                    test_policy={"run_tests": True, "repair": bool(index % 2 == 0)},
                    metadata={"edit_type": "harness_skill", "operator": self.name, "candidate": str(index)},
                )
            )
        return candidates


@dataclass(frozen=True)
class MutationSelfEdit:
    """Explicit prompt/config/loop mutation driven by public diagnostics."""

    name: str = "mutation_self_edit"

    def propose(self, incumbent: AgentPolicy, context: ProposalContext, *, count: int) -> list[AgentPolicy]:
        if count <= 0:
            raise ValueError("count must be positive")
        generator = _rng(context, self.name)
        candidates: list[AgentPolicy] = []
        base_steps = int(incumbent.agent_loop_config.get("max_steps", 8))
        for index in range(count):
            steps = min(context.resource_budget + 4, base_steps + 1 + generator.randrange(3))
            context_tokens = int(incumbent.context_policy.get("max_tokens", 2048)) + 128 * (index + 1)
            candidates.append(
                incumbent.with_updates(
                    agent_loop_config={"max_steps": steps, "stop_on_failure": bool(index % 2)},
                    tool_policy={"shell": True, "read": True, "write": True, "tool_count": 3 + index},
                    context_policy={"max_tokens": context_tokens, "summarize": bool(index % 2 == 0)},
                    metadata={"edit_type": "mutation_self_edit", "operator": self.name, "candidate": str(index)},
                )
            )
        return candidates
