"""Model-driven proposal operators for the external V15 study.

The operators receive only the incumbent policy, public proxy diagnostics, and
the registered resource budget.  They return structured policy edits; hidden
task planes and deployment outcomes never enter this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .external_runtime import RuntimeSettings, _model_kwargs
from .operators import ProposalContext
from .protocol import AgentPolicy, canonical_json, content_hash

_FORBIDDEN_PROMPT_TERMS = ("hypothesis", "gate", "assessment", "strategic", "pivot")
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("proposal response did not contain choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("proposal response did not contain text content")
    return content.strip()


def _parse_updates(content: str) -> list[dict[str, Any]]:
    clean = _CODE_FENCE.sub("", content.strip()).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError("proposal response is not valid JSON") from exc
    if isinstance(payload, dict):
        payload = payload.get("candidates", payload.get("updates"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("proposal response must be a non-empty JSON array")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("each proposal must be a JSON object")
    return [dict(item) for item in payload]


def assert_public_operator_input(
    incumbent: AgentPolicy,
    context: ProposalContext,
    hidden_task_descriptors: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed if an operator payload contains a sealed task identity.

    The proposal operator is intentionally given only proxy diagnostics. This
    check runs immediately before each proposal call and compares the exact
    serializable input against gate/assessment identifiers and hashes. It does
    not inspect hidden task contents or return any hidden value to the operator.
    """

    payload = {
        "incumbent": incumbent.to_record(include_hash=False),
        "proxy_score": context.proxy_score,
        "proxy_feedback": context.proxy_feedback,
        "round": context.round_index,
        "seed": context.seed,
        "resource_budget": context.resource_budget,
    }
    serialized = canonical_json(payload)
    for descriptor in hidden_task_descriptors:
        if not isinstance(descriptor, Mapping):
            raise TypeError("hidden task descriptors must be mappings")
        for key in ("task_id", "task_hash"):
            value = str(descriptor.get(key, ""))
            if value and value in serialized:
                raise RuntimeError(f"sealed {key} leaked into operator input")


@dataclass(frozen=True)
class ExternalProposalOperator:
    """A natural-language proposal operator with a frozen edit scope."""

    name: str
    focus: str
    allowed_fields: tuple[str, ...]
    settings: RuntimeSettings

    def __post_init__(self) -> None:
        if not self.name or not self.focus:
            raise ValueError("operator name and focus are required")
        if not self.allowed_fields:
            raise ValueError("operator edit scope must not be empty")
        invalid = set(self.allowed_fields) - {
            "system_prompt",
            "agent_loop_config",
            "tool_policy",
            "search_policy",
            "test_policy",
            "context_policy",
        }
        if invalid:
            raise ValueError(f"unsupported policy edit fields: {sorted(invalid)}")
        if any(term in self.focus.casefold() for term in _FORBIDDEN_PROMPT_TERMS):
            raise ValueError("operator focus contains forbidden information")

    def _prompt(self, incumbent: AgentPolicy, context: ProposalContext, count: int) -> str:
        payload = {
            "incumbent": incumbent.to_record(include_hash=False),
            "proxy_score": context.proxy_score,
            "proxy_feedback": context.proxy_feedback,
            "round": context.round_index,
            "resource_budget": context.resource_budget,
            "requested_candidates": count,
        }
        prompt = (
            "Improve the coding agent under the provided evaluation. You are a proposal operator, "
            "not the evaluator. Use only the incumbent policy, public proxy diagnostics, and the "
            "resource budget below. Propose concrete edits that may improve task success. "
            f"Your registered focus is: {self.focus} "
            f"You may edit only these policy dimensions: {', '.join(self.allowed_fields)}. "
            "Return exactly a JSON array with one object per candidate. Each object contains only "
            "partial updates to the permitted dimensions; do not include metadata, hashes, or prose.\n\n"
            f"Public input:\n{canonical_json(payload)}"
        )
        if any(term in prompt.casefold() for term in _FORBIDDEN_PROMPT_TERMS):
            raise ValueError("operator prompt contains forbidden information")
        return prompt

    def _call(self, prompt: str) -> str:
        import litellm

        model_kwargs, _ = _model_kwargs(self.settings)
        response = litellm.completion(
            model=self.settings.model_key,
            messages=[
                {
                    "role": "system",
                    "content": "Return concise JSON policy edits. Do not include markdown or explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            **model_kwargs,
        )
        return _response_content(response)

    def propose(self, incumbent: AgentPolicy, context: ProposalContext, *, count: int) -> list[AgentPolicy]:
        if count <= 0:
            raise ValueError("count must be positive")
        prompt = self._prompt(incumbent, context, count)
        updates = _parse_updates(self._call(prompt))
        if len(updates) != count:
            raise ValueError(f"expected {count} proposals, received {len(updates)}")
        prompt_hash = content_hash(prompt)
        candidates: list[AgentPolicy] = []
        for index, update in enumerate(updates):
            unknown = set(update) - set(self.allowed_fields)
            if unknown:
                raise ValueError(f"proposal fields not allowed: {sorted(unknown)}")
            for field, value in update.items():
                if field != "system_prompt" and not isinstance(value, dict):
                    raise ValueError(f"policy update for {field} must be an object")
            merged_update = dict(update)
            for field, value in update.items():
                if field == "system_prompt":
                    continue
                # The response contract is a partial edit.  Merge nested
                # controls with the incumbent so omitted keys retain their
                # registered defaults and cannot disappear accidentally.
                merged = dict(getattr(incumbent, field))
                merged.update(value)
                merged_update[field] = merged
            candidate = incumbent.with_updates(
                **merged_update,
                metadata={
                    "operator": self.name,
                    "edit_type": self.name,
                    "candidate": str(index),
                    "proposal_prompt_hash": prompt_hash,
                },
            )
            if candidate.policy_hash == incumbent.policy_hash:
                raise ValueError("proposal did not change the incumbent policy")
            candidates.append(candidate)
        return candidates


__all__ = ["ExternalProposalOperator", "assert_public_operator_input"]
