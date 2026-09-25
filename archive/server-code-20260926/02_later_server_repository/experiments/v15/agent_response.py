"""Identity-blind independent-agent response interface.

The response model is an optional, separately callable diagnostic layer.  It
sees the executed patch, changed interfaces, and bounded command trace, but no
candidate identifier, proxy score, hidden task, deployment outcome, or method
label.  The module deliberately stops at structured review findings; a finding
is not silently converted into deployment utility or strategic improvement.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .external_runtime import RuntimeSettings, _model_kwargs
from .protocol import canonical_json, content_hash

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_FINDING_FIELDS = frozenset({"path", "risk", "test"})


def reviewer_request_digest(
    changed_interfaces: Sequence[str], execution_trace: Sequence[str], patch: str
) -> str:
    """Hash only the identity-blind inputs sent to the reviewer."""

    return content_hash(
        {
            "patch": patch,
            "changed_interfaces": sorted(str(value) for value in changed_interfaces),
            "execution_trace": [str(value) for value in execution_trace[-128:]],
        },
        length=64,
    )


def build_reviewer_prompt(
    *,
    patch: str,
    changed_interfaces: Sequence[str],
    execution_trace: Sequence[str],
    token_budget: int,
) -> str:
    """Build the frozen prompt for the independent review response."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    payload = {
        "patch": str(patch),
        "changed_interfaces": sorted(str(value) for value in changed_interfaces),
        "execution_trace": [str(value) for value in execution_trace[-128:]],
        "token_budget": int(token_budget),
    }
    return (
        "Review this executed software patch as an independent code reviewer. "
        "Use only the patch, changed interfaces, and bounded execution trace. "
        "Identify concrete correctness or regression risks and a focused test "
        "for each risk. Do not infer provenance, scores, or hidden evaluation "
        "context. Return exactly one JSON object with a `findings` array; each "
        "finding must contain only `path`, `risk`, and `test` string fields. "
        "An empty array is valid.\n\n"
        f"Review input:\n{canonical_json(payload)}"
    )


def parse_reviewer_response(response_text: str) -> tuple[dict[str, str], ...]:
    """Parse and validate the review model's strict JSON response."""

    clean = _CODE_FENCE_RE.sub("", response_text.strip()).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError("reviewer response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("reviewer response must be a JSON object")
    unsupported = set(payload) - {"findings"}
    if unsupported:
        raise ValueError(f"unsupported reviewer response fields: {sorted(unsupported)}")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise TypeError("reviewer response findings must be a JSON array")
    parsed: list[dict[str, str]] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise TypeError("each reviewer finding must be a JSON object")
        if set(finding) != _FINDING_FIELDS:
            raise ValueError("each reviewer finding must contain path, risk, and test fields")
        if not all(isinstance(finding[field], str) and finding[field].strip() for field in _FINDING_FIELDS):
            raise ValueError("reviewer finding fields must be non-empty strings")
        parsed.append({field: str(finding[field]).strip() for field in ("path", "risk", "test")})
    return tuple(parsed)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("reviewer response did not contain choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item) for item in content
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("reviewer response did not contain text content")
    return content.strip()


@dataclass(frozen=True)
class IndependentReview:
    """A redacted, identity-blind reviewer response."""

    status: str
    findings: tuple[dict[str, str], ...]
    request_digest: str
    response_digest: str
    model_calls: int = 1

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [dict(item) for item in self.findings],
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "model_calls": self.model_calls,
            "identity_blind": True,
        }


def review_with_model(
    settings: RuntimeSettings,
    *,
    patch: str,
    changed_interfaces: Sequence[str],
    execution_trace: Sequence[str],
) -> IndependentReview:
    """Call the independently configured reviewer and return structured output."""

    import litellm

    prompt = build_reviewer_prompt(
        patch=patch,
        changed_interfaces=changed_interfaces,
        execution_trace=execution_trace,
        token_budget=settings.model_output_tokens,
    )
    request_digest = reviewer_request_digest(changed_interfaces, execution_trace, patch)
    model_kwargs, _ = _model_kwargs(settings)
    response = litellm.completion(
        model=settings.model_key,
        messages=[
            {
                "role": "system",
                "content": "Return only the requested JSON review object.",
            },
            {"role": "user", "content": prompt},
        ],
        **model_kwargs,
    )
    content = _response_text(response)
    findings = parse_reviewer_response(content)
    return IndependentReview(
        status="COMPLETED",
        findings=findings,
        request_digest=request_digest,
        response_digest=content_hash(content, length=64),
    )


__all__ = [
    "IndependentReview",
    "build_reviewer_prompt",
    "parse_reviewer_response",
    "review_with_model",
    "reviewer_request_digest",
]
