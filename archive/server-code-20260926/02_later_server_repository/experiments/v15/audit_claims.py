from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .audit_language import _body_source_closure
from .audit_support import cli, write_audit

_REQUIRED_FIELDS = (
    "claim_id",
    "statement",
    "required_experiment",
    "required_terminal_state",
    "allowed_scope",
    "forbidden_scope",
    "paper_location",
)
_TERMINAL_STATES = {
    "IMPLEMENTATION_FAILURE",
    "DESIGN_INVALID",
    "UNDERPOWERED",
    "HYPOTHESIS_SUPPORTED",
    "HYPOTHESIS_NOT_SUPPORTED",
}


def audit_claims(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / "research/claims_v15.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    claims = payload.get("claims", []) if isinstance(payload, dict) else []
    ids = [str(item.get("claim_id", item.get("id", ""))) for item in claims if isinstance(item, dict)]
    _, body = _body_source_closure(root / "paper/iclr2027/main.tex")
    version_tokens = sorted(set(re.findall(r"\b(?:v\d+)\b", body, flags=re.IGNORECASE)))
    missing_fields = {
        str(item.get("claim_id", item.get("id", ""))): [
            field for field in _REQUIRED_FIELDS if not item.get(field)
        ]
        for item in claims
        if isinstance(item, dict) and any(not item.get(field) for field in _REQUIRED_FIELDS)
    }
    invalid_terminal_states = {
        str(item.get("claim_id", item.get("id", ""))): sorted(
            set(item.get("required_terminal_state", [])) - _TERMINAL_STATES
        )
        for item in claims
        if isinstance(item, dict)
        and set(item.get("required_terminal_state", [])) - _TERMINAL_STATES
    }
    statuses = {str(item.get("status", "")) for item in claims if isinstance(item, dict)}
    required_ids = {"C1", "C2", "C3", "C4", "C5"}
    result = {
        "registry": str(path.relative_to(root)) if path.is_file() else None,
        "claim_ids": ids,
        "required_ids_present": required_ids.issubset(set(ids)),
        "claims_have_statements": all(bool(item.get("statement")) for item in claims if isinstance(item, dict)),
        "required_fields": list(_REQUIRED_FIELDS),
        "missing_fields": missing_fields,
        "invalid_terminal_states": invalid_terminal_states,
        "unique_claim_ids": len(ids) == len(set(ids)) and all(ids),
        "statuses": sorted(statuses),
        "forbidden_inference_count": len(payload.get("forbidden_inference", [])) if isinstance(payload, dict) else 0,
        "body_version_tokens": version_tokens,
        "valid": path.is_file()
        and required_ids.issubset(set(ids))
        and all(bool(item.get("statement")) for item in claims if isinstance(item, dict))
        and not missing_fields
        and not invalid_terminal_states
        and len(ids) == len(set(ids))
        and all(ids)
        and not version_tokens,
    }
    return write_audit(
        root,
        "claim_audit",
        result,
        "Claim Audit",
        "The claim registry, permitted evidence, forbidden inferences, and paper-facing scope are checked together.",
    )


if __name__ == "__main__":
    cli(audit_claims, "Audit the registered claim boundary")
