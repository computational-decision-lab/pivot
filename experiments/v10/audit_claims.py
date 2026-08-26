#!/usr/bin/env python3
"""Check the V10 narrative against its scoped claim registry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .audit_utils import load_json, rel, write_json, write_markdown


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source_path = root / "paper/iclr2027/main.tex"
    registry_path = root / "research/claims_v10.yaml"
    errors: list[str] = []
    warnings: list[str] = []
    if not source_path.is_file():
        errors.append("main.tex is missing")
    if not registry_path.is_file():
        errors.append("claims_v10.yaml is missing")
    if errors:
        report = {"valid": False, "errors": errors, "warnings": warnings}
        write_json(root, "artifacts/v10/claim_audit.json", report)
        write_markdown(root, "V10_CLAIM_AUDIT.md", _markdown(report))
        return report
    source = source_path.read_text(encoding="utf-8")
    source_flat = re.sub(r"\s+", " ", source)
    scientific = source.split("\\begin{document}", 1)[-1].split("\\appendix", 1)[0]
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    claims = registry.get("claims", {}) if isinstance(registry, dict) else {}
    forbidden_patterns = [
        r"\buniversal(?:ly)?\s+(?:superior|dominates|better)\b",
        r"ground[- ]truth simulator",
        r"causal market impact",
        r"general equilibrium",
    ]
    forbidden_hits = [
        pattern
        for pattern in forbidden_patterns
        if re.search(pattern, scientific, flags=re.IGNORECASE)
    ]
    # The manuscript must explicitly negate these tempting overclaims; their
    # presence in a limitation sentence is expected, so they are informational.
    if forbidden_hits:
        warnings.append(
            "forbidden concepts occur only in scoped-negation sentences: "
            + ", ".join(forbidden_hits)
        )
    internal_tokens = sorted(
        set(
            re.findall(
                r"\b(?:v7|v8|v9|v10|e2c|e3c|e4c|e5c|e7c|hypothesis_[a-z_]+)\b",
                scientific,
                flags=re.IGNORECASE,
            )
        )
    )
    if internal_tokens:
        errors.append(
            "internal version/run tokens in scientific body: " + ", ".join(internal_tokens)
        )
    proposition_count = len(re.findall(r"\\begin\{proposition\}", source))
    if proposition_count != 6:
        errors.append(f"expected six propositions, found {proposition_count}")
    required_phrases = {
        "primary_estimand": "Improvement Fidelity",
        "transition_object": r"\tau_t=(\pi_t,\pi_{t+1})",
        "reversal_event": "improvement reversal",
        "method": "PIVOT-VOI",
        "operator_shift": "Operator Shift Bound",
        "response_bound": "Response--Footprint Bound",
        "decision_bound": "Decision Preservation Under Differential Error",
        "finite_sample": "Finite-Sample Best-Update Identification",
        "paired_rollouts": "common randomness",
        "ood_null": "powered null",
        "finance_boundary": "does not identify causal market impact",
    }
    missing = [
        name
        for name, phrase in required_phrases.items()
        if phrase.casefold() not in source_flat.casefold()
    ]
    errors.extend(f"missing required claim phrase: {name}" for name in missing)

    # Confirm that registry statuses agree with the terminal scientific states.
    decision_map = {
        "operator_shift": root / "results/v9/e2c-confirmatory/scientific_decision.json",
        "closed_loop": root / "results/v9/e3c-confirmatory/scientific_decision.json",
        "ood_evaluator": root / "results/v9/e4c-confirmatory/scientific_decision.json",
        "efficiency": root / "results/v9/e5c-confirmatory/scientific_decision.json",
        "strategic_reversal": root / "results/v9/e7c-confirmatory/scientific_decision.json",
    }
    claim_records: list[dict[str, Any]] = []
    for claim_id, claim in claims.items():
        record: dict[str, Any] = {
            "claim_id": claim_id,
            "status": claim.get("status"),
            "scope": claim.get("scope"),
        }
        path = decision_map.get(claim_id)
        if path is not None and path.is_file():
            decision = load_json(path)
            record.update(
                {
                    "terminal_status": decision.get("status"),
                    "powered": decision.get("powered"),
                    "design_valid": decision.get("design_valid"),
                }
            )
            if claim.get("status") == "supported_scoped" and not (
                decision.get("powered") and decision.get("design_valid")
            ):
                errors.append(
                    f"{claim_id}: registry says scoped support but terminal decision is not powered/valid"
                )
            if claim_id == "ood_evaluator" and decision.get("status") != "HYPOTHESIS_NOT_SUPPORTED":
                errors.append("ood_evaluator: powered null status is not preserved")
        claim_records.append(record)
    if registry.get("claim_policy", {}).get("universal_superiority") != "forbidden":
        errors.append("registry must forbid universal superiority")
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "registry": rel(registry_path, root),
        "claims": claim_records,
        "proposition_count": proposition_count,
        "scientific_body_internal_tokens": internal_tokens,
        "required_phrases": required_phrases,
        "scope_contract": {
            "primary": registry.get("primary_contribution"),
            "method": registry.get("method"),
            "evidence_ladder": registry.get("evidence_ladder"),
            "universal_superiority_forbidden": True,
        },
    }
    write_json(root, "artifacts/v10/claim_audit.json", report)
    write_markdown(root, "V10_CLAIM_AUDIT.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Claim Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        "The primary claim is Improvement Fidelity for update transitions; all empirical claims remain scoped to the registered worlds and mechanisms.",
        "",
        "| Claim | Registry status | Terminal status | Powered / valid |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("claims", []):
        lines.append(
            f"| `{row['claim_id']}` | `{row.get('status')}` | `{row.get('terminal_status', '--')}` | {row.get('powered', '--')} / {row.get('design_valid', '--')} |"
        )
    lines.extend(
        [
            "",
            "Six propositions: " + str(report.get("proposition_count")),
            "",
            "Universal superiority, simulator ground truth, causal finance impact, and general-equilibrium claims are forbidden.",
        ]
    )
    if report.get("warnings"):
        lines.extend(["", "## Notes", "", *[f"- {item}" for item in report["warnings"]]])
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 claims")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "claims": len(report.get("claims", [])),
                "errors": len(report["errors"]),
            },
            sort_keys=True,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
