"""Check V9 claim registry against scientific decision artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pivot.v9.artifacts import write_json


def audit(root: Path) -> dict[str, Any]:
    registry = yaml.safe_load((root / "research/claims_v9.yaml").read_text(encoding="utf-8"))
    errors: list[str] = []
    claims: list[dict[str, Any]] = []
    for claim_id, claim in registry.get("claims", {}).items():
        experiments = [str(value).lower() for value in claim.get("required_experiment", [])]
        decisions = []
        for experiment in experiments:
            path = _preferred_decision(root, experiment)
            if path is None:
                decisions.append({"experiment": experiment, "status": "MISSING"})
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                decisions.append({"experiment": experiment, "status": payload.get("status"), "powered": payload.get("powered"), "design_valid": payload.get("design_valid")})
        allowed = any(item.get("status") in claim.get("required_status", []) and item.get("powered") and item.get("design_valid") for item in decisions)
        if not allowed:
            # This is expected before confirmatory runs and is recorded, not
            # silently promoted to a claim.
            claims.append({"claim_id": claim_id, "allowed": False, "decisions": decisions, "reason": "required powered decision is absent"})
        else:
            claims.append({"claim_id": claim_id, "allowed": True, "decisions": decisions, "reason": "registered powered decision supports scoped claim"})
    report = {"valid": not errors, "claims": claims, "errors": errors, "universal_claims_forbidden": True}
    write_json(root / "artifacts/v9/claim_audit.json", report)
    (root / "V9_CLAIM_AUDIT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _preferred_decision(root: Path, experiment: str) -> Path | None:
    for path in (root / f"results/v9/{experiment}-confirmatory/scientific_decision.json", root / f"results/v9/{experiment}-development/scientific_decision.json", root / f"results/v9/{experiment}-smoke/scientific_decision.json"):
        if path.is_file():
            return path
    return None


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PIVOT V9 Claim Audit", "", f"Structural status: **{'PASS' if report['valid'] else 'FAIL'}**", "", "| Claim | Scoped decision available |", "| --- | --- |"]
    for claim in report["claims"]:
        lines.append(f"| `{claim['claim_id']}` | {claim['allowed']} |")
    lines.extend(["", "No universal PIVOT-superiority, market-ground-truth, or general-equilibrium claim is allowed."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V9 claim scope")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root.resolve())
    print(json.dumps({"valid": report["valid"], "claims": len(report["claims"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
