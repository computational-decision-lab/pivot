"""Aggregate frozen V9 decisions into reviewer-facing Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import write_json


def analyze_root(root: Path) -> dict[str, Any]:
    result_root = root / "results/v9"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for directory in sorted(path for path in result_root.glob("*") if path.is_dir()):
        decision_path = directory / "scientific_decision.json"
        if decision_path.is_file():
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append({"run": directory.name, **payload})
        ledger = directory / "failure_ledger.jsonl"
        if ledger.is_file():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        failures.append({"run": directory.name, **value})
    report = {"runs": records, "failure_count": len(failures), "failures": failures}
    write_json(root / "artifacts/v9/analysis.json", report)
    _write_results_report(root, records)
    _write_failure_report(root, failures)
    return report


def _write_results_report(root: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# PIVOT V9 Results Report", "", "This report is generated from per-run scientific decision artifacts.", "", "| Run | Status | Powered | Design valid | Reason |", "| --- | --- | --- | --- | --- |"]
    for record in records:
        reason = str(record.get("reason", "")).replace("|", "\\|")
        lines.append(f"| `{record.get('run', '--')}` | `{record.get('status', '--')}` | {record.get('powered', False)} | {record.get('design_valid', False)} | {reason} |")
    lines.extend(["", "## Claim boundary", "", "Numbers are interpreted only within the registered environments, operator families, splits, budgets, and opponent mechanisms. Underpowered and null decisions are retained as outcomes. All-HF is an oracle reference and is not treated as a comparable acquisition method.", "", "## Reversal and efficiency reading", "", "E2C reports operator-relative improvement fidelity and reversal diagnostics. E3C reports closed-loop selection outcomes. E4C reports matched-evidence OOD/calibration diagnostics. E5C reports fixed-budget efficiency. E7C reports strategic response effects. No row supports a universal PIVOT-superiority claim."])
    (root / "V9_RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_report(root: Path, failures: list[dict[str, Any]]) -> None:
    lines = ["# PIVOT V9 Failure Ledger", "", "Generated aggregate of all per-run failure ledgers.", "", "| Run | Seed | Failure | Rerun status | Included |", "| --- | --- | --- | --- | --- |"]
    if failures:
        for failure in failures:
            lines.append(f"| `{failure.get('run', '--')}` | {failure.get('seed', '--')} | `{failure.get('failure_type', '--')}` | `{failure.get('rerun_status', '--')}` | {failure.get('result_included', False)} |")
    else:
        lines.append("| -- | -- | no recorded failures | -- | -- |")
    lines.extend(["", "A missing or failed seed is not silently imputed. The associated experiment decision remains the authority for scientific inclusion."])
    (root / "V9_FAILURE_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PIVOT V9 results")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = analyze_root(args.root.resolve())
    print(json.dumps({"runs": len(report["runs"]), "failure_count": report["failure_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
