"""Run lightweight structural and statistical checks over V9 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import read_jsonl_gz, write_json


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    for directory in sorted(path for path in (root / "results/v9").glob("*") if path.is_dir()):
        decision_path = directory / "scientific_decision.json"
        if not decision_path.is_file():
            continue
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        rows_path = directory / "transition_rows.jsonl.gz"
        row_kind = "transition"
        if rows_path.is_file():
            rows = read_jsonl_gz(rows_path)
        elif directory.name.startswith("e4c-") and (directory / "ood_reports.json").is_file():
            payload = json.loads((directory / "ood_reports.json").read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else []
            row_kind = "ood_report"
        elif directory.name.startswith("e7c-") and (directory / "strategic_rows.jsonl.gz").is_file():
            rows = read_jsonl_gz(directory / "strategic_rows.jsonl.gz")
            row_kind = "strategic"
        else:
            rows = []
        metric_keys = {
            "transition": ("delta_proxy", "delta_true"),
            "strategic": ("delta_actor", "delta_strategic"),
            "ood_report": ("transition_ISC", "global_ISC"),
        }[row_kind]
        finite = all(math.isfinite(float(row[key])) for row in rows for key in metric_keys if row.get(key) is not None)
        if row_kind == "ood_report":
            summary_path = directory / "ood_summary.json"
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            seed_count = int(summary_payload.get("independent_seed_count", 0)) if isinstance(summary_payload, dict) else 0
            grouped = len(rows)
        elif row_kind == "strategic":
            seed_count = len({int(row["opponent_seed"]) for row in rows}) if rows else 0
            grouped = seed_count
        else:
            seed_count = len({int(row["seed"]) for row in rows}) if rows else 0
            grouped = len({(str(row.get("trajectory_id")), int(row.get("seed", 0))) for row in rows}) if rows else 0
        if not finite:
            errors.append(f"{directory.name}: non-finite {row_kind} metric")
        summary = decision.get("metrics", {}) if isinstance(decision, dict) else {}
        if isinstance(summary, dict):
            for low, high in (("shift_effect_ci_low", "shift_effect_ci_high"), ("adaptive_effect_ci_low", "adaptive_effect_ci_high")):
                if low in summary and high in summary and float(summary[low]) > float(summary[high]):
                    errors.append(f"{directory.name}: inverted interval {low}/{high}")
        checks.append({"run": directory.name, "status": decision.get("status"), "row_kind": row_kind, "row_count": len(rows), "seed_count": seed_count, "cluster_count": grouped, "finite": finite})
    report = {"valid": not errors, "checks": checks, "errors": errors, "bootstrap_unit": "seed_or_trajectory_cluster"}
    write_json(root / "artifacts/v9/statistical_audit.json", report)
    (root / "V9_STATISTICAL_AUDIT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PIVOT V9 Statistical Audit", "", f"Status: **{'PASS' if report['valid'] else 'FAIL'}**", "", "| Run | Artifact | Rows | Seeds | Clusters | Finite |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for check in report["checks"]:
        lines.append(f"| `{check['run']}` | `{check['row_kind']}` | {check['row_count']} | {check['seed_count']} | {check['cluster_count']} | {check['finite']} |")
    lines.extend(["", "Bootstrap unit: `seed_or_trajectory_cluster`; transition rows are not treated as independent seeds."])
    if report["errors"]:
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in report["errors"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V9 statistical artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root.resolve())
    print(json.dumps({"valid": report["valid"], "runs": len(report["checks"]), "errors": len(report["errors"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
