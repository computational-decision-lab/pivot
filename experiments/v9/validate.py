"""Validate V9 run contracts and manifests without changing result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import read_jsonl_gz, validate_manifest, write_json
from pivot.v9.schema import V9_TERMINAL_STATES, V9_TRANSITION_COLUMNS


def validate_root(root: Path, *, strict: bool = False) -> dict[str, Any]:
    result_root = root / "results/v9"
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    if not result_root.is_dir():
        errors.append("results/v9 is missing")
    else:
        for directory in sorted(path for path in result_root.iterdir() if path.is_dir()):
            manifest = validate_manifest(directory)
            decision_path = directory / "scientific_decision.json"
            decision: dict[str, Any] = {}
            if decision_path.is_file():
                payload = json.loads(decision_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    decision = payload
                else:
                    errors.append(f"{directory.name}: decision is not an object")
            else:
                errors.append(f"{directory.name}: scientific_decision.json missing")
            status = str(decision.get("status", ""))
            if status and status not in V9_TERMINAL_STATES:
                errors.append(f"{directory.name}: invalid terminal status {status}")
            row_path = directory / "transition_rows.jsonl.gz"
            if row_path.is_file():
                rows = read_jsonl_gz(row_path)
                missing = sorted(set(V9_TRANSITION_COLUMNS) - set(rows[0])) if rows else list(V9_TRANSITION_COLUMNS)
                if missing:
                    errors.append(f"{directory.name}: transition schema missing {','.join(missing)}")
                runs.append({"run": directory.name, "rows": len(rows), "row_kind": "transition", "status": status, "manifest": manifest})
            elif directory.name.startswith("e4c-") and (directory / "ood_reports.json").is_file():
                reports = json.loads((directory / "ood_reports.json").read_text(encoding="utf-8"))
                if not isinstance(reports, list) or not reports:
                    errors.append(f"{directory.name}: OOD report artifact is empty")
                    report_count = 0
                else:
                    report_count = len(reports)
                summary = _read_object(directory / "ood_summary.json")
                runs.append({"run": directory.name, "rows": report_count, "row_kind": "ood_report", "seed_count": summary.get("independent_seed_count"), "status": status, "manifest": manifest})
            elif directory.name.startswith("e7c-") and (directory / "strategic_rows.jsonl.gz").is_file():
                rows = read_jsonl_gz(directory / "strategic_rows.jsonl.gz")
                if not rows:
                    errors.append(f"{directory.name}: strategic row artifact is empty")
                runs.append({"run": directory.name, "rows": len(rows), "row_kind": "strategic", "seed_count": len({int(row["opponent_seed"]) for row in rows}) if rows else 0, "status": status, "manifest": manifest})
            else:
                runs.append({"run": directory.name, "rows": None, "row_kind": "none", "status": status, "manifest": manifest})
            if not bool(manifest.get("valid")):
                errors.extend(f"{directory.name}: {error}" for error in manifest.get("errors", []))
    expected = {"e2c-confirmatory", "e3c-confirmatory", "e4c-confirmatory", "e5c-confirmatory", "e7c-confirmatory"}
    present = {str(item["run"]) for item in runs}
    missing_confirmatory = sorted(expected - present)
    if strict and missing_confirmatory:
        errors.append("missing confirmatory runs: " + ", ".join(missing_confirmatory))
    report = {"valid": not errors, "strict": strict, "runs": runs, "missing_confirmatory": missing_confirmatory, "errors": errors}
    output = root / "artifacts/v9/validation.json"
    write_json(output, report)
    return report


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PIVOT V9 artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate_root(args.root.resolve(), strict=args.strict)
    print(json.dumps({"valid": report["valid"], "runs": len(report["runs"]), "errors": len(report["errors"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
