"""Audit V9 figure/source/metadata bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import sha256, write_json

REQUIRED = ("pdf", "svg", "png", "csv", "parquet", "meta.json")


def audit(root: Path) -> dict[str, Any]:
    directory = root / "figures/v9"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    manifest_path = directory / "figure_manifest.json"
    if not manifest_path.is_file():
        errors.append("figure_manifest.json missing")
    for metadata_path in sorted(directory.glob("*.meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = metadata_path.name.removesuffix(".meta.json")
        missing = [suffix for suffix in REQUIRED if not (directory / f"{stem}.{suffix}").is_file()]
        if missing:
            errors.append(f"{stem}: missing {','.join(missing)}")
        for input_path, digest in metadata.get("input_sha256", {}).items():
            target = root / input_path
            if not target.is_file() or sha256(target) != digest:
                errors.append(f"{stem}: input hash mismatch {input_path}")
        if not metadata.get("experiment_ids") or not metadata.get("style_version"):
            errors.append(f"{stem}: incomplete metadata")
        records.append({"figure_id": stem, "missing": missing, "main_text": bool(metadata.get("main_text")), "valid": not missing})
    report = {"valid": not errors, "figure_count": len(records), "records": records, "errors": errors}
    write_json(root / "artifacts/v9/figure_audit.json", report)
    (root / "V9_FIGURE_AUDIT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PIVOT V9 Figure Audit", "", f"Status: **{'PASS' if report['valid'] else 'FAIL'}**", "", "| Figure | Main text | Valid |", "| --- | --- | --- |"]
    for record in report["records"]:
        lines.append(f"| `{record['figure_id']}` | {record['main_text']} | {record['valid']} |")
    if report["errors"]:
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in report["errors"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V9 figures")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root.resolve())
    print(json.dumps({"valid": report["valid"], "figure_count": report["figure_count"], "errors": len(report["errors"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
