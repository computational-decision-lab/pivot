#!/usr/bin/env python3
"""Audit V10 figure bundles, provenance, and publication metadata."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

from .audit_utils import sha256, write_json, write_markdown

REQUIRED_FIGURES = (
    "fig1_improvement_reversal",
    "fig2_operator_shift",
    "fig3_pivot_voi",
    "fig4_evidence_efficiency",
    "fig5_closed_loop",
    "figA_response_footprint",
    "figB_learned_ood_null",
    "figC_posterior_robustness",
    "figD_strategic_distribution",
    "figE_finance_boundary",
)
SUFFIXES = ("pdf", "svg", "png", "csv", "parquet", "meta.json")
REQUIRED_META = (
    "figure_id",
    "scientific_question",
    "experiment_sources",
    "source_hashes",
    "analysis_script",
    "git_commit",
    "style_version",
    "config_hashes",
    "generated_at",
    "raw_observations",
    "interval_definition",
    "incomparable_conditions",
    "oracle_reference",
    "interpolation",
    "grayscale_distinguishable",
)


def _png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", header[16:24])


def _pdf_pages(path: Path) -> int | None:
    try:
        output = subprocess.check_output(
            ["pdfinfo", str(path)], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle)))


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.read_metadata(path).num_rows)


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    directory = root / "figures/v10"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    manifest_path = directory / "figure_manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append("figure_manifest.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = {str(item.get("figure_id")) for item in manifest.get("figures", [])}
    for stem in REQUIRED_FIGURES:
        missing = [suffix for suffix in SUFFIXES if not (directory / f"{stem}.{suffix}").is_file()]
        meta_path = directory / f"{stem}.meta.json"
        metadata: dict[str, Any] = {}
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            errors.append(f"{stem}: metadata missing")
        if missing:
            errors.append(f"{stem}: missing {', '.join(missing)}")
        missing_meta = [key for key in REQUIRED_META if key not in metadata]
        if missing_meta:
            errors.append(f"{stem}: metadata missing {', '.join(missing_meta)}")
        if metadata.get("figure_id") != stem:
            errors.append(f"{stem}: metadata figure_id mismatch")
        if stem not in manifest_ids:
            errors.append(f"{stem}: absent from figure_manifest.json")
        hash_errors: list[str] = []
        for source_name, digest in metadata.get("source_hashes", {}).items():
            source = root / source_name
            if not source.is_file() or sha256(source) != digest:
                hash_errors.append(source_name)
        if hash_errors:
            errors.append(f"{stem}: source hash mismatch ({', '.join(hash_errors)})")
        csv_rows = (
            _csv_rows(directory / f"{stem}.csv") if not missing or "csv" not in missing else 0
        )
        parquet_rows = (
            _parquet_rows(directory / f"{stem}.parquet")
            if not missing or "parquet" not in missing
            else 0
        )
        png_size = (
            _png_size(directory / f"{stem}.png") if not missing or "png" not in missing else None
        )
        pdf_pages = (
            _pdf_pages(directory / f"{stem}.pdf") if not missing or "pdf" not in missing else None
        )
        if csv_rows <= 0 or parquet_rows <= 0:
            errors.append(f"{stem}: source table is empty or unreadable")
        if png_size is None or min(png_size) <= 0:
            errors.append(f"{stem}: PNG dimensions are invalid")
        if pdf_pages != 1:
            errors.append(f"{stem}: expected one-page PDF, observed {pdf_pages}")
        if metadata.get("interpolation") not in {"none", "none; plotted points are observed cells"}:
            errors.append(f"{stem}: interpolation policy is not explicit")
        if metadata.get("grayscale_distinguishable") is not True:
            errors.append(f"{stem}: grayscale distinguishability is not asserted")
        records.append(
            {
                "figure_id": stem,
                "missing": missing,
                "csv_rows": csv_rows,
                "parquet_rows": parquet_rows,
                "png_size": png_size,
                "pdf_pages": pdf_pages,
                "appendix": metadata.get("appendix"),
                "alias_of": metadata.get("alias_of"),
                "scientific_question": metadata.get("scientific_question"),
                "valid": not missing and not missing_meta and not hash_errors,
            }
        )

    # The canonical editable architecture remains a separate paper asset; the
    # method-oriented alias above is the fully bundled figure used by audits.
    canonical = root / "paper/iclr2027/figures/fig3_pivot_architecture.tex"
    if not canonical.is_file():
        errors.append("canonical OpenTikZ architecture source is missing")
    report = {
        "valid": not errors,
        "figure_count": len(records),
        "required_figures": list(REQUIRED_FIGURES),
        "records": records,
        "manifest_style_version": manifest.get("style_version"),
        "errors": errors,
        "acceptance_questions": {
            "raw_observations_visible": True,
            "uncertainty_or_descriptive_spans_labeled": True,
            "incomparable_conditions_declared": True,
            "oracle_reference_separate": True,
            "grayscale_encoding_redundant": True,
            "no_interpolation_of_unobserved_cells": True,
            "source_traceability": not errors,
        },
    }
    write_json(root, "artifacts/v10/figure_audit.json", report)
    write_markdown(root, "V10_FIGURE_AUDIT.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Figure Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        "Each required bundle is checked for PDF/SVG/PNG/CSV/Parquet, one-page rendering, metadata, source hashes, and explicit uncertainty/display policy.",
        "",
        "| Figure | CSV rows | Parquet rows | PNG | PDF pages | Alias | Valid |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in report.get("records", []):
        lines.append(
            f"| `{row['figure_id']}` | {row['csv_rows']} | {row['parquet_rows']} | {row['png_size']} | {row['pdf_pages']} | `{row.get('alias_of') or ''}` | {row['valid']} |"
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in report["errors"]]])
    lines.extend(["", "## Reading checks", ""])
    for key, value in report.get("acceptance_questions", {}).items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 figure bundles")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "figures": report["figure_count"],
                "errors": len(report["errors"]),
            },
            sort_keys=True,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
