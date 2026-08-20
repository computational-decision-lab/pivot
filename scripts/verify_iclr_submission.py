#!/usr/bin/env python3
"""Audit the local ICLR 2027 PIVOT submission package.

The auditor separates deterministic local checks from gates that only the
authors can close in OpenReview.  A passing local PDF is therefore not
silently reported as a submitted paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PRIVATE_PATH_PATTERNS = (
    "/opt/projects",
    "/home/ubuntu",
    "/tmp/",
    "\\\\Users\\\\",
    "api_key",
    "api-key",
    "secret",
)
RAW_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".parquet", ".feather")
IDENTITY_PATTERNS = (
    r"\\author\s*\{[^}]*@",
    r"\\author\s*\{(?!\s*anonymous)[^}]+\}",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
)


def audit_source_text(source: str) -> dict[str, bool]:
    """Check source-level anonymity and required ICLR statements."""

    lowered = source.casefold()
    identity_leak = any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in IDENTITY_PATTERNS)
    private_leak = any(token.casefold() in lowered for token in PRIVATE_PATH_PATTERNS)
    return {
        "official_style": "iclr2027_conference" in lowered,
        "ai_use_statement": "ai use statement" in lowered,
        "reproducibility_statement": "reproducibility statement" in lowered,
        "no_final_copy": r"\iclrfinalcopy" not in source,
        "no_identity_leak": not identity_leak,
        "no_private_path": not private_leak,
        "paper_topic_tokens": all(token in lowered for token in ("improvement fidelity", "pivot")),
    }


def audit_archive_members(members: Iterable[str]) -> dict[str, bool]:
    """Check a supplementary archive member list for private/raw artifacts."""

    names = [name.replace("\\", "/") for name in members]
    private = [
        name
        for name in names
        if any(token.casefold() in name.casefold() for token in PRIVATE_PATH_PATTERNS)
    ]
    raw = [name for name in names if name.casefold().endswith(RAW_ARCHIVE_SUFFIXES)]
    return {
        "no_private_paths": not private,
        "no_raw_archives": not raw,
        "has_readme": any(name.casefold().endswith("readme.md") for name in names),
        "has_snapshot_manifest": "snapshot/manifest.json" in names,
        "private_members": not private,
        "raw_members": not raw,
    }


def audit_style_hashes(style_dir: Path, manifest_path: Path) -> bool:
    """Confirm local style files match the recorded official archive hashes."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    if not isinstance(expected, dict) or not expected:
        return False
    for name, digest in expected.items():
        candidate = style_dir / name
        if not candidate.is_file():
            return False
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            return False
    return True


def build_decision(report: dict[str, Any]) -> dict[str, Any]:
    """Return GO only when machine, manual, and scientific gates all pass."""

    blocking: list[str] = []
    for category in ("machine_checks", "manual_gates", "scientific_gates"):
        values = report.get(category, {})
        for name, value in values.items():
            passed = value is True or str(value).casefold() in {"pass", "passed", "true"}
            if not passed:
                blocking.append(f"{category}.{name}")
    ready = not blocking
    return {
        "decision": "GO" if ready else "CONDITIONAL GO",
        "submission_ready": ready,
        "blocking_gates": blocking,
    }


def _command(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _pdf_text(pdf: Path) -> str:
    return _command(["pdftotext", "-layout", str(pdf), "-"])


def _compact_pdf_text(text: str) -> str:
    """Normalize style small-caps spacing for robust phrase checks."""

    return re.sub(r"\s+", "", text.casefold())


def _pdf_pages(pdf: Path) -> int:
    info = _command(["pdfinfo", str(pdf)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, flags=re.MULTILINE)
    if match is None:
        raise ValueError("pdfinfo did not report page count")
    return int(match.group(1))


def _appendix_page(pdf: Path) -> int:
    aux = pdf.with_suffix(".aux")
    text = aux.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\\newlabel\{app:artifact\}\{\{[^{}]*\}\{(\d+)\}", text)
    if match is None:
        raise ValueError("appendix label app:artifact is missing from LaTeX aux")
    return int(match.group(1))


def audit_submission(
    *,
    pdf: Path,
    source: Path,
    supplement: Path,
    style_dir: Path,
    output: Path,
    max_main_pages: int = 9,
) -> dict[str, Any]:
    """Run local checks and write a JSON decision report."""

    source_text = source.read_text(encoding="utf-8")
    source_checks = audit_source_text(source_text)
    pdf_text = _pdf_text(pdf)
    pdf_lower = pdf_text.casefold()
    pdf_compact = _compact_pdf_text(pdf_text)
    with zipfile.ZipFile(supplement) as archive:
        members = archive.namelist()
    archive_checks = audit_archive_members(members)
    page_count = _pdf_pages(pdf)
    appendix_page = _appendix_page(pdf)
    main_pages = appendix_page - 1
    style_manifest = style_dir.parent / "style_manifest.json"
    machine_checks = {
        "pdf_exists": pdf.is_file() and pdf.stat().st_size > 0,
        "pdf_text_nonempty": len(pdf_text.strip()) > 500,
        "pdf_page_count_present": page_count >= 1,
        "main_text_within_nine_pages": main_pages <= max_main_pages,
        "pdf_ai_use_statement": "aiusestatement" in pdf_compact,
        "pdf_reproducibility_statement": "reproducibilitystatement" in pdf_compact,
        "pdf_anonymous": "anonymous authors" in pdf_lower and "paper under double-blind review" in pdf_lower,
        "style_files_present": all(any(style_dir.glob(pattern)) for pattern in ("*.sty", "*.bst")),
        "style_hashes_match": style_manifest.is_file()
        and audit_style_hashes(style_dir, style_manifest),
        "supplement_exists": supplement.is_file() and supplement.stat().st_size > 0,
        "supplement_archive": all(archive_checks.values()),
    }
    machine_checks.update({f"source_{name}": value for name, value in source_checks.items()})
    machine_checks.update(
        {
            f"archive_{name}": value
            for name, value in archive_checks.items()
            if name not in {"private_members", "raw_members"}
        }
    )
    report: dict[str, Any] = {
        "package": "PIVOT ICLR 2027 anonymous submission package",
        "pdf": str(pdf.resolve()),
        "source": str(source.resolve()),
        "supplement": str(supplement.resolve()),
        "pdf_pages": page_count,
        "appendix_start_page": appendix_page,
        "main_pages": main_pages,
        "machine_checks": machine_checks,
        "manual_gates": {
            "openreview_profile": "pending",
            "author_quota_and_reciprocal_review": "pending",
            "final_author_metadata_and_conflicts": "pending",
            "no_parallel_submission_confirmation": "pending",
        },
        "scientific_gates": {
            "external_interactive_response": "open",
            "external_strategic_validation": "open",
            "confirmatory_update_rule_and_holdout": "open",
        },
        "archive_members": members,
    }
    report.update(build_decision(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the PIVOT ICLR submission package")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--style-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_submission(
        pdf=args.pdf,
        source=args.source,
        supplement=args.supplement,
        style_dir=args.style_dir,
        output=args.output,
    )
    print(json.dumps({"decision": report["decision"], "blocking_gates": report["blocking_gates"]}))


if __name__ == "__main__":
    main()
