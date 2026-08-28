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
SEALED_ARCHIVE_MEMBERS = {
    "configs/v15/task_manifest.json",
    "experiments/v15/confirmatory_lock_history.jsonl",
}
IDENTITY_PATTERNS = (
    r"\\author\s*\{[^}]*@",
    r"\\author\s*\{(?!\s*anonymous)[^}]+\}",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
)
ARCHIVE_PRIVATE_CONTENT_RE = re.compile(rb"(?i)/(?:opt/projects|home/ubuntu|tmp)/")
ARCHIVE_CREDENTIAL_RE = re.compile(rb"(?:sk-[A-Za-z0-9]{12,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{12,})")
ARCHIVE_ASSISTANT_RE = re.compile(rb"(?i)\bcodex\b")


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
        "paper_topic_tokens": all(
            token in lowered
            for token in (
                "improvement fidelity",
                "pivot",
                "replacement operation",
                "decision preservation under differential error",
            )
        ),
    }


def audit_archive_members(members: Iterable[str]) -> dict[str, Any]:
    """Check a supplementary archive member list for private/raw artifacts."""

    names = [name.replace("\\", "/") for name in members]
    private = [
        name
        for name in names
        if any(token.casefold() in name.casefold() for token in PRIVATE_PATH_PATTERNS)
    ]
    generated_parquet = [
        name
        for name in names
        if name.casefold().endswith(".parquet")
        and name.casefold().startswith(("results/v15/", "figures/v15/", "artifacts/v15/"))
    ]
    raw = [
        name
        for name in names
        if name.casefold().endswith(RAW_ARCHIVE_SUFFIXES) and name not in generated_parquet
    ]
    sealed = [name for name in names if name in SEALED_ARCHIVE_MEMBERS]
    return {
        "no_private_paths": not private,
        "no_raw_archives": not raw,
        "no_sealed_inputs": not sealed,
        "has_readme": any(name.casefold().endswith("readme.md") for name in names),
        "has_snapshot_manifest": "snapshot/manifest.json" in names,
        "private_members": not private,
        "raw_members": not raw,
        "sealed_members": sealed,
        "generated_parquet_members": generated_parquet,
    }


def audit_archive_contents(archive_path: Path) -> dict[str, Any]:
    """Scan member bytes for private paths or unreplaced credentials.

    The member-name audit cannot see values embedded in JSONL or binary
    Parquet.  This second pass intentionally uses narrow absolute-path and
    credential signatures so ordinary public URLs and source identifiers are
    not treated as leaks.
    """

    private_hits: list[str] = []
    credential_hits: list[str] = []
    assistant_hits: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                data = archive.read(name)
                if ARCHIVE_PRIVATE_CONTENT_RE.search(data):
                    private_hits.append(name)
                if ARCHIVE_CREDENTIAL_RE.search(data):
                    credential_hits.append(name)
                if ARCHIVE_ASSISTANT_RE.search(data):
                    assistant_hits.append(name)
    except (OSError, zipfile.BadZipFile):
        return {
            "valid": False,
            "private_content_hits": [],
            "credential_content_hits": [],
            "assistant_content_hits": [],
        }
    return {
        "valid": not private_hits and not credential_hits and not assistant_hits,
        "private_content_hits": sorted(private_hits),
        "credential_content_hits": sorted(credential_hits),
        "assistant_content_hits": sorted(assistant_hits),
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

    inventory_fields = {"generated_parquet_members", "private_members", "raw_members", "sealed_members"}
    blocking: list[str] = []
    for category in ("machine_checks", "manual_gates", "scientific_gates"):
        values = report.get(category, {})
        for name, value in values.items():
            if name.removeprefix("archive_") in inventory_fields:
                continue
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


def _portable_path(path: Path) -> str:
    """Render a report path without leaking the local checkout prefix."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _compact_pdf_text(text: str) -> str:
    """Normalize style small-caps spacing for robust phrase checks."""

    return re.sub(r"\s+", "", text.casefold())


def _pdf_pages(pdf: Path) -> int:
    info = _command(["pdfinfo", str(pdf)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, flags=re.MULTILINE)
    if match is None:
        raise ValueError("pdfinfo did not report page count")
    return int(match.group(1))


def _aux_path(pdf: Path, aux: Path | None = None) -> Path:
    """Resolve the LaTeX sidecar for a copied submission PDF.

    The release PDF is intentionally copied without build intermediates. When
    no explicit sidecar is supplied, look next to the PDF first and then in
    the conventional ``build/main.aux`` location used by ``paper/iclr2027``.
    """

    candidates = [aux] if aux is not None else [pdf.with_suffix(".aux"), pdf.parent / "build" / "main.aux"]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"LaTeX aux sidecar is missing; searched: {searched}")


def _appendix_page(pdf: Path, aux: Path | None = None) -> int:
    aux_file = _aux_path(pdf, aux)
    text = aux_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\\newlabel\{app:artifact\}\{\{[^{}]*\}\{(\d+)\}", text)
    if match is None:
        raise ValueError("appendix label app:artifact is missing from LaTeX aux")
    return int(match.group(1))


def _references_page(pdf: Path, aux: Path | None = None) -> int:
    """Read the bibliography boundary excluded by the ICLR page limit."""

    aux_file = _aux_path(pdf, aux)
    text = aux_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\\newlabel\{refs:start\}\{\{[^{}]*\}\{(\d+)\}", text)
    if match is None:
        raise ValueError("references label refs:start is missing from LaTeX aux")
    return int(match.group(1))


def audit_submission(
    *,
    pdf: Path,
    source: Path,
    supplement: Path,
    style_dir: Path,
    output: Path,
    aux: Path | None = None,
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
    archive_content_checks = audit_archive_contents(supplement)
    page_count = _pdf_pages(pdf)
    aux_path = _aux_path(pdf, aux)
    references_page = _references_page(pdf, aux_path)
    appendix_page = _appendix_page(pdf, aux_path)
    main_pages = references_page - 1
    style_manifest = style_dir.parent / "style_manifest.json"
    machine_checks = {
        "pdf_exists": pdf.is_file() and pdf.stat().st_size > 0,
        "pdf_text_nonempty": len(pdf_text.strip()) > 500,
        "pdf_page_count_present": page_count >= 1,
        "main_text_within_nine_pages": main_pages <= max_main_pages,
        "references_before_appendix": references_page < appendix_page,
        "pdf_ai_use_statement": "aiusestatement" in pdf_compact,
        "pdf_reproducibility_statement": "reproducibilitystatement" in pdf_compact,
        "pdf_anonymous": "anonymous authors" in pdf_lower and "paper under double-blind review" in pdf_lower,
        "style_files_present": all(any(style_dir.glob(pattern)) for pattern in ("*.sty", "*.bst")),
        "style_hashes_match": style_manifest.is_file()
        and audit_style_hashes(style_dir, style_manifest),
        "supplement_exists": supplement.is_file() and supplement.stat().st_size > 0,
        "supplement_archive": all(
            bool(archive_checks.get(name))
            for name in (
                "no_private_paths",
                "no_raw_archives",
                "no_sealed_inputs",
                "has_readme",
                "has_snapshot_manifest",
            )
        ),
        "supplement_content_clean": bool(archive_content_checks.get("valid", False)),
    }
    machine_checks.update({f"source_{name}": value for name, value in source_checks.items()})
    machine_checks.update(
        {
            f"archive_{name}": value
            for name, value in archive_checks.items()
            if name not in {"private_members", "raw_members", "sealed_members", "generated_parquet_members"}
        }
    )
    report: dict[str, Any] = {
        "package": "PIVOT ICLR 2027 anonymous submission package",
        "pdf": _portable_path(pdf),
        "source": _portable_path(source),
        "supplement": _portable_path(supplement),
        "aux": _portable_path(aux_path),
        "pdf_pages": page_count,
        "appendix_start_page": appendix_page,
        "references_start_page": references_page,
        "main_pages": main_pages,
        "machine_checks": machine_checks,
        "manual_gates": {
            "openreview_profile": "pending",
            "author_quota_and_reciprocal_review": "pending",
            "final_author_metadata_and_conflicts": "pending",
            "ai_use_disclosure_in_submission_form": "pending",
            "no_parallel_submission_confirmation": "pending",
        },
        "scientific_gates": {
            # These labels distinguish a frozen bounded result from a claim
            # that would require a stronger causal or ecological study.
            "external_interactive_response": "bounded_null",
            "external_strategic_validation": "scoped_pass",
            "confirmatory_update_rule_and_holdout": "pass",
        },
        "archive_members": members,
        "archive_content_checks": archive_content_checks,
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
    parser.add_argument("--aux", type=Path, help="LaTeX aux sidecar when the PDF was copied from the build")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_submission(
        pdf=args.pdf,
        source=args.source,
        supplement=args.supplement,
        style_dir=args.style_dir,
        output=args.output,
        aux=args.aux,
    )
    print(json.dumps({"decision": report["decision"], "blocking_gates": report["blocking_gates"]}))


if __name__ == "__main__":
    main()
