#!/usr/bin/env python3
"""Verify the reproducible PIVOT paper PDF and its source-level invariants.

The verifier intentionally checks the rendered artifact rather than trusting a
successful LaTeX exit code.  In particular, the nine-page ICLR main-text gate
is read from the references and appendix labels in the generated ``.aux``
file, while the complete PDF may contain a bibliography and appendix after
that boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def parse_appendix_start_page(aux_text: str) -> int:
    """Return the page on which the labelled appendix starts.

    ``hyperref`` can add a fourth field to ``\\newlabel`` records, so the
    expression deliberately only relies on the first two fields.
    """

    match = re.search(r"\\newlabel\{app:artifact\}\{\{[^{}]*\}\{(\d+)\}", aux_text)
    if match is None:
        raise ValueError("appendix label app:artifact is missing from LaTeX aux")
    return int(match.group(1))


def parse_references_start_page(aux_text: str) -> int:
    """Return the first bibliography page used for the ICLR page gate."""

    match = re.search(r"\\newlabel\{refs:start\}\{\{[^{}]*\}\{(\d+)\}", aux_text)
    if match is None:
        raise ValueError("references label refs:start is missing from LaTeX aux")
    return int(match.group(1))


def _portable_path(path: Path) -> str:
    """Render a report path without leaking the local checkout prefix."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def scan_log(log_text: str) -> dict[str, Any]:
    """Extract warning counts that should block a paper handoff."""

    undefined_references = bool(
        re.search(r"(?:Reference|citation) `[^']+' undefined", log_text, flags=re.IGNORECASE)
    )
    overfull_hboxes = len(re.findall(r"Overfull \\hbox", log_text))
    return {
        "undefined_references": undefined_references,
        "overfull_hboxes": overfull_hboxes,
        "rerun_warning": "Rerun to get cross-references right" in log_text,
    }


def verify_paper(
    pdf: Path,
    source: Path,
    output: Path,
    preview: Path | None = None,
    max_main_pages: int = 9,
) -> dict[str, Any]:
    """Run PDF, source, font, text, figure, and warning checks.

    The function raises ``RuntimeError`` on a failed gate and always writes a
    JSON report when the caller supplies an output path.
    """

    pdf = pdf.resolve()
    source = source.resolve()
    output = output.resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if not source.is_file():
        raise FileNotFoundError(source)

    pdfinfo = _command(["pdfinfo", str(pdf)])
    pages = _parse_int_field(pdfinfo, "Pages")
    title = _parse_text_field(pdfinfo, "Title")
    author = _parse_text_field(pdfinfo, "Author")
    text = _command(["pdftotext", "-layout", str(pdf), "-"])
    font_output = _command(["pdffonts", str(pdf)]) if shutil.which("pdffonts") else ""
    fonts = _parse_fonts(font_output)

    aux = pdf.with_suffix(".aux")
    if not aux.is_file():
        raise FileNotFoundError(f"expected LaTeX aux next to PDF: {aux}")
    aux_text = aux.read_text(encoding="utf-8", errors="replace")
    references_page = parse_references_start_page(aux_text)
    appendix_page = parse_appendix_start_page(aux_text)
    main_pages = references_page - 1

    log = pdf.with_suffix(".log")
    log_scan = scan_log(log.read_text(encoding="utf-8", errors="replace")) if log.is_file() else {}
    source_text = source.read_text(encoding="utf-8")
    required_tokens = [
        "improvement reversal",
        "improvement fidelity",
        "pivot",
        "cumulative true improvement",
        "finance audit",
        "replacement operation",
        "rank policies correctly while ranking improvements incorrectly",
        "contribution 1",
        "contribution 2",
        "contribution 3",
        "operator shift bound",
        "finite-sample best-update identification",
        "decision preservation under differential error",
        "why transition validation differs from active learning",
        "fig3_pivot_voi",
        "stress tests beyond controlled environments",
        "value fidelity versus improvement fidelity",
        "operator-relative improvement fidelity",
        "q_{\\mathcal a}",
        "raw sampled reversal-rate cells",
        "false improvement (improvement reversal)",
    ]
    source_lower = source_text.casefold()
    missing_tokens = [token for token in required_tokens if token.casefold() not in source_lower]

    scientific_source = source_text.split("\\begin{document}", 1)[-1].split("\\appendix", 1)[0]
    forbidden_version_tokens = sorted(
        set(
            re.findall(
                r"\b(?:v7|v8|v9|v10|e2c|e3c|e4c|e5c|e7c|hypothesis_[a-z_]+)\b",
                scientific_source,
                flags=re.IGNORECASE,
            )
        )
    )
    proposition_count = len(re.findall(r"\\begin\{proposition\}", source_text))
    required_assets = {
        "fig1_improvement_reversal.png",
        "fig2_operator_shift.png",
        "fig3_pivot_voi.pdf",
        "fig4_evidence_efficiency.png",
        "fig5_closed_loop.png",
        "figA_response_footprint.png",
        "figB_learned_ood_null.png",
        "figC_posterior_robustness.png",
        "figD_strategic_distribution.png",
        "figE_finance_boundary.png",
    }
    asset_missing = [
        name
        for name in sorted(required_assets)
        if not (
            (source.parent / "figures" / "v10" / name).is_file()
            or (source.parent / "figures" / name).is_file()
        )
    ]

    preview_path = _render_preview(pdf, preview) if preview is not None else None
    checks = {
        "pdf_exists": pdf.stat().st_size > 0,
        "text_nonempty": len(text.strip()) > 500,
        "main_pages_within_limit": main_pages <= max_main_pages,
        "references_before_appendix": references_page < appendix_page,
        "appendix_after_main": appendix_page > references_page,
        "anonymous_author": author in {"", "-", "Anonymous", "Anonymous Authors"},
        "embedded_fonts": bool(fonts) and all(font["embedded"] for font in fonts),
        "required_source_tokens": not missing_tokens,
        "six_propositions": proposition_count == 6,
        "no_internal_version_language_in_main": not forbidden_version_tokens,
        "required_figure_assets": not asset_missing,
        "no_undefined_references": not log_scan.get("undefined_references", False),
        "no_overfull_hboxes": log_scan.get("overfull_hboxes", 0) == 0,
        "preview_nonempty": preview_path is None or preview_path.stat().st_size > 1000,
    }
    report: dict[str, Any] = {
        "pdf": _portable_path(pdf),
        "source": _portable_path(source),
        "bytes": pdf.stat().st_size,
        "pages": pages,
        "main_pages": main_pages,
        "references_start_page": references_page,
        "appendix_start_page": appendix_page,
        "title": title,
        "author": author,
        "fonts": fonts,
        "log": log_scan,
        "missing_source_tokens": missing_tokens,
        "forbidden_version_tokens": forbidden_version_tokens,
        "proposition_count": proposition_count,
        "missing_figure_assets": asset_missing,
        "preview": _portable_path(preview_path) if preview_path else None,
        "checks": checks,
        "valid": all(checks.values()) and pages >= appendix_page,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["valid"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"paper verification failed: {', '.join(failed)}")
    return report


def _command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}") from exc
    return result.stdout


def _parse_int_field(text: str, field: str) -> int:
    match = re.search(rf"^{re.escape(field)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"pdfinfo field missing: {field}")
    return int(match.group(1))


def _parse_text_field(text: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s?(.*)$", text, flags=re.MULTILINE)
    return "" if match is None else match.group(1).strip()


def _parse_fonts(text: str) -> list[dict[str, Any]]:
    fonts: list[dict[str, Any]] = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        # The type column can itself contain spaces (for example, ``Type 1``).
        # The five trailing fields are always emb, sub, uni, object, and ID.
        fonts.append({"name": fields[0], "type": fields[1], "embedded": fields[-5] == "yes"})
    return fonts


def _render_preview(pdf: Path, target: Path) -> Path:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pivot-paper-preview-") as temp_dir:
        prefix = Path(temp_dir) / "page"
        _command(
            [
                "gs",
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=png16m",
                "-r120",
                f"-sOutputFile={prefix}-%02d.png",
                f"{pdf}",
            ]
        )
        first = Path(f"{prefix}-01.png")
        if not first.is_file():
            raise RuntimeError("ghostscript did not render a first-page preview")
        shutil.copyfile(first, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a PIVOT paper PDF")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--max-main-pages", type=int, default=9)
    args = parser.parse_args()
    report = verify_paper(args.pdf, args.source, args.output, args.preview, args.max_main_pages)
    print(
        json.dumps(
            {"valid": report["valid"], "main_pages": report["main_pages"], "pages": report["pages"]}
        )
    )


if __name__ == "__main__":
    main()
