from __future__ import annotations

import re
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit

_INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_VERSION_RE = re.compile(r"\b(?:v\d+)\b", flags=re.IGNORECASE)
_CODEX_RE = re.compile(r"codex", flags=re.IGNORECASE)


def _candidate_include_paths(source_file: Path, include_name: str) -> Iterable[Path]:
    """Yield safe local paths for a TeX include with or without an extension."""

    requested = Path(include_name.strip())
    if requested.is_absolute():
        return ()
    base = (source_file.parent / requested).resolve()
    candidates = [base]
    if base.suffix == "":
        candidates.extend(base.with_suffix(extension) for extension in (".tex", ".ltx", ".txt"))
    return tuple(candidate for candidate in candidates if candidate.is_file())


def _body_source_closure(source_path: Path) -> tuple[dict[str, str], str]:
    """Read the main body and transitively included text sources.

    The preamble is intentionally excluded from the main source's scan because
    package comments and build metadata are not manuscript prose.  Included
    files are scanned in full: they are body dependencies by construction.
    """

    if not source_path.is_file():
        return {}, ""
    root = source_path.resolve().parent
    seen: set[Path] = set()
    texts: dict[str, str] = {}
    queue: list[tuple[Path, str]] = []
    source = source_path.read_text(encoding="utf-8", errors="replace")
    body = source.split("\\begin{document}", 1)[-1].split("\\appendix", 1)[0]
    queue.append((source_path.resolve(), body))
    while queue:
        path, text = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        texts[relative] = text
        for include_name in _INCLUDE_RE.findall(text):
            for candidate in _candidate_include_paths(path, include_name):
                if candidate not in seen:
                    child = candidate.read_text(encoding="utf-8", errors="replace")
                    queue.append((candidate, child))
    return texts, "\n".join(texts.values())


def scan_language(root: Path) -> dict[str, Any]:
    """Return the language audit without writing any files."""

    root = Path(root).resolve()
    source_path = root / "paper/iclr2027/main.tex"
    body_sources, body = _body_source_closure(source_path)
    pdf_path = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    try:
        pdf_text = subprocess.check_output(["pdftotext", str(pdf_path), "-"], text=True)
    except (OSError, subprocess.CalledProcessError):
        pdf_text = ""
    supplement = root / "paper/iclr2027/pivot_iclr2027_supplementary.zip"
    supplement_names: list[str] = []
    supplement_text = ""
    if supplement.is_file():
        with zipfile.ZipFile(supplement) as archive:
            supplement_names = archive.namelist()
            for name in supplement_names:
                if name.casefold().endswith((".py", ".md", ".json", ".yaml", ".yml", ".tex", ".bib", ".txt", ".csv")):
                    supplement_text += archive.read(name).decode("utf-8", errors="replace") + "\n"
    release_root = root / "paper/iclr2027/figures/release"
    release_text = ""
    release_names: list[str] = []
    if release_root.is_dir():
        for path in sorted(release_root.iterdir()):
            release_names.append(path.name)
            if path.is_file() and path.suffix.casefold() in {".json", ".tex", ".txt", ".csv"}:
                release_text += path.read_text(encoding="utf-8", errors="replace") + "\n"
    reviewer_text = supplement_text + release_text + "\n".join(supplement_names + release_names)
    reviewer_codex_hits = sorted(set(_CODEX_RE.findall(reviewer_text)))
    body_source_hits = {
        name: sorted(set(_VERSION_RE.findall(text)))
        for name, text in body_sources.items()
        if _VERSION_RE.search(text)
    }
    payload = {
        "source_exists": source_path.is_file(),
        "pdf_exists": pdf_path.is_file(),
        "body_source_files": sorted(body_sources),
        "body_source_version_hits": body_source_hits,
        "body_version_tokens": sorted(set(_VERSION_RE.findall(body))),
        "pdf_version_tokens": sorted(set(_VERSION_RE.findall(pdf_text))),
        "paper_facing_codex_tokens": len(_CODEX_RE.findall(body + pdf_text)),
        "reviewer_artifact_codex_tokens": len(_CODEX_RE.findall(reviewer_text)),
        "reviewer_artifact_codex_hits": reviewer_codex_hits,
        "valid": not _VERSION_RE.search(body)
        and not _VERSION_RE.search(pdf_text)
        and not _CODEX_RE.search(body + pdf_text)
        and not reviewer_codex_hits,
    }
    return payload


def audit_language(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    payload = scan_language(root)
    return write_audit(
        root,
        "language_audit",
        payload,
        "Language and Paper-Facing Token Audit",
        "The manuscript body and rendered PDF are checked for internal iteration labels and implementation-assistant provenance.",
    )


if __name__ == "__main__":
    cli(audit_language, "Audit paper-facing language")
