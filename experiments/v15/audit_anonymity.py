from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit

PRIVATE_RE = re.compile(r"(?:/opt/projects|/home/ubuntu|/tmp/[A-Za-z0-9_.-]+)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AUTHOR_RE = re.compile(r"\\author\s*\{\s*[^}]+\}")


def audit_anonymity(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    source_path = root / "paper/iclr2027/main.tex"
    source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    pdf_path = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    try:
        pdf_text = subprocess.check_output(["pdftotext", str(pdf_path), "-"], text=True)
    except (OSError, subprocess.CalledProcessError):
        pdf_text = ""
    supplement = root / "paper/iclr2027/pivot_iclr2027_supplementary.zip"
    archive_text = ""
    names: list[str] = []
    if supplement.is_file():
        with zipfile.ZipFile(supplement) as archive:
            names = archive.namelist()
            for name in names:
                if name.casefold().endswith((".md", ".tex", ".json", ".yaml", ".yml", ".txt", ".bib")):
                    archive_text += archive.read(name).decode("utf-8", errors="replace") + "\n"
    all_text = source + pdf_text + archive_text + "\n".join(names)
    private_hits = sorted(set(PRIVATE_RE.findall(all_text)))
    email_hits = sorted(set(EMAIL_RE.findall(all_text)))
    author_hits = AUTHOR_RE.findall(source)
    result: dict[str, Any] = {
        "pdf_exists": pdf_path.is_file(),
        "supplement_exists": supplement.is_file(),
        "private_path_hits": private_hits,
        "email_hits": email_hits,
        "nonempty_author_commands": author_hits,
        "valid": source_path.is_file() and pdf_path.is_file() and supplement.is_file() and not private_hits and not email_hits and not author_hits,
        "manual_platform_gate": "OpenReview profile, conflicts, affiliations, and upload metadata",
    }
    return write_audit(
        root,
        "anonymity_audit",
        result,
        "Anonymity Audit",
        "The source, rendered PDF, and supplementary archive are scanned for identity, local paths, and contact metadata.",
    )


if __name__ == "__main__":
    cli(audit_anonymity, "Audit anonymous submission artifacts")
