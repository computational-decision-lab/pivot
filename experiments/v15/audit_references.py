from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit

ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\s*\{([^}]+)\}")


def audit_references(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    bib_path = root / "paper/iclr2027/references.bib"
    source_path = root / "paper/iclr2027/main.tex"
    bib = bib_path.read_text(encoding="utf-8") if bib_path.is_file() else ""
    source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    keys = list(ENTRY_RE.findall(bib))
    cited = {
        key.strip()
        for group in CITE_RE.findall(source)
        for key in group.split(",")
        if key.strip()
    }
    unique_keys = set(keys)
    missing = sorted(cited - unique_keys)
    duplicate_keys = sorted({key for key in unique_keys if keys.count(key) > 1})
    result: dict[str, Any] = {
        "bibliography_exists": bib_path.is_file(),
        "entry_count": len(keys),
        "unique_entry_count": len(unique_keys),
        "cited_entry_count": len(cited),
        "missing_citation_keys": missing,
        "duplicate_entry_keys": duplicate_keys,
        "valid": bib_path.is_file() and source_path.is_file() and not missing and not duplicate_keys,
    }
    return write_audit(
        root,
        "reference_audit",
        result,
        "Reference Audit",
        "Citation keys are reconciled against the bundled bibliography; this local audit does not invent external metadata.",
    )


if __name__ == "__main__":
    cli(audit_references, "Audit bibliography and citation keys")
