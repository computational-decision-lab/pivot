from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit
from .reports import _number_audit


def audit_numbers(root: Path) -> dict[str, Any]:
    payload = _number_audit(Path(root).resolve())
    return write_audit(
        Path(root),
        "number_audit",
        payload,
        "Number Audit",
        "Semantic macros are compared with the frozen source decisions, including family-balanced strategic aggregation.",
    )


if __name__ == "__main__":
    cli(audit_numbers, "Audit paper numbers")
