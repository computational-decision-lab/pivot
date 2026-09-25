from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from figures.v15.audit import audit as audit_figures

from .audit_anonymity import audit_anonymity
from .audit_claims import audit_claims
from .audit_language import audit_language
from .audit_numbers import audit_numbers
from .audit_references import audit_references
from .audit_reproducibility import audit_reproducibility
from .audit_support import cli, write_audit
from .audit_terminal_states import audit_terminal_states
from .figure_pipeline import bundle_figures
from .reports import generate_reports


def _submission_verify(root: Path) -> dict[str, Any]:
    script = root / "scripts/verify_iclr_submission.py"
    pdf = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    supplement = root / "paper/iclr2027/pivot_iclr2027_supplementary.zip"
    if not script.is_file() or not pdf.is_file() or not supplement.is_file():
        return {"status": "NOT_RUN", "reason": "submission inputs missing"}
    command = [
        str(root / ".venv/bin/python"),
        str(script),
        "--pdf",
        str(pdf),
        "--source",
        str(root / "paper/iclr2027/main.tex"),
        "--supplement",
        str(supplement),
        "--style-dir",
        str(root / "paper/iclr2027/style"),
        "--aux",
        str(root / "paper/iclr2027/build/main.aux"),
        "--output",
        str(root / "paper/iclr2027/submission_verification.json"),
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {"status": "ERROR", "error": str(exc)}
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def finalize(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    # Rendering is not a visual sign-off.  ``bundle_figures`` promotes assets
    # only when the hash-bound review manifest created after a human
    # render/view/fix pass is present.
    figures = bundle_figures(root)
    figure_audit = audit_figures(root, all_figures=True)
    reports = generate_reports(root)
    audits = {
        "figures": figure_audit,
        "language": audit_language(root),
        "numbers": audit_numbers(root),
        "claims": audit_claims(root),
        "references": audit_references(root),
        "anonymity": audit_anonymity(root),
        "reproducibility": audit_reproducibility(root),
        "terminal_states": audit_terminal_states(root),
    }
    verify = _submission_verify(root)
    audit_failures = [name for name, value in audits.items() if not bool(value.get("valid", False))]
    status = "BLOCKED" if reports.get("status") == "BLOCKED" or audit_failures or verify.get("status") == "FAIL" else "READY_WITH_MINOR_MANUAL_CHECKS"
    payload: dict[str, Any] = {
        "status": status,
        "scientific_report": reports,
        "figure_count": figures.get("figure_count", 0),
        "audits": audits,
        "submission_verification": verify,
        "audit_failures": audit_failures,
    }
    return write_audit(
        root,
        "finalize",
        payload,
        "Finalization Audit",
        f"Local package status: **{status}**. External modern-agent evidence remains separate from local paper compliance.",
    )


if __name__ == "__main__":
    cli(finalize, "Run the final V15 artifact and submission audit")
