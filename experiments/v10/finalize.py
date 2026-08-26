#!/usr/bin/env python3
"""Build and audit the complete PIVOT V10 ICLR submission package.

The command transforms frozen evidence only.  It never invokes a scientific
experiment runner or changes confirmatory seeds/configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.build_iclr_supplement import build_supplement, write_deterministic_zip
from scripts.build_opentikz_architecture import build_architecture
from scripts.build_v10_paper_snippets import build as build_snippets
from scripts.build_v10_tables import build as build_tables
from scripts.verify_iclr_submission import audit_submission

from .audit_bibliography import audit as audit_bibliography
from .audit_claims import audit as audit_claims
from .audit_figures import audit as audit_figures
from .audit_metric_scale import audit as audit_metric_scale
from .audit_numbers import audit as audit_numbers
from .audit_release import pre_final_audit, reviewer_audit
from .audit_utils import load_json, sha256, write_json, write_markdown
from .figures import build as build_figures

FROZEN_SOURCE_COMMIT = "8541638d9115fb71c1b4d780e702762a030e3f59"
EXPECTED_DECISIONS = {
    "e2c": "HYPOTHESIS_SUPPORTED",
    "e3c": "HYPOTHESIS_SUPPORTED",
    "e4c": "HYPOTHESIS_NOT_SUPPORTED",
    "e5c": "HYPOTHESIS_SUPPORTED",
    "e7c": "HYPOTHESIS_SUPPORTED",
}
_PRIVATE_TEMP_PATH = re.compile(r"/tmp/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.@+-]+)*")


def _publicize(value: Any, root: Path) -> Any:
    """Remove machine-local paths from committed machine-readable reports."""

    if isinstance(value, str):
        text = value.replace(str(root), "<project-root>")
        return _PRIVATE_TEMP_PATH.sub("<temporary>", text)
    if isinstance(value, list):
        return [_publicize(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _publicize(item, root) for key, item in value.items()}
    return value


def _run(command: list[str], root: Path, log: list[dict[str, Any]]) -> None:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    log.append(
        {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout.splitlines()[-30:],
            "stderr_tail": result.stderr.splitlines()[-30:],
        }
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).splitlines()[-50:])
        raise RuntimeError(f"command failed ({' '.join(command)}):\n{detail}")


def _validate_frozen_sources(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    decisions: dict[str, Any] = {}
    for name, expected in EXPECTED_DECISIONS.items():
        path = root / f"results/v9/{name}-confirmatory/scientific_decision.json"
        if not path.is_file():
            errors.append(f"missing {path.relative_to(root)}")
            continue
        payload = load_json(path)
        decisions[name] = {
            "status": payload.get("status"),
            "powered": payload.get("powered"),
            "design_valid": payload.get("design_valid"),
            "sha256": sha256(path),
        }
        if payload.get("status") != expected:
            errors.append(f"{name}: expected {expected}, found {payload.get('status')}")
        if not payload.get("powered") or not payload.get("design_valid"):
            errors.append(f"{name}: frozen decision is not powered/design-valid")
    finance = root / "results/raw/e6-public-calibration/summary.json"
    if not finance.is_file():
        errors.append("public finance summary is missing")
    else:
        payload = load_json(finance)
        if payload.get("causal_impact_identified") is not False:
            errors.append("finance artifact causal-impact boundary changed")
    finance_expansion = root / "paper/snapshot/summaries/public-expansion-summary.json"
    if not finance_expansion.is_file():
        errors.append("expanded public finance summary is missing")
    else:
        expansion = load_json(finance_expansion)
        if (
            expansion.get("n_primary_sessions") != 12
            or expansion.get("n_f1_positive_sessions") != 7
        ):
            errors.append("expanded public finance session counts changed")
        if expansion.get("causal_impact_identified") is not False:
            errors.append("expanded public finance causal-impact boundary changed")
        holdout = expansion.get("holdout", {})
        if (
            expansion.get("n_depth_reversal_sessions") != 0
            or holdout.get("n_depth_reversal_sessions") != 0
        ):
            errors.append("expanded public finance reversal boundary changed")
    snapshot_manifest = root / "snapshot/v10_pre_final/SHA256SUMS"
    checked = 0
    if not snapshot_manifest.is_file():
        errors.append("pre-final snapshot manifest is missing")
    else:
        for line in snapshot_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split(maxsplit=1)
            target = root / relative
            # A manifest cannot contain its own final digest.  Its external
            # binding is recorded in the pre-final audit instead.
            if target.resolve() == snapshot_manifest.resolve():
                continue
            if not target.is_file() or sha256(target) != digest:
                errors.append(f"snapshot hash mismatch: {relative}")
            checked += 1
    report = {
        "valid": not errors,
        "errors": errors,
        "scientific_experiments_rerun": False,
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "decisions": decisions,
        "snapshot_files_checked": checked,
        "snapshot_manifest_sha256": sha256(snapshot_manifest)
        if snapshot_manifest.is_file()
        else None,
    }
    write_json(root, "artifacts/v10/source_validation.json", report)
    if errors:
        raise RuntimeError("frozen source validation failed: " + "; ".join(errors))
    return report


def _build_publication_assets(root: Path) -> dict[str, Any]:
    paper = root / "paper/iclr2027"
    architecture = build_architecture(
        paper / "figures/fig3_pivot_architecture.tex",
        paper / "figures/fig3_pivot_architecture.pdf",
        paper / "figures/fig3_pivot_architecture.svg",
        root / ".tools/opentikz",
    )
    snippets = build_snippets(root)
    tables = build_tables(root, root / "paper/tables")
    figures = build_figures(root)
    return {
        "architecture": architecture,
        "snippets": str(snippets.relative_to(root)),
        "tables": [str(path.relative_to(root)) for path in tables],
        "figures": [item["figure_id"] for item in figures],
    }


def _run_audits(root: Path) -> dict[str, Any]:
    reports = {
        "pre_final": pre_final_audit(root),
        "metric_scale": audit_metric_scale(root),
        "numbers": audit_numbers(root),
        "figures": audit_figures(root),
        "claims": audit_claims(root),
        "bibliography": audit_bibliography(root),
        "reviewer_attacks": reviewer_audit(root),
    }
    failed = [name for name, report in reports.items() if not report.get("valid")]
    if failed:
        raise RuntimeError("V10 audits failed: " + ", ".join(failed))
    return reports


def _submission_audit(root: Path) -> dict[str, Any]:
    paper = root / "paper/iclr2027"
    return audit_submission(
        pdf=paper / "build/main.pdf",
        source=paper / "main.tex",
        supplement=paper / "pivot_iclr2027_supplementary.zip",
        style_dir=paper / "style",
        output=paper / "submission_verification.json",
        aux=paper / "build/main.aux",
        max_main_pages=9,
    )


def _final_markdown(
    *,
    status: str,
    source_validation: dict[str, Any],
    audits: dict[str, Any],
    paper_verification: dict[str, Any] | None,
    submission: dict[str, Any] | None,
) -> str:
    main_pages = paper_verification.get("main_pages") if paper_verification else "pending"
    pages = paper_verification.get("pages") if paper_verification else "pending"
    machine_checks = submission.get("machine_checks", {}) if submission else {}
    machine_pass = bool(submission) and all(bool(value) for value in machine_checks.values())
    return f"""# PIVOT V10 Final Report

## Scientific status

The paper supports Improvement Fidelity as an operator-relative estimand, the possibility and controlled occurrence of Improvement Reversal, a response--footprint mechanism, scoped evidence-efficiency gains over Proxy Only, and a strategic reversal layer across the tested adaptive opponent families. Six propositions delimit the theory. Scientific experiments were not rerun during the publication rebuild: `{source_validation.get("scientific_experiments_rerun")}`.

## Important nulls

The registered transition-specific OOD learner does not outperform the global evaluator. PIVOT-VOI does not uniformly dominate Global-VOI or Paired LUCB. The frozen external adaptive reference remains a null. The public finance audit is observational, uses virtual fills/depth proxies, and does not identify causal market impact.

## Main-text changes

The manuscript now centers the directed update `pi -> pi'`, states all six propositions, defines ISR/CTI/CISR and matched-cell FER, removes internal run/version language from the scientific body, separates PIVOT-VOI from PIVOT-H, and uses a light AI Use Statement. The current PDF has `{main_pages}` main pages and `{pages}` total pages; references and appendix are outside the main-text gate.

## Figure changes

- Figure 1: sparse phenomenon display -> full-row traceable reversal planes.
- Figure 2: operator-shift mechanism -> aligned local-error/global-rank/reversal panels.
- Figure 3: dense overlapping systems diagram -> compact two-band OpenTikZ PIVOT-VOI decision path with semantic provenance.
- Figure 4: heterogeneous frontier -> fixed-K environment panels plus paired fixed-budget forest.
- Figure 5: endpoint summary -> seed-level closed-loop CTI/CISR trajectories with the external null kept endpoint-only.
- Figure A (old Figure 6): sparse world points -> paired response paths, effect distributions, and a strategic reversal plane.
- Figure B (old Figure 7): bars -> powered-null forest and paired raw scatter.
- Figure C (old Figure 8): Jaccard-only curves -> numerical and decision robustness diagnostics.
- Figure D (old Figure 9): sensitivity line -> opponent distributions, cluster forest, and strategic reversal plane.
- Figure E: finance is retained as an observational boundary diagnostic.

## Metric audit

CISR is expressed in native reward units. In the closed loop it sums ISR over 40 rounds; in the evidence-efficiency study the horizon is one candidate set, so CISR equals ISR. Raw magnitudes are not compared across environments. FER is allowed only inside matched environment/K/horizon cells when the Proxy-to-All-HF denominator is stable.

## Bibliography changes

The novelty boundary now explicitly covers Self-Improvement Reversal, AI4AI-Bench, SEAL-style external verification, autonomous policy evolution, policy-aware simulation, recent harness reliability, endogenous market simulators, and strategic financial environments. Recent preprints remain labeled as preprints; top-conference references are identified only when the venue is verified. The erroneous Thomas--Brunskill link was corrected to its official PMLR page.

## Remaining risks

- Controlled worlds establish mechanism and falsifiability, not natural-world prevalence.
- The learned transition evaluator has a powered negative result.
- Acquisition comparisons are environment- and budget-dependent.
- Strategic evidence uses finite opponent mechanisms rather than general equilibrium.
- Finance does not supply causal actor/strategic validation.
- OpenReview profiles, author quota, conflicts, final author metadata, and parallel-submission confirmation remain author-side manual gates.

## Submission readiness

`{status}`

All local machine checks pass: `{machine_pass}`. The package is locally submission-ready within its stated scientific scope, but it has not been uploaded to OpenReview and the manual author gates above remain pending.
"""


def _write_final_report(
    root: Path,
    *,
    status: str,
    source_validation: dict[str, Any],
    audits: dict[str, Any],
    paper_verification: dict[str, Any] | None,
    submission: dict[str, Any] | None,
) -> Path:
    return write_markdown(
        root,
        "V10_FINAL_REPORT.md",
        _final_markdown(
            status=status,
            source_validation=source_validation,
            audits=audits,
            paper_verification=paper_verification,
            submission=submission,
        ),
    )


def finalize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1787227200")
    steps: list[dict[str, Any]] = []
    source_validation = _validate_frozen_sources(root)
    assets = _build_publication_assets(root)
    audits = _run_audits(root)
    _write_final_report(
        root,
        status="NOT_READY",
        source_validation=source_validation,
        audits=audits,
        paper_verification=None,
        submission=None,
    )
    _run(["bash", "paper/iclr2027/build.sh"], root, steps)

    # The build regenerates figures/tables, so audit their final bytes again.
    audits = _run_audits(root)
    paper_verification = load_json(root / "paper/iclr2027/verification.json")
    if not paper_verification.get("valid"):
        raise RuntimeError("paper verification report is not valid")
    submission = _submission_audit(root)
    machine_pass = all(bool(value) for value in submission.get("machine_checks", {}).values())
    status = "READY_WITH_MINOR_MANUAL_CHECKS" if machine_pass else "NOT_READY"
    _write_final_report(
        root,
        status=status,
        source_validation=source_validation,
        audits=audits,
        paper_verification=paper_verification,
        submission=submission,
    )
    finalize_report: dict[str, Any] = {
        "valid": status != "NOT_READY",
        "status": status,
        "scientific_experiments_rerun": False,
        "source_validation": _publicize(source_validation, root),
        "asset_summary": _publicize(assets, root),
        "audit_status": {name: bool(report.get("valid")) for name, report in audits.items()},
        "paper": {
            "valid": paper_verification.get("valid"),
            "main_pages": paper_verification.get("main_pages"),
            "pages": paper_verification.get("pages"),
            "sha256": sha256(root / "paper/iclr2027/pivot_iclr2027_submission.pdf"),
        },
        "submission_machine_checks": submission.get("machine_checks"),
        "manual_gates": submission.get("manual_gates"),
        "scientific_boundaries": _publicize(submission.get("scientific_gates"), root),
        "build_steps": _publicize(steps, root),
    }
    write_json(root, "artifacts/v10/finalize_report.json", finalize_report)

    # Rebuild the supplement once more so it contains the final audits/report,
    # then rerun the archive and submission checks.  The finalizer JSON omits
    # the archive hash to avoid a self-referential artifact.
    paper = root / "paper/iclr2027"
    build_supplement(root, paper / "supplementary")
    write_deterministic_zip(paper / "supplementary", paper / "pivot_iclr2027_supplementary.zip")
    submission = _submission_audit(root)
    if not all(bool(value) for value in submission.get("machine_checks", {}).values()):
        raise RuntimeError("final supplementary/submission machine audit failed")
    print(
        json.dumps(
            {
                "status": status,
                "main_pages": paper_verification.get("main_pages"),
                "pdf_sha256": finalize_report["paper"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return finalize_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the PIVOT V10 ICLR package")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = finalize(args.root)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
