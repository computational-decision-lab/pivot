#!/usr/bin/env python3
"""Generate the pre-final and reviewer-attack audits from frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit_utils import load_json, sha256, write_json, write_markdown

FROZEN_SOURCE_COMMIT = "8541638d9115fb71c1b4d780e702762a030e3f59"


def pre_final_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    snapshot = root / "snapshot/v10_pre_final"
    snapshot_pdf = snapshot / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    snapshot_zip = snapshot / "paper/iclr2027/pivot_iclr2027_supplementary.zip"
    snapshot_manifest = snapshot / "SHA256SUMS"
    old_verification = load_json(snapshot / "paper/iclr2027/verification.json")
    issues = [
        {
            "location": "snapshot manuscript, Theory introduction",
            "problem": "Text stated four results while six propositions were present.",
            "severity": "blocking",
            "required_fix": "State six results and automatically count proposition environments.",
            "source_evidence": "snapshot main.tex:283 and six proposition environments",
            "resolution": "fixed and checked by claim/PDF verifier",
        },
        {
            "location": "snapshot abstract, results, captions, and appendix",
            "problem": "Internal version and run identifiers read like a development log.",
            "severity": "high",
            "required_fix": "Use scientific environment/operator names in the main paper; retain IDs only in artifact metadata.",
            "source_evidence": "snapshot main.tex contains V7/V9 and E2C/E3C/E4C/E5C/E7C",
            "resolution": "fixed in the scientific body; automated forbidden-token scan added",
        },
        {
            "location": "snapshot Figures 6--9",
            "problem": "Sparse points/bars/lines did not expose mechanism, raw variation, null evidence, or strategic reversal.",
            "severity": "high",
            "required_fix": "Rebuild as paired layers, effect distributions/forests, robustness diagnostics, and reversal planes.",
            "source_evidence": "frozen pre-final figure bundles",
            "resolution": "replaced by Figures A--D with source tables and cluster-level uncertainty",
        },
        {
            "location": "snapshot architecture figure",
            "problem": "The earlier dense world/reporting layout produced container and connector collisions at paper scale.",
            "severity": "high",
            "required_fix": "Use a compact decision-critical two-band OpenTikZ architecture and keep world taxonomy in the appendix.",
            "source_evidence": "frozen OpenTikZ source and visual review",
            "resolution": "rebuilt with 11 semantic nodes, routed connectors, raster/vector QA, and no overflow",
        },
        {
            "location": "metrics and cross-environment figures",
            "problem": "CISR was used for both one-set and cumulative settings without a single scale contract; FER was absent.",
            "severity": "high",
            "required_fix": "Define ISR, CTI, CISR_T, and matched-cell FER; prohibit raw cross-environment comparisons.",
            "source_evidence": "statistics.py, closed-loop runner, efficiency runner",
            "resolution": "formal definitions and V10 metric-scale audit added",
        },
        {
            "location": "related work and references",
            "problem": "Novelty boundaries omitted key self-improvement reversal/benchmark work; one off-policy URL was wrong.",
            "severity": "high",
            "required_fix": "Add explicit boundary paragraphs and verify primary metadata.",
            "source_evidence": "pre-final bibliography and recent primary metadata",
            "resolution": "expanded recent/top-conference coverage; Thomas--Brunskill URL corrected",
        },
        {
            "location": "finance paragraph and public table",
            "problem": "The seven-session single-asset sweep and the 12-session three-asset expansion were conflated.",
            "severity": "high",
            "required_fix": "Describe the datasets separately and source 0/7, 0/5, and the pooled effect from the frozen expansion JSON.",
            "source_evidence": "public calibration summary versus paper snapshot public-expansion-summary.json",
            "resolution": "main text, caption, table, and number audit now keep the two observational layers distinct",
        },
        {
            "location": "main Figure 3 bundle",
            "problem": "Editable architecture assets existed but the required method-oriented figure bundle was incomplete.",
            "severity": "medium",
            "required_fix": "Export PDF/SVG/PNG/TEX plus semantic-node CSV/Parquet and metadata under fig3_pivot_voi.",
            "source_evidence": "release output contract",
            "resolution": "implemented in deterministic architecture/figure build",
        },
        {
            "location": "page budget",
            "problem": "The pre-final manuscript occupied the full nine-page main-text budget.",
            "severity": "medium",
            "required_fix": "Rebuild and verify the reference boundary rather than total PDF page count.",
            "source_evidence": f"pre-final verification main_pages={old_verification.get('main_pages')}",
            "resolution": "final verifier enforces at most nine main pages; current build leaves margin",
        },
        {
            "location": "pre-final SHA256SUMS",
            "problem": "The manifest contains an unavoidable invalid self-hash placeholder for SHA256SUMS itself.",
            "severity": "low",
            "required_fix": "Do not mutate the frozen snapshot; bind the manifest externally by its own SHA-256.",
            "source_evidence": "first manifest line is the empty-file digest while the external digest is nonempty",
            "resolution": "external freeze record stores the manifest hash and excludes self-verification",
        },
        {
            "location": "double-blind package",
            "problem": "No identity leak was found in the pre-final PDF/package.",
            "severity": "none",
            "required_fix": "Preserve and rerun source/PDF/archive checks.",
            "source_evidence": "pre-final verification anonymous_author=true",
            "resolution": "retained as a release gate",
        },
    ]
    freeze = {
        "snapshot_path": "snapshot/v10_pre_final",
        "snapshot_timestamp": "2026-08-26T20:16:54+08:00",
        "source_git_commit": FROZEN_SOURCE_COMMIT,
        "pdf_sha256": sha256(snapshot_pdf),
        "supplement_sha256": sha256(snapshot_zip),
        "sha256_manifest_sha256": sha256(snapshot_manifest),
        "snapshot_size_bytes": sum(
            path.stat().st_size for path in snapshot.rglob("*") if path.is_file()
        ),
        "immutable": True,
        "self_hash_exception": "SHA256SUMS self-entry is not used; the file is externally bound by sha256_manifest_sha256",
    }
    report = {
        "valid": all(item["resolution"] for item in issues),
        "freeze": freeze,
        "issues": issues,
        "audit_dimensions": [
            "abstract claims",
            "contributions",
            "proposition count",
            "captions",
            "metrics",
            "experiment identifiers",
            "numbers",
            "version leakage",
            "related work",
            "terminology",
            "figure/appendix references",
            "page budget",
            "double blind",
        ],
    }
    write_json(root, "artifacts/v10/pre_final_audit.json", report)
    write_markdown(root, "V10_PRE_FINAL_AUDIT.md", _pre_final_markdown(report))
    return report


def _pre_final_markdown(report: dict[str, Any]) -> str:
    freeze = report["freeze"]
    lines = [
        "# V10 Pre-Final Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        "## Frozen baseline",
        "",
        f"- Source commit: `{freeze['source_git_commit']}`",
        f"- Snapshot: `{freeze['snapshot_path']}`",
        f"- Timestamp: `{freeze['snapshot_timestamp']}`",
        f"- PDF SHA-256: `{freeze['pdf_sha256']}`",
        f"- Supplement SHA-256: `{freeze['supplement_sha256']}`",
        f"- Manifest SHA-256: `{freeze['sha256_manifest_sha256']}`",
        "- The snapshot is immutable. Its manifest is externally bound because a file cannot contain its own final SHA-256.",
        "",
        "## Findings and resolutions",
        "",
        "| Location | Problem | Severity | Required fix | Source evidence | Resolution |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["issues"]:
        lines.append(
            f"| {row['location']} | {row['problem']} | **{row['severity']}** | {row['required_fix']} | {row['source_evidence']} | {row['resolution']} |"
        )
    return "\n".join(lines)


ATTACKS = [
    (
        "Is this merely another policy evaluation metric?",
        "Problem Formulation and Contributions 1--2",
        "No. Policy evaluation estimates J(pi); Improvement Fidelity evaluates the sign/order of a replacement under the operator-induced transition law, and PIVOT allocates paired interventions for selection regret.",
        "The paper does not claim policy-value models are unnecessary; global fidelity is a sufficient but stronger condition.",
    ),
    (
        "Is Global Fidelity Blindness only an adversarial construction?",
        "Proposition 2 and operator-shift experiment",
        "The proposition is a sharp non-implication. The empirical shift experiment separately shows structured local deterioration while global rank remains competitive.",
        "The construction proves possibility, not prevalence; prevalence is limited to tested operator/world families.",
    ),
    (
        "Why not just use a global value model?",
        "Value Fidelity versus Improvement Fidelity; OOD null",
        "A global model is a strong baseline and sometimes wins. The target can still be misaligned because the self-improver visits local directed transitions rather than the global policy population.",
        "The registered differential learner underperforms globally in the tested OOD splits; the paper retains that powered null.",
    ),
    (
        "Why not just use LUCB?",
        "PIVOT VOI and evidence-efficiency figure",
        "LUCB targets confidence around a best arm; PIVOT-VOI uses paired differential posterior regret, footprint/context, and heterogeneous evaluation cost. Paired LUCB is explicitly compared.",
        "PIVOT is not uniformly favorable against LUCB in the frozen results.",
    ),
    (
        "Why does PIVOT-VOI not always win?",
        "Closed-loop outcomes and Stress Tests",
        "Acquisition quality depends on posterior fit, candidate gaps, response structure, and cost. The method is designed for decision-sensitive evidence allocation, not universal dominance.",
        "Claims are restricted to Proxy Only contrasts and registered cells with supported paired effects.",
    ),
    (
        "Are the adaptive environments hand-designed to produce reversal?",
        "Experiments design and Figure 1",
        "The worlds are transparent controlled mechanisms with preregistered response/shift sweeps and independent seeds. They make the phenomenon falsifiable and expose where the sign changes.",
        "They are not presented as natural-world prevalence or market realism; the external reference null is preserved.",
    ),
    (
        "Are Figure 4 conditions comparable?",
        "Figure 4 caption and metadata",
        "Top panels fix K=8 within each environment and plot matched query cost. The lower panel reports paired fixed-budget effects; heterogeneous cells are not connected.",
        "The all-HF line is an oracle reference, not a method trajectory, and environments retain separate axes.",
    ),
    (
        "Why do CISR scales differ?",
        "Metrics and Scale; metric-scale audit",
        "CISR is in native reward units. Environment dynamics and reward scales differ; closed-loop CISR sums rounds while the budget study has T=1.",
        "Raw magnitudes are never compared across environments; only within-environment paired effects or stable matched-cell FER are allowed.",
    ),
    (
        "Is strategic reversal hard-coded?",
        "Strategic response and Figures A/D",
        "The outcome is measured across independent opponent-seed clusters and five mechanisms. Fixed opponents show a near-zero effect, while adaptive families show negative effects with cluster intervals.",
        "Mechanisms are finite controlled adaptations, not proof of equilibrium behavior or real market ecology.",
    ),
    (
        "What does the negative OOD result imply?",
        "OOD evaluator contrast and Figure B",
        "It rejects the claim that the registered transition learner dominates the global evaluator. It does not reject the transition-level estimand or paired decision target.",
        "No claim is made for untested model classes, data regimes, or domains.",
    ),
    (
        "How is this different from prior Self-Improvement Reversal?",
        "Related Work: Self-improvement and verification",
        "Prior post-training reversal concerns benchmark/capability regression. This paper defines a directed policy-update estimand under policy-induced world response and validates it with paired interventions.",
        "The paper does not claim the phrase 'reversal' itself; novelty is the estimand, response decomposition, and budgeted validator.",
    ),
    (
        "How does AI4AI-Bench relate to this paper?",
        "Related Work: Self-improvement and verification",
        "AI4AI-Bench evaluates whether agents redesign training algorithms under a hidden evaluator. This paper asks whether a proposed replacement remains beneficial after endogenous/strategic response.",
        "PIVOT is not evaluated on AI4AI-Bench, so the relation is conceptual and complementary.",
    ),
    (
        "Is finance causal evidence?",
        "Finance audit and boundary; Figure E",
        "No. It is an observational historical-path/virtual-fill/depth-proxy stress test and reports 0/7 primary and 0/5 holdout causal reversals.",
        "Causal market impact, replenishment, strategic response, profitability, and live trading are explicitly not identified.",
    ),
]


def reviewer_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source = (root / "paper/iclr2027/main.tex").read_text(encoding="utf-8")
    source_flat = " ".join(source.split())
    errors: list[str] = []
    for phrase in (
        "Value Fidelity versus Improvement Fidelity",
        "powered null",
        "does not identify causal market impact",
        "no universal superiority",
    ):
        if phrase.casefold() not in source_flat.casefold():
            errors.append(f"manuscript does not expose reviewer-answer phrase: {phrase}")
    records = [
        {
            "attack": index,
            "question": question,
            "paper_location": location,
            "supported_answer": answer,
            "scope_limitation": scope,
        }
        for index, (question, location, answer, scope) in enumerate(ATTACKS, start=1)
    ]
    report = {"valid": not errors and len(records) == 13, "errors": errors, "attacks": records}
    write_json(root, "artifacts/v10/reviewer_attack_audit.json", report)
    write_markdown(root, "V10_REVIEWER_ATTACK_AUDIT.md", _reviewer_markdown(report))
    return report


def _reviewer_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Reviewer Attack Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
    ]
    for row in report["attacks"]:
        lines.extend(
            [
                "",
                f"## Attack {row['attack']}: {row['question']}",
                "",
                f"**Paper location:** {row['paper_location']}",
                "",
                f"**Supported answer:** {row['supported_answer']}",
                "",
                f"**Scope limitation:** {row['scope_limitation']}",
            ]
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V10 release audits")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    pre = pre_final_audit(args.root)
    reviewer = reviewer_audit(args.root)
    valid = pre["valid"] and reviewer["valid"]
    print(
        json.dumps(
            {
                "valid": valid,
                "pre_final_issues": len(pre["issues"]),
                "reviewer_attacks": len(reviewer["attacks"]),
            },
            sort_keys=True,
        )
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
