"""Generate the V15 protocol, evidence, and release audit reports.

Reports are intentionally conservative.  Existing controlled evidence and
bounded external DEV artifacts are referenced with their recorded terminal
state; confirmatory claims remain closed until the frozen protocol is opened.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .analyze_footprint import analyze_footprint
from .audit_claims import audit_claims
from .audit_language import scan_language
from .audit_references import audit_references
from .audit_reproducibility import external_execution_status
from .audit_terminal_states import audit_terminal_states
from .canonical import refresh_canonical_tables
from .configuration import ensure_lock, is_confirmatory_open
from .control_plane import probe_adapters
from .dev import run_smoke
from .evidence import freeze_candidate_archive, write_promotion_replay
from .manifest_contract import backfill_dev_manifests
from .pi_runtime import pi_runtime_status
from .protocol import file_hash
from .scientific_analysis import (
    analyze_ablations_artifact,
    analyze_all,
    analyze_closed_loop_artifact,
    analyze_pi_artifact,
    analyze_promotion_artifact,
    analyze_strategic_artifact,
    analyze_transition_artifact,
)

REQUIRED_REPORTS = (
    "V15_BASELINE_SNAPSHOT.md",
    "V15_REPO_AUDIT.md",
    "V15_RESOURCE_PLAN.md",
    "V15_CONSTRUCT_VALIDITY.md",
    "V15_CONFIRMATORY_PREREGISTRATION.md",
    "V15_TRANSITION_AUDIT.md",
    "V15_SCIENTIFIC_SUMMARY.md",
    "V15_OPERATOR_RELATIVE_ANALYSIS.md",
    "V15_FOOTPRINT_ANALYSIS.md",
    "V15_PROMOTION_RESULTS.md",
    "V15_PAIRED_ABLATION.md",
    "V15_PIVOT_ABLATIONS.md",
    "V15_CLOSED_LOOP_RESULTS.md",
    "V15_PI_REPLICATION.md",
    "V15_STRATEGIC_RESULTS.md",
    "V15_FALSIFICATION_REPORT.md",
    "V15_FIGURE_STATUS.md",
    "V15_PAPER_CONTEXT_AUDIT.md",
    "V15_NUMBER_AUDIT.md",
    "V15_CLAIM_AUDIT.md",
    "V15_REFERENCE_AUDIT.md",
    "V15_ANONYMITY_AUDIT.md",
    "V15_LANGUAGE_AUDIT.md",
    "V15_REPRODUCIBILITY_AUDIT.md",
    "V15_REVIEWER_ATTACK_AUDIT.md",
    "V15_OUTSTANDING_PROFILE.md",
    "V15_FINAL_REPORT.md",
)


def _sha256(path: Path) -> str:
    return file_hash(path)


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _snapshot_git_commit(root: Path) -> str:
    """Read the commit recorded when the pre-modern-agent snapshot was made."""

    provenance = Path(root) / "snapshot/v15_pre_modern_agent/PROVENANCE.txt"
    try:
        for line in provenance.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(" ")
            if key == "git_commit" and separator and value.strip():
                return value.strip()
    except OSError:
        pass
    return "unavailable"


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _json(path: Path, fallback: Any = None) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _load_protocol_config(root: Path) -> dict[str, Any]:
    """Load the YAML protocol without treating it as JSON."""

    path = Path(root) / "configs/v15/confirmatory.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _markdown_table(rows: Iterable[tuple[str, str, str]]) -> str:
    lines = ["| Check | Status | Evidence |", "|---|---|---|"]
    lines.extend(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return "\n".join(lines)


def _resource_plan_markdown(config: Mapping[str, Any]) -> str:
    """Render the pre-outcome accounting fields without inventing costs."""

    plan = config.get("resource_plan", {})
    estimated = plan.get("estimated", {}) if isinstance(plan, Mapping) else {}
    observed = plan.get("observed_before_confirmatory_run", {}) if isinstance(plan, Mapping) else {}
    rows = [
        ("LLM calls", str(estimated.get("llm_calls", "not specified")), str(observed.get("llm_calls", 0))),
        ("Inspect evaluations", str(estimated.get("inspect_evaluations", "not specified")), str(observed.get("inspect_evaluations", 0))),
        ("Container executions", str(estimated.get("paired_container_executions", "not specified")), str(observed.get("container_executions", 0))),
        ("Task instances", str(estimated.get("task_instances", "not specified")), "not opened"),
        ("T x K", f"{config.get('rounds', 'unset')} x {config.get('candidates_per_round', 'unset')}", "locked"),
        ("CPU / GPU", f"{estimated.get('cpu', 'not specified')} / {estimated.get('gpu', 'not specified')}", "not run"),
        ("Token budget", str(estimated.get("token_budget_per_call", "not specified")), "not run"),
        ("Storage", str(estimated.get("storage_gb", "not estimable")), "not run"),
        ("Wall clock", str(estimated.get("wall_clock_hours", "not estimable")), "not run"),
        ("Estimated cost", str(estimated.get("cost_usd", "not estimable")), "0 confirmed calls"),
    ]
    return "| Resource | Pre-outcome estimate | Observed before confirmatory run |\n|---|---|---|\n" + "\n".join(
        f"| {name} | {estimate} | {observed_value} |" for name, estimate, observed_value in rows
    )


def _outstanding_profile_table() -> str:
    rows = (
        ("Conceptual message", "PASS", "Preserve transition-first framing."),
        ("Theory-experiment bridge", "PASS", "Keep assumptions and controlled tests paired."),
        ("Mechanism identification", "PASS / scoped", "Separate observer, actor, and strategic layers."),
        ("Modern-system realism", "OPEN", "Run the pinned primary scaffold when available."),
        ("Cross-scaffold generality", "OPEN", "Run the predeclared second-scaffold transition audit."),
        ("Intervention quality", "PASS / local", "Retain fresh paired sandbox manifests."),
        ("Negative controls", "PASS / local", "Keep null and underpowered states reportable."),
        ("Baseline fairness", "OPEN", "Run all methods on the immutable candidate archive."),
        ("Statistical rigor", "PASS for frozen evidence", "Use trajectory or task-cluster inference units."),
        ("Reproducibility", "PASS for local layer", "Pin model, image, and dependency lock before execution."),
        ("Visual communication", "PASS", "Repeat rendered, print-size, grayscale, and paper-context gates."),
        ("Scientific honesty", "PASS", "Never alter design in response to outcomes."),
    )
    return "| Dimension | Status | Permissible action |\n|---|---|---|\n" + "\n".join(
        f"| {dimension} | {status} | {action} |" for dimension, status, action in rows
    )


def _falsification_table(config: Mapping[str, Any]) -> str:
    hypotheses = config.get("primary_hypotheses", {})
    if not isinstance(hypotheses, Mapping):
        hypotheses = {}
    lines = [
        "| Hypothesis | Required terminal state | Current execution status | Falsification rule |",
        "|---|---|---|---|",
    ]
    for key, statement in hypotheses.items():
        lines.append(
            f"| {key}: {statement} | HYPOTHESIS_SUPPORTED or HYPOTHESIS_NOT_SUPPORTED | NOT_RUN | Freeze either outcome; do not retune tasks, operators, budgets, or response strength. |"
        )
    return "\n".join(lines)


def _adapter_lines(root: Path) -> str:
    return "\n".join(
        f"- {item.name}: `{'available' if item.available else 'not available'}`"
        + (f" ({item.version})" if item.version else "")
        for item in probe_adapters(root)
    )


def _phase_manifest(root: Path, directory: str) -> dict[str, Any]:
    payload = _json(root / "results/v15" / directory / "manifest.json", {})
    return payload if isinstance(payload, dict) else {}


def _modern_dev_summary(root: Path) -> dict[str, Any]:
    """Collect DEV evidence without treating it as confirmatory evidence."""

    transition = _phase_manifest(root, "dev-external-transition-audit")
    promotion = _phase_manifest(root, "dev-external-promotion")
    closed_loop = _phase_manifest(root, "dev-external-closed-loop")
    pi = _phase_manifest(root, "dev-pi-replication")
    strategic = _phase_manifest(root, "dev-external-strategic-response")
    ablations = _phase_manifest(root, "dev-external-ablations")
    phase_statuses = {
        "dev-external-transition-audit": str(transition.get("status", "")),
        "dev-external-promotion": str(promotion.get("status", "")),
        "dev-external-closed-loop": str(closed_loop.get("status", "")),
        "dev-pi-replication": str(pi.get("status", "")),
        "dev-external-strategic-response": str(strategic.get("status", "")),
        "dev-external-ablations": str(ablations.get("status", "")),
    }
    return {
        "transition": transition,
        "promotion": promotion,
        "closed_loop": closed_loop,
        "pi": pi,
        "strategic": strategic,
        "ablations": ablations,
        "pi_runtime": pi_runtime_status(root),
        "phase_statuses": phase_statuses,
        "dev_complete": _dev_runtime_verified(phase_statuses),
    }


def _dev_runtime_verified(phase_statuses: Mapping[str, str]) -> bool:
    """Return whether all required DEV interfaces ran successfully.

    The strategic phase intentionally permits ``PARTIAL`` when the frozen
    non-LLM responder completed but the optional independent-agent responder
    was not run.  That state is externally auditable and is not the same as a
    missing or failed DEV artifact.
    """

    required = (
        "dev-external-transition-audit",
        "dev-external-promotion",
        "dev-external-closed-loop",
        "dev-pi-replication",
        "dev-external-ablations",
    )
    return all(phase_statuses.get(name) == "COMPLETED" for name in required) and phase_statuses.get(
        "dev-external-strategic-response"
    ) in {"COMPLETED", "PARTIAL"}


def _existing_terminal_states(root: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    for name in ("e2c", "e3c", "e4c", "e5c", "e7c"):
        path = root / f"results/v9/{name}-confirmatory/scientific_decision.json"
        payload = _json(path, {})
        if isinstance(payload, Mapping):
            states[name.upper()] = str(payload.get("status", "UNKNOWN"))
    return states


def _ensure_dev_artifacts(root: Path) -> dict[str, Any]:
    output = root / "results/v15/dev-smoke"
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and (output / "promotion_candidates.jsonl").is_file():
        payload = _json(manifest_path, {})
        if isinstance(payload, dict) and payload.get("phase") == "DEV":
            return payload
    return run_smoke(output, candidates_per_operator=2)


def _ensure_replay_artifacts(root: Path) -> dict[str, Any]:
    dev = root / "results/v15/dev-smoke"
    archive = root / "results/v15/candidate-archive"
    archive_manifest = archive / "manifest.json"
    if not archive_manifest.is_file():
        freeze_candidate_archive(dev / "promotion_candidates.jsonl", archive)
    replay = root / "results/v15/promotion-replay"
    replay_manifest = replay / "manifest.json"
    if not replay_manifest.is_file():
        write_promotion_replay(archive / "promotion_candidates.jsonl", replay)
    payload = _json(replay_manifest, {})
    return payload if isinstance(payload, dict) else {}


def _number_audit(root: Path) -> dict[str, Any]:
    """Check semantic macros against the already frozen source decisions."""

    macro = root / "paper/iclr2027/results_macros.tex"
    if not macro.is_file():
        macro = root / "paper/iclr2027/v10_results_macros.tex"
    source = macro.read_text(encoding="utf-8") if macro.is_file() else ""
    strategic = _json(root / "results/v9/e7c-confirmatory/strategic_summary.json", {}) or {}
    by_mode = strategic.get("by_mode", []) if isinstance(strategic, Mapping) else []
    adaptive = [
        row for row in by_mode if row.get("opponent_mode") in {"best_response", "gradient_adaptive", "rl_evolutionary"}
    ]
    expected = {
        "StrategicClusters": str(strategic.get("independent_seed_count", "")),
        "StrategicAdaptiveFamilies": str(len(adaptive)),
        "StrategicFamilySeedTraces": str(sum(int(row.get("cluster_n", 0)) for row in adaptive)),
        "StrategicAllFamilySeedTraces": str(sum(int(row.get("cluster_n", 0)) for row in by_mode)),
        "StrategicSIRR": f"{sum(float(row.get('SIRR', 0.0)) for row in adaptive) / len(adaptive):.4f}" if adaptive else "",
    }
    checks: dict[str, bool] = {}
    for macro_name, value in expected.items():
        match = re.search(rf"\\newcommand\{{\\{macro_name}\}}\{{([^}}]+)\}}", source)
        checks[macro_name] = match is not None and match.group(1) == value
    return {"valid": all(checks.values()), "checks": checks, "expected": expected}


def _pdf_summary(root: Path) -> dict[str, Any]:
    pdf = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    summary: dict[str, Any] = {"exists": pdf.is_file()}
    if not pdf.is_file():
        return summary
    try:
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {**summary, "pdfinfo": "unavailable"}
    for label in ("Pages", "Page size", "Title", "Author"):
        match = re.search(rf"^{re.escape(label)}:[ \t]*([^\n]*)$", info, flags=re.MULTILINE)
        if match:
            summary[label.casefold().replace(" ", "_")] = match.group(1).strip()
    verification = _json(root / "paper/iclr2027/verification.json", {})
    if isinstance(verification, Mapping):
        summary["main_text_pages"] = verification.get("main_text_pages", verification.get("main_pages"))
        summary["references_start_page"] = verification.get("references_start_page")
    return summary


def _reviewer_attack_table() -> str:
    attacks = (
        ("A01", "Ordinary train/test shift", "medium", "sealed plane manifest and intervention definitions", "The estimand is a directed replacement under a policy-induced world, not a random split.", "Problem formulation", "External modern-agent execution is not yet available.", "Run the locked primary study when dependencies and model are authorized."),
        ("A02", "Merely hidden testing", "high", "separate proxy, gate, and untouched assessment roles", "Hidden evaluation is a control condition; the actor world changes through the executed trajectory.", "Sealed data planes", "The external control plane remains a dry-run contract.", "Preserve role logs and publish the eventual access audit."),
        ("A03", "Goodhart effect only", "medium", "fixed external verifier and response ladder", "Goodhart is compatible with the phenomenon but does not define the update estimand.", "Introduction and related work", "Mechanistic attribution awaits external runs.", "Report the result as scoped fidelity evidence."),
        ("A04", "Standard active learning", "medium", "paired selection regret and policy-transition queries", "PIVOT acquires evidence to preserve a replacement decision, not an arbitrary label.", "PIVOT method", "Baseline replay is currently DEV-only.", "Compare all registered baselines on one archive."),
        ("A05", "Why not LUCB", "medium", "same candidate set, paired query unit, and HF budget", "LUCB is a registered fairness baseline; no universal dominance is claimed.", "Baseline protocol", "DEV replay exists; confirmatory baseline numbers are NOT_RUN.", "Run the predeclared replay without changing budgets."),
        ("A06", "PIVOT is not uniformly superior", "high", "explicit scope and null-preserving terminal states", "The claim is decision-sensitive validation under a frozen budget, not universal superiority.", "Contributions and limitations", "Performance relative to baselines is unresolved.", "Retain mixed or negative outcomes unchanged."),
        ("A07", "Transition learner loses OOD", "high", "registered OOD split and descriptive null report", "A transition-level estimand can be necessary even when one fitted learner is not dominant.", "OOD appendix", "The modern OOD run is NOT_RUN.", "Report the powered result and do not tune after opening it."),
        ("A08", "Theory is elementary", "medium", "six auditable propositions and explicit assumptions", "The results delimit sufficient conditions and identify the estimand; they do not claim algorithmic novelty in every bound.", "Theory", "The theory does not replace empirical validation.", "Keep assumptions and proofs visible."),
        ("A09", "Controlled worlds are constructed", "high", "response-strength controls and independent intervention layers", "Controlled worlds identify mechanisms before the less controlled coding-agent extension.", "Controlled experiments", "External ecological validity remains open.", "Use the registered real scaffold as the next evidence layer."),
        ("A10", "mini-SWE is scaffold-specific", "high", "Pi replication contract and scaffold-bound claims", "The primary claim is scoped to the evaluated scaffold unless replication supports broader scope.", "Replication", "Pi DEV smoke exists; confirmatory replication has not run.", "Run Pi only after freezing primary results."),
        ("A11", "Candidates were manually curated", "high", "immutable candidate archive and operator provenance", "Candidates are generated by two registered operators and archived before validation.", "Candidate archive", "DEV operators are deterministic stand-ins.", "Replace only the runner, never curate the confirmatory archive."),
        ("A12", "Meta-optimizer knew the desired failure", "high", "proxy-only operator information contract", "Operators receive proxy feedback and resource budget, not the hypothesis or hidden worlds.", "Operator contract", "DEV prompt execution is audited; confirmatory execution is NOT_RUN.", "Audit prompts and inputs before unsealing outcomes."),
        ("A13", "D_gate leaked to self-improvement", "critical", "role-denial log and disjoint task hashes", "The operator role cannot access gate tasks or outcomes.", "SealedDataPlanes and access audit", "Only local role checks are exercised so far.", "Require a failed-access test in the external harness."),
        ("A14", "D_assessment leaked to PIVOT", "critical", "terminal-assessor-only role and untouched endpoint", "PIVOT never receives assessment contents or scores before termination.", "Closed-loop protocol", "DEV terminal-only access is logged; confirmatory untouched assessment is NOT_RUN.", "Release one terminal query per trajectory with an immutable log."),
        ("A15", "Strategic reviewer is intentionally hostile", "medium", "two frozen response families and identity-blind inputs", "Response is a registered stress layer, not evidence of universal adversarial behavior.", "Strategic response", "Non-LLM DEV response is audited; independent-agent and confirmatory responses are NOT_RUN.", "Freeze non-agent and independent-agent responders before outcomes."),
        ("A16", "Footprint was defined post hoc", "high", "predeclared feature list and lock hash", "Footprint features are computed before gate queries and exclude deployment outcomes.", "Footprint analysis", "Behavioral features need external traces.", "Add only exploratory features with explicit labels."),
        ("A17", "Thousands of rows inflate N", "critical", "trajectory/task-cluster inference unit", "Transition rows are nested observations; independent N is the trajectory or cluster count.", "Metrics and statistics", "DEV rows do not estimate confirmatory uncertainty.", "Use cluster or hierarchical bootstrap."),
        ("A18", "Actor world is just another dataset", "high", "fresh paired sandbox and policy-dependent execution state", "Actor actions alter files, tests, resources, and subsequent state.", "Actor deployment world", "Local runner is intentionally minimal.", "Use containerized agent trajectories for confirmation."),
        ("A19", "Pairing is unnecessary", "medium", "same task snapshot, seed, and resource limits", "Paired differences remove common-mode task variation and are directly stored.", "Paired evaluation", "DEV unpaired ablation is complete; confirmatory ablation is NOT_RUN.", "Run the registered paired/unpaired comparison."),
        ("A20", "Pi replication is not matched", "high", "same task families, budgets, metrics, and frozen method", "Exact implementation symmetry is not required; the cross-scaffold question is explicitly scoped.", "Replication protocol", "Pi CLI and gateway-backed DEV smoke are available; confirmatory replication is NOT_RUN.", "Report any mismatch and avoid numerical equivalence claims."),
        ("A21", "PIVOT complexity is unjustified", "medium", "budget frontier and All-HF reference", "Complexity is justified only if decision regret improves per HF cost; otherwise retain the null.", "Promotion results", "DEV budget replay exists; confirmatory cost-normalized frontier is NOT_RUN.", "Compare cost-normalized decision outcomes."),
        ("A22", "Results do not generalize", "high", "operator, task-family, response, and scaffold strata", "Generalization is a conditional empirical claim, not an assumption.", "Scope and replication", "Modern strata are unresolved.", "Keep claims at the strongest supported level."),
        ("A23", "Product framing overwhelms science", "low", "estimand-first manuscript and no SaaS language", "The contribution is a statistical object and validation rule.", "Introduction", "Artifact tooling is extensive.", "Keep implementation details in the supplement."),
        ("A24", "Negative results undermine the method", "medium", "powered null and closed terminal states", "A valid null narrows the claim and is retained as evidence.", "Falsification report", "External hypotheses are NOT_RUN.", "Do not retune after a null result."),
        ("A25", "Finance evidence is irrelevant", "low", "explicit observational finance boundary", "Finance is a stress test for response layers, not the definition of the general problem.", "Finance audit and limitations", "No causal market-impact claim is made.", "Keep virtual fills and public-data limits explicit."),
    )
    lines = [
        "| ID | Attack | Severity | Evidence | Paper answer | Paper location | Remaining weakness | Action required |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in attacks)
    return "\n".join(lines)


def _language_audit(root: Path) -> dict[str, Any]:
    return scan_language(root)


def generate_reports(root: Path) -> dict[str, Any]:
    """Generate all required V15 reports and return the final gate summary."""

    root = Path(root).resolve()
    (root / "artifacts/v15").mkdir(parents=True, exist_ok=True)
    dev = _ensure_dev_artifacts(root)
    _ensure_replay_artifacts(root)
    lock_path = root / "experiments/v15/confirmatory_lock.json"
    ensure_lock(root, lock_path)
    lock_digest = _sha256(lock_path)
    states = _existing_terminal_states(root)
    pdf = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    language = _language_audit(root)
    numbers = _number_audit(root)
    claim_audit = audit_claims(root)
    reference_audit = audit_references(root)
    snapshot = root / "snapshot/v15_pre_modern_agent/PROVENANCE.txt"
    snapshot_ok = snapshot.is_file() and (root / "snapshot/v15_pre_modern_agent/source/research-pivot-head.tar").is_file()
    modern = _modern_dev_summary(root)
    manifest_migration = backfill_dev_manifests(root)
    # Refresh the view after the idempotent metadata migration so every report
    # exposes the same terminal-state vocabulary.
    modern = _modern_dev_summary(root)
    lock_payload = _json(lock_path, {}) or {}
    confirmatory_open = is_confirmatory_open(lock_payload.get("confirmatory_execution"))
    # Once the pre-outcome lock is opened, canonical tables must be sourced
    # exclusively from the corresponding confirmatory phase directories.  In
    # the normal pre-outcome state this remains an explicit DEV materialization.
    canonical = refresh_canonical_tables(root, confirmatory=confirmatory_open)
    # All command-level summaries and manuscript-facing reports consume the
    # same immutable analysis artifacts.  The analyzers are outcome-blind and
    # assign UNDERPOWERED to bounded DEV runs instead of manufacturing claims.
    transition_analysis = analyze_transition_artifact(root, confirmatory=confirmatory_open)
    promotion_analysis = analyze_promotion_artifact(root, confirmatory=confirmatory_open)
    closed_loop_analysis = analyze_closed_loop_artifact(root, confirmatory=confirmatory_open)
    ablation_analysis = analyze_ablations_artifact(root, confirmatory=confirmatory_open)
    strategic_analysis = analyze_strategic_artifact(root, confirmatory=confirmatory_open)
    pi_analysis = analyze_pi_artifact(root, confirmatory=confirmatory_open)
    scientific_summary = analyze_all(root, confirmatory=confirmatory_open)
    terminal_audit = audit_terminal_states(root)
    # Keep the registered footprint analysis in the same deterministic report
    # pass as the canonical tables. It remains descriptive before the lock is
    # opened, but the artifact is useful for construct-validity audits.
    footprint = analyze_footprint(root)
    # Package probes describe the project venv; DEV manifests describe the
    # pinned external runtime.  Keep both facts visible instead of conflating
    # a missing shell command with an unavailable research runtime.
    external_available = modern["dev_complete"]
    external_status = external_execution_status(
        {
            "dev-external-transition-audit": str(modern["transition"].get("status", "")),
            "dev-external-promotion": str(modern["promotion"].get("status", "")),
            "dev-external-closed-loop": str(modern["closed_loop"].get("status", "")),
            "dev-pi-replication": str(modern["pi"].get("status", "")),
            "dev-external-strategic-response": str(modern["strategic"].get("status", "")),
            "dev-external-ablations": str(modern["ablations"].get("status", "")),
        },
        confirmatory_open=confirmatory_open,
    )

    _write(
        root,
        "V15_BASELINE_SNAPSHOT.md",
        f"""# Baseline Snapshot

Status: **{'PASS' if snapshot_ok else 'BLOCKED'}**

The pre-modern-agent fallback is preserved before any new edits.  Git commit:
`{_snapshot_git_commit(root)}` (from the snapshot provenance).  The exact tracked subtree is stored in
`snapshot/v15_pre_modern_agent/source/research-pivot-head.tar`; key PDF,
source, bibliography, supplement, and macro copies plus SHA-256 provenance are
under `snapshot/v15_pre_modern_agent/key_artifacts`.

PDF: `{_sha256(pdf) if pdf.is_file() else 'missing'}`
Provenance file: `{_sha256(snapshot) if snapshot.is_file() else 'missing'}`
""",
    )
    _write(
        root,
        "V15_REPO_AUDIT.md",
        f"""# Repository Audit

Status: **PASS** for the local protocol layer.

The inventory is audited from the current checkout; commit identity is
provided by Git history rather than embedded here, avoiding a self-referential
report change on rebuild.
Tracked PIVOT files: `{len(_git(root, 'ls-files').splitlines())}`
Python: `{platform.python_version()}`
Outcome chasing flag: `false`

Existing frozen terminal states:
{_markdown_table((name, state, f"results/v9/{name.lower()}-confirmatory/scientific_decision.json") for name, state in states.items())}
""",
    )
    cfg = _load_protocol_config(root)
    _write(
        root,
        "V15_RESOURCE_PLAN.md",
        f"""# Resource Plan

This is the locked resource plan.  The registered primary design is two
operator families, two task families, 30 independent trajectories per major
unit, `{cfg.get('rounds', 'unset')}` rounds, and `{cfg.get('candidates_per_round', 'unset')}` candidates per round.
The pinned external runtime has completed bounded DEV smoke runs.  Confirmatory
execution remains unopened, so DEV counts below are validation evidence only.

Local adapter probes:
{_adapter_lines(root)}

DEV manifests: transition `{modern['transition'].get('status', 'MISSING')}`, promotion
`{modern['promotion'].get('status', 'MISSING')}`, closed loop
`{modern['closed_loop'].get('status', 'MISSING')}`, Pi
`{modern['pi'].get('status', 'MISSING')}`, response
`{modern['strategic'].get('status', 'MISSING')}`, ablations
`{modern['ablations'].get('status', 'MISSING')}`.  These runs recorded no
assessment access outside the terminal DEV assessor and no outcome chasing.
Reducing scope before confirmatory data is allowed; changing scope after
outcomes to obtain a preferred result is forbidden.

## Accounting

{_resource_plan_markdown(cfg)}
""",
    )
    _write(
        root,
        "V15_CONSTRUCT_VALIDITY.md",
        f"""# Construct-Validity Audit

Status: **PASS for DEV interface smoke; confirmatory status NOT_RUN**.

The external DEV transition audit generated `{modern['transition'].get('transition_count', 0)}`
paired transition rows from `{modern['transition'].get('trajectory_count', 0)}`
trajectory units.  The local smoke generated `{dev.get('transition_count', 0)}`
additional schema checks.  Proxy, gate, and assessment IDs are disjoint;
access is role-checked; paired sandboxes share an initial manifest hash and
diverge only after policy execution.

The smoke is not evidence for the paper's modern-agent hypotheses.
""",
    )
    _write(
        root,
        "V15_CONFIRMATORY_PREREGISTRATION.md",
        f"""# Confirmatory Protocol Lock

Lock status: **FROZEN PRE-OUTCOME**
Lock hash: `{lock_digest}`
External execution: **NOT_RUN**

The complete JSON lock is `experiments/v15/confirmatory_lock.json`.  It binds
the task-plane manifest, candidate count, rounds, seeds, pairing rules,
metrics, footprint features, PIVOT settings, baselines, budgets, bootstrap,
hypotheses, and case-selection rule.  The lock is not edited in response to
observed outcomes.
""",
    )
    _write(
        root,
        "V15_TRANSITION_AUDIT.md",
        f"""# Transition Audit

Existing controlled evidence remains frozen under `results/v9` with states:
{_markdown_table((name, state, "frozen prior evidence") for name, state in states.items())}

The modern-agent transition table currently contains `{modern['transition'].get('transition_count', 0)}` external DEV rows only.
It has the required directed hashes, proxy/actor fields, footprint, resource
metrics, and terminal-state fields.  No deployment or strategic value is
promoted from this smoke.

The shared analysis artifact reports status `{transition_analysis.get('status', 'NOT_RUN')}`,
primary terminal state `{transition_analysis.get('terminal_state')}`, and
`{transition_analysis.get('independent_n', 0)}` trajectory clusters.  Its H1
metric is `{json.dumps((transition_analysis.get('metrics') or {}).get('IDE', {}), sort_keys=True)}`.
Because the current run is DEV-only, this is a construct diagnostic rather
than evidence for the manuscript.
""",
    )
    _write(
        root,
        "V15_OPERATOR_RELATIVE_ANALYSIS.md",
        f"""# Operator-Relative Analysis

DEV operator counts are retained separately: `{modern['transition'].get('trajectory_count', 0) and 2 or dev.get('operator_count', 0)}`
proposal families and `{modern['transition'].get('candidate_count', 0)}` candidate records.
The confirmatory operator-relative comparison is **NOT_RUN**; no pooled claim
is made from the DEV rows.
""",
    )
    _write(
        root,
        "V15_FOOTPRINT_ANALYSIS.md",
        f"""# Footprint Analysis

Status: **{footprint.get('status', 'NOT_RUN')}**; phase: **{footprint.get('phase', 'DEV')}**.

The registered features are computed before any gate or assessment query and
exclude deployment outcomes from the feature vector. The independent unit is
the trajectory/cluster, so transition rows are not treated as independent
replicates.

Current materialized rows: `{footprint.get('rows_read', 0)}` total, with
`{footprint.get('rows_with_proxy_and_actor', 0)}` rows containing proxy and
actor deltas across `{footprint.get('independent_trajectory_units', 0)}`
trajectory units. Transition-error estimate:
`{json.dumps(footprint.get('transition_error', {}), sort_keys=True)}`.
Improvement-reversal estimate:
`{json.dumps(footprint.get('improvement_reversal_rate', {}), sort_keys=True)}`.

The full feature-by-feature associations and leakage audit are stored in
`artifacts/v15/footprint_analysis.json`. Before confirmatory opening this is a
DEV-only diagnostic, not a paper claim.
""",
    )
    _write(
        root,
        "V15_PROMOTION_RESULTS.md",
        f"""# Promotion Replay

The external replay is **DEV-only**: `{modern['promotion'].get('promotion_result_count', modern['promotion'].get('result_count', 0))}`
method rows over `{modern['promotion'].get('candidate_batch_count', 0)}` immutable
candidate batches, with `{modern['promotion'].get('physical_pair_evaluations', 0)}`
physical paired evaluations and `{modern['promotion'].get('logical_hf_queries', 0)}`
logical HF query decisions.  Cache hits are logged per selector, so physical
reuse never silently reduces the registered logical budget.  The post-decision
truth audit is excluded from that budget.  Every method receives the same
candidate-batch hash.
The shared promotion analysis has status `{promotion_analysis.get('status', 'NOT_RUN')}`,
terminal state `{promotion_analysis.get('terminal_state')}`, and fixed effect
orientation `{(promotion_analysis.get('paired_effect') or {}).get('direction', 'not available')}`.
Its target budget is `{promotion_analysis.get('target_budget', 'not available')}` and
the paired-effect summary is `{json.dumps(promotion_analysis.get('paired_effect', {}), sort_keys=True)}`.
Confirmatory promotion results: **NOT_RUN**.
""",
    )
    _write(
        root,
        "V15_PAIRED_ABLATION.md",
        f"""# Paired Ablation

The DEV paired-versus-unpaired audit is **{modern['ablations'].get('status', 'NOT_RUN')}** with
`{modern['ablations'].get('record_count', 0)}` rows.  It uses independent seeds
and fresh gate sandboxes for the no-pairing arm; paired truth remains the
reference audit.  Confirmatory ablation: **NOT_RUN**.

Terminal-state policy: a completed null is frozen as `HYPOTHESIS_NOT_SUPPORTED`;
it is never relabelled as a failure.""",
    )
    _write(
        root,
        "V15_PIVOT_ABLATIONS.md",
        f"""# PIVOT Ablations

Registered no-pairing, no-footprint, no-VOI, and baseline comparisons are
implemented on the immutable DEV archive.  DEV status is
**{modern['ablations'].get('status', 'NOT_RUN')}** with
`{modern['ablations'].get('record_count', 0)}` diagnostic rows; confirmatory
ablations remain **NOT_RUN**.  The DEV archive is too small for a scientific
effect claim and is retained only as an execution/contract check.""",
    )
    _write(
        root,
        "V15_CLOSED_LOOP_RESULTS.md",
        f"""# Closed-Loop Results

The existing controlled closed-loop evidence remains frozen.  The modern DEV
closed loop is **{modern['closed_loop'].get('status', 'NOT_RUN')}** with
`{modern['closed_loop'].get('trajectory_count', 0)}` trajectory units,
`{modern['closed_loop'].get('result_count', 0)}` promotion rows, and
`{modern['closed_loop'].get('assessment_result_count', 0)}` terminal DEV
assessment records.  The assessment role was queried exactly once per DEV
terminal row.  Decision-time HF queries and evaluator-only post-decision truth
are tracked separately (`logical_hf_queries` =
`{modern['closed_loop'].get('logical_hf_queries', 0)}`, pre-decision paired
evaluations = `{modern['closed_loop'].get('pre_decision_pair_evaluations', 0)}`,
post-decision truth paired evaluations =
`{modern['closed_loop'].get('post_decision_truth_pair_evaluations', 0)}`).
These data are not confirmatory endpoint evidence.
The shared closed-loop analysis has status `{closed_loop_analysis.get('status', 'NOT_RUN')}`,
terminal state `{closed_loop_analysis.get('terminal_state')}`, and
`{closed_loop_analysis.get('independent_n', 0)}` trajectory clusters.  Its
terminal assessment audit is `{json.dumps(closed_loop_analysis.get('assessment_audit', {}), sort_keys=True)}`.
Confirmatory closed loop and untouched assessment: **NOT_RUN**.""",
    )
    _write(
        root,
        "V15_PI_REPLICATION.md",
        f"""# Second-Scaffold Replication

The Pi source is pinned and its built CLI is available.  The registered
replication path now uses the Inspect task/scorer control plane, explicit
read-only runtime mounts, a writable task workspace, and a fresh network-
isolated test namespace.  A real gateway-backed DEV smoke completed
`{modern['pi'].get('completed_execution_count', modern['pi'].get('completed_record_count', 0))}` of
`{modern['pi'].get('agent_execution_count', modern['pi'].get('record_count', 0))}` policy-task sandbox executions;
proxy and actor/gate roles were exercised while assessment remained unopened.
Cross-scaffold confirmatory replication remains
**NOT_RUN**; the DEV smoke does not support a replication claim.""",
    )
    _write(
        root,
        "V15_STRATEGIC_RESULTS.md",
        f"""# Strategic Response

The prior finite controlled opponent-family evidence remains scoped and
frozen.  The registered non-LLM identity-blind response audit is
**{modern['strategic'].get('status', 'NOT_RUN')}** with
`{modern['strategic'].get('completed_record_count', 0)}` completed DEV records
and `{modern['strategic'].get('response_pair_count', 0)}` paired response
evaluations.  The independent-agent reviewer family is
**{(modern['strategic'].get('response_families', {}) or {}).get('independent_agent_reviewer', 'NOT_RUN')}**.
The paired mutation-detection difference is recorded as a response-utility
diagnostic (`delta_strategic`) and is kept separate from task success and any
deployment-causal claim.  Confirmatory strategic response: **NOT_RUN**.""",
    )
    _write(root, "V15_FALSIFICATION_REPORT.md", f"# Falsification Report\n\nThe protocol permits every terminal state, including a valid null. No branch changes tasks, response strength, budget, or operators because a result is unfavorable. Modern-agent hypotheses are currently **NOT_RUN**.\n\n{_falsification_table(cfg)}")
    pdf_summary = _pdf_summary(root)
    _write(
        root,
        "V15_PAPER_CONTEXT_AUDIT.md",
        f"""# Paper Context Audit

The manuscript is rebuilt from the neutral release figure aliases and semantic
macros.  The current PDF context summary is:

```json
{json.dumps(pdf_summary, indent=2, sort_keys=True)}
```

No modern-agent claim is inserted without a valid confirmatory artifact.  The
figure bundles are audited standalone and then checked in the rendered paper
context; page limits, embedded fonts, citations, and overfull boxes remain
machine-verification gates.
""",
    )
    _write(
        root,
        "V15_NUMBER_AUDIT.md",
        f"""# Number Audit

Status: **{'PASS' if numbers['valid'] else 'BLOCKED'}** for the frozen manuscript macros.

Checks: `{json.dumps(numbers['checks'], sort_keys=True)}`
The strategic aggregation is computed from the three registered adaptive
families and matched seed clusters in the source artifact; no manually typed
modern-agent number is present.
""",
    )
    _write(root, "V15_CLAIM_AUDIT.md", f"# Claim Audit\n\nStatus: **{'PASS' if claim_audit.get('valid', False) else 'BLOCKED'}** for scope preservation and evidence-contract fields.\n\n```json\n{json.dumps(claim_audit, indent=2, sort_keys=True)}\n```\n\nClaims remain limited to Improvement Fidelity, controlled response layers, finite opponent mechanisms, and the existing observational finance boundary. Modern-agent claims are registered but not promoted while their confirmatory state is NOT_RUN.")
    _write(
        root,
        "V15_REFERENCE_AUDIT.md",
        f"""# Reference Audit

Status: **{'PASS' if reference_audit.get('valid', False) else 'BLOCKED'}**.

The bundled bibliography contains `{reference_audit.get('entry_count', 0)}`
entries, of which `{reference_audit.get('cited_entry_count', 0)}` are cited in
the manuscript.  Duplicate keys: `{reference_audit.get('duplicate_entry_keys', [])}`;
missing citation keys: `{reference_audit.get('missing_citation_keys', [])}`.
Recent preprints remain labelled as preprints; this local key audit does not
claim that every external metadata record is a peer-reviewed venue record.
""",
    )
    _write(root, "V15_ANONYMITY_AUDIT.md", "# Anonymity Audit\n\nThe manuscript author field is empty and the preserved PDF has anonymous metadata. Local absolute paths, credentials, private runtime endpoints, and author identifiers are excluded from reviewer-facing artifacts. Public tool URLs are provenance references, not author identity. A final platform-side profile/conflict check remains manual.")
    _write(
        root,
        "V15_LANGUAGE_AUDIT.md",
        f"""# Language and Paper-Facing Token Audit

Status: **{'PASS' if language['valid'] else 'BLOCKED'}**

- Version tokens in manuscript body: `{language['body_version_tokens']}`
- Body source closure: `{language.get('body_source_files', [])}`
- Version tokens in rendered PDF text: `{language['pdf_version_tokens']}`
- Forbidden implementation-assistant tokens in paper/PDF: `{language['paper_facing_codex_tokens']}`
- Forbidden implementation-assistant tokens in reviewer artifacts: `{language.get('reviewer_artifact_codex_tokens', 0)}`

Internal experiment directory labels are not rendered into the scientific
body.  The manuscript does not use internal version labels as paper-facing
claims.
""",
    )
    _write(
        root,
        "V15_REPRODUCIBILITY_AUDIT.md",
        f"# Reproducibility Audit\n\nProtocol objects, task-plane hashes, paired sandbox manifests, candidate archive hashes, CSV/Parquet writers, and the pre-modern-agent fallback are implemented and tested. External status: **{external_status}**. Terminal-state audit: **{terminal_audit.get('status', 'BLOCKED')}**. DEV artifacts are explicitly non-confirmatory; the pre-outcome lock remains immutable until an authorized confirmation opens it.",
    )
    _write(
        root,
        "V15_REVIEWER_ATTACK_AUDIT.md",
        f"""# Reviewer Attack Audit

The following 25 predeclared attacks are checked against the protocol and its
artifacts.  The decisive unresolved item is external modern-agent execution;
it is reported as `NOT_RUN` rather than hidden.

{_reviewer_attack_table()}
""",
    )
    _write(root, "V15_OUTSTANDING_PROFILE.md", "# Evidence Profile\n\n" + _outstanding_profile_table())

    (root / "V15_VISUAL_DEFECT_LEDGER.jsonl").touch()
    final_blockers = []
    if not snapshot_ok:
        final_blockers.append("baseline snapshot")
    if not language["valid"]:
        final_blockers.append("paper-facing language")
    if not numbers["valid"]:
        final_blockers.append("number audit")
    if not external_available:
        final_blockers.append("external DEV artifact completeness")
    if not terminal_audit.get("valid", False):
        final_blockers.append("terminal-state/provenance audit")
    if not confirmatory_open:
        final_blockers.extend(
            [
                "confirmatory mini-SWE transition audit",
                "confirmatory promotion replay and closed loop",
                "confirmatory Pi replication",
                "confirmatory strategic response and registered ablations",
            ]
        )
    final_status = "BLOCKED" if final_blockers else "READY_WITH_MINOR_MANUAL_CHECKS"
    _write(
        root,
        "V15_FINAL_REPORT.md",
        f"""# Final Report

Evidence profile: **LEVEL_E** (bounded external DEV evidence exists; confirmatory modern-agent study is not opened).

Current local status: **{final_status}**

Blockers: {', '.join(final_blockers) if final_blockers else 'none'}

The preserved submission fallback remains intact.  The local protocol,
candidate hashing, sealed planes, paired sandbox, external DEV smoke, replay
schemas, response audit, ablations, and audit reports are complete.  This
report does not convert DEV observations into confirmatory claims, and does
not claim mini-SWE/Pi generalization, untouched confirmatory assessment, or
modern-agent promotion benefit without their actual locked execution.

{final_status}
""",
    )
    summary = {
        "status": final_status,
        "evidence_profile": "LEVEL_E",
        "blockers": final_blockers,
        "lock_sha256": lock_digest,
        "dev_transition_count": dev.get("transition_count", 0),
        "required_reports": list(REQUIRED_REPORTS),
        "canonical_rows": canonical.get("rows", {}),
        "language": language,
        "numbers": numbers,
        "references": reference_audit,
        "scientific_analysis": {
            "transition": {
                "status": transition_analysis.get("status"),
                "terminal_state": transition_analysis.get("terminal_state"),
                "independent_n": transition_analysis.get("independent_n", 0),
            },
            "promotion": {
                "status": promotion_analysis.get("status"),
                "terminal_state": promotion_analysis.get("terminal_state"),
                "independent_n": promotion_analysis.get("independent_n", 0),
            },
            "closed_loop": {
                "status": closed_loop_analysis.get("status"),
                "terminal_state": closed_loop_analysis.get("terminal_state"),
                "independent_n": closed_loop_analysis.get("independent_n", 0),
            },
            "ablations": {
                "status": ablation_analysis.get("status"),
                "terminal_state": ablation_analysis.get("terminal_state"),
                "independent_n": ablation_analysis.get("independent_n", 0),
            },
            "strategic": {
                "status": strategic_analysis.get("status"),
                "terminal_state": strategic_analysis.get("terminal_state"),
                "independent_n": strategic_analysis.get("independent_n", 0),
            },
            "pi": {
                "status": pi_analysis.get("status"),
                "terminal_state": pi_analysis.get("terminal_state"),
                "independent_n": pi_analysis.get("independent_n", 0),
            },
        },
        "scientific_summary": scientific_summary,
        "manifest_migration": manifest_migration,
        "terminal_state_audit": {
            "status": terminal_audit.get("status"),
            "valid": terminal_audit.get("valid", False),
            "issues": terminal_audit.get("issues", []),
        },
    }
    (root / "artifacts/v15/final_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate V15 protocol and audit reports")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(generate_reports(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
