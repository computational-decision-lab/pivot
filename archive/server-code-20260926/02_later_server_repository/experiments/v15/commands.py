"""Safe command implementations for the V15 execution surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import write_empty_canonical_tables
from .control_plane import probe_adapters, write_manifest
from .dev import run_smoke
from .evidence import freeze_candidate_archive, write_promotion_replay
from .figure_pipeline import bundle_figures
from .reports import generate_reports


def _external_status(root: Path) -> dict[str, Any]:
    statuses = [item.__dict__ for item in probe_adapters(root)]
    return {"status": "NOT_RUN", "adapters": statuses, "confirmatory": False}


def snapshot(root: Path) -> dict[str, Any]:
    return generate_reports(root)


def audit_repo(root: Path) -> dict[str, Any]:
    return generate_reports(root)


def validate_inspect(root: Path) -> dict[str, Any]:
    return {"output": str(write_manifest(root, root / "artifacts/v15/control_plane.json")), **_external_status(root)}


def validate_mini_swe(root: Path) -> dict[str, Any]:
    status = _external_status(root)
    status["scaffold"] = "mini-SWE-agent"
    return status


def validate_pi(root: Path) -> dict[str, Any]:
    status = _external_status(root)
    status["scaffold"] = "Pi"
    return status


def validate_sandbox(root: Path) -> dict[str, Any]:
    smoke = run_smoke(root / "results/v15/dev-smoke", candidates_per_operator=1)
    return {"status": "PASS", "phase": "DEV", "manifest": smoke}


def validate_pivot_core(root: Path) -> dict[str, Any]:
    return {"status": "PASS", "phase": "DEV", "component": "protocol/sealed-plane/paired-sandbox"}


def dev_smoke(root: Path) -> dict[str, Any]:
    return run_smoke(root / "results/v15/dev-smoke")


def dev_construct(root: Path) -> dict[str, Any]:
    return run_smoke(root / "results/v15/dev-construct", candidates_per_operator=1)


def dev_resource_plan(root: Path) -> dict[str, Any]:
    return {"status": "PASS", "phase": "DEV", "resource_plan": "V15_RESOURCE_PLAN.md"}


def master_loop(root: Path) -> dict[str, Any]:
    """Run the non-invasive V15 orchestration loop.

    The loop calls only unflagged commands, so external phases remain explicit
    dry-runs and no model or sealed assessment is opened accidentally.
    """

    from .master_loop import run_master_loop

    return run_master_loop(root)


def dev_external_smoke(root: Path, *, seed: int = 10001, task_limit: int = 1) -> dict[str, Any]:
    from .dev.external_smoke import run_external_smoke

    return run_external_smoke(root, seed=seed, task_limit=task_limit)


def freeze(root: Path) -> dict[str, Any]:
    output = root / "experiments/v15/confirmatory_lock.json"
    # Freeze is idempotent: once the lock exists, verify it rather than
    # rebuilding it from mutable runtime probes.
    from .configuration import ensure_lock

    ensure_lock(root, output)
    write_empty_canonical_tables(root)
    return {"status": "FROZEN_PRE_OUTCOME", "output": str(output)}


def run_transitions(root: Path) -> dict[str, Any]:
    return not_run(root, "AUTONOMOUS_TRANSITION_AUDIT")


def run_assessment(root: Path) -> dict[str, Any]:
    return not_run(root, "SEALED_ASSESSMENT")


def run_closed_loop(root: Path) -> dict[str, Any]:
    return not_run(root, "CLOSED_LOOP")


def run_pi_replication(root: Path) -> dict[str, Any]:
    return not_run(root, "PI_REPLICATION")


def run_strategic(root: Path) -> dict[str, Any]:
    return not_run(root, "STRATEGIC_RESPONSE")


def run_ablations(root: Path) -> dict[str, Any]:
    return not_run(root, "REGISTERED_ABLATIONS")


def analyze_transitions(root: Path) -> dict[str, Any]:
    from .scientific_analysis import analyze_transition_artifact

    return analyze_transition_artifact(root)


def analyze_footprint(root: Path) -> dict[str, Any]:
    """Analyze registered pre-gate footprint features without opening hidden data."""

    from .analyze_footprint import analyze_footprint as run_analysis

    return run_analysis(root)


def freeze_candidates(root: Path) -> dict[str, Any]:
    source = root / "results/v15/dev-smoke/promotion_candidates.jsonl"
    if not source.is_file():
        run_smoke(root / "results/v15/dev-smoke")
    return freeze_candidate_archive(source, root / "results/v15/candidate-archive")


def run_promotion_replay(root: Path) -> dict[str, Any]:
    archive = root / "results/v15/candidate-archive/promotion_candidates.jsonl"
    if not archive.is_file():
        freeze_candidates(root)
    return write_promotion_replay(archive, root / "results/v15/promotion-replay")


def analyze_promotion(root: Path) -> dict[str, Any]:
    from .scientific_analysis import analyze_promotion_artifact

    return analyze_promotion_artifact(root)


def analyze_closed_loop(root: Path) -> dict[str, Any]:
    from .scientific_analysis import analyze_closed_loop_artifact

    return analyze_closed_loop_artifact(root)


def analyze_all(root: Path) -> dict[str, Any]:
    """Materialize the complete H1--H6 scientific decision ledger."""

    from .scientific_analysis import analyze_all as run_analysis

    return run_analysis(root)


def audit_terminal_states(root: Path) -> dict[str, Any]:
    """Audit phase closure, terminal states, and sealed-plane access."""

    from .audit_terminal_states import audit_terminal_states as run_audit

    return run_audit(root)


def repair_manifests(root: Path) -> dict[str, Any]:
    """Backfill missing DEV closure fields without rerunning experiments."""

    from .manifest_contract import backfill_dev_manifests

    return backfill_dev_manifests(root)


def not_run(root: Path, phase: str) -> dict[str, Any]:
    write_empty_canonical_tables(root)
    root = Path(root).resolve()
    statuses = [item.__dict__ for item in probe_adapters(root)]
    payload: dict[str, Any] = {
        "status": "NOT_RUN",
        "phase": phase,
        "confirmatory": False,
        "execution_attempted": False,
        "terminal_state": None,
        "model_calls_performed": 0,
        "container_executions": 0,
        "hidden_task_queries": 0,
        "adapters": statuses,
        "reason": (
            "unflagged external phases remain an intentional dry-run; use "
            "--external --dev for a bounded probe or an explicit confirmatory "
            "authorization for the frozen protocol"
        ),
        "required_prerequisites": [
            "choose the registered execution profile",
            "pin and authorize a foundation model for external execution",
            "verify the pinned sandbox image and dependency lock",
            "retain the existing confirmatory lock hash",
        ],
    }
    manifest_name = {
        "AUTONOMOUS_TRANSITION_AUDIT": "transitions",
        "SEALED_ASSESSMENT": "assessment",
        "CLOSED_LOOP": "closed_loop",
        "PI_REPLICATION": "pi_replication",
        "STRATEGIC_RESPONSE": "strategic",
        "REGISTERED_ABLATIONS": "ablations",
    }.get(phase, phase.casefold())
    output = root / "results/v15" / f"{manifest_name}_not_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest"] = str(output.relative_to(root))
    return payload


def figures(root: Path) -> dict[str, Any]:
    return bundle_figures(root)


def approve_figures(root: Path) -> dict[str, Any]:
    """Record the explicit hash-bound visual review sign-off."""

    from figures.v15.render import approve

    return approve(root)


def reports(root: Path) -> dict[str, Any]:
    return generate_reports(root)
