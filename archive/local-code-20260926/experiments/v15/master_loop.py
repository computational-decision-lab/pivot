"""Non-invasive V15 orchestration entry point.

The master loop performs only pre-outcome local preparation when external
scaffolds are unavailable.  It never substitutes a local smoke for the locked
confirmatory run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .commands import (
    analyze_all,
    analyze_closed_loop,
    analyze_promotion,
    analyze_transitions,
    audit_terminal_states,
    dev_smoke,
    figures,
    freeze,
    freeze_candidates,
    repair_manifests,
    reports,
    run_ablations,
    run_assessment,
    run_closed_loop,
    run_pi_replication,
    run_promotion_replay,
    run_strategic,
    run_transitions,
    validate_inspect,
    validate_mini_swe,
    validate_pi,
    validate_pivot_core,
    validate_sandbox,
)


def run_master_loop(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    results = {
        "validate_inspect": validate_inspect(root),
        "validate_mini_swe": validate_mini_swe(root),
        "validate_pi": validate_pi(root),
        "validate_sandbox": validate_sandbox(root),
        "validate_pivot_core": validate_pivot_core(root),
        "dev_smoke": dev_smoke(root),
        "freeze": freeze(root),
        "transitions": run_transitions(root),
        "transition_analysis": analyze_transitions(root),
        "freeze_candidates": freeze_candidates(root),
        "promotion_replay": run_promotion_replay(root),
        "promotion_analysis": analyze_promotion(root),
        "closed_loop": run_closed_loop(root),
        "assessment": run_assessment(root),
        "closed_loop_analysis": analyze_closed_loop(root),
        "scientific_analysis": analyze_all(root),
        "pi_replication": run_pi_replication(root),
        "strategic": run_strategic(root),
        "ablations": run_ablations(root),
        "figures": figures(root),
        "reports": reports(root),
        "manifest_contract": repair_manifests(root),
        "terminal_state_audit": audit_terminal_states(root),
    }
    return {
        "status": str(results["reports"].get("status", "UNKNOWN")),
        "steps": results,
        "outcome_chasing": False,
        "scientific_execution": {
            "primary_scaffold": results["transitions"].get("status", "UNKNOWN"),
            "untouched_assessment": results["assessment"].get("status", "UNKNOWN"),
            "replication": results["pi_replication"].get("status", "UNKNOWN"),
            "strategic_response": results["strategic"].get("status", "UNKNOWN"),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and audit the V15 protocol without outcome execution")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_master_loop(args.root.resolve()), sort_keys=True))
