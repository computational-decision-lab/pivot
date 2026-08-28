"""Small command dispatcher for the V15 protocol surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audit_anonymity import audit_anonymity
from .audit_claims import audit_claims
from .audit_language import audit_language
from .audit_numbers import audit_numbers
from .audit_references import audit_references
from .audit_reproducibility import audit_reproducibility
from .commands import (
    analyze_all,
    analyze_closed_loop,
    analyze_footprint,
    analyze_promotion,
    analyze_transitions,
    approve_figures,
    audit_repo,
    audit_terminal_states,
    dev_construct,
    dev_external_smoke,
    dev_resource_plan,
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
from .finalize import finalize as run_finalize


def _reexec_external_command(root: Path, argv: list[str]) -> None:
    """Run external commands in the pinned Python runtime.

    The project virtual environment intentionally remains lightweight.  Agent
    execution commands therefore re-exec in the pinned Inspect/mini-SWE
    environment while keeping the repository importable through ``PYTHONPATH``.
    """

    from .external_runtime import locked_runtime_python, running_under_locked_runtime

    if os.getenv("PIVOT_V15_RUNTIME_REEXEC") == "1" or running_under_locked_runtime(root):
        return
    runtime = locked_runtime_python(root)
    environment = os.environ.copy()
    environment["PIVOT_V15_RUNTIME_REEXEC"] = "1"
    project_paths = (str(root), str(root / "src"))
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        os.pathsep.join((*project_paths, existing_path)) if existing_path else os.pathsep.join(project_paths)
    )
    os.chdir(root)
    os.execve(str(runtime), [str(runtime), "-m", "experiments.v15", *argv], environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 modern-agent protocol commands")
    parser.add_argument(
        "command",
        choices=(
            "snapshot", "audit-repo", "validate-inspect", "validate-mini-swe", "validate-pi",
            "validate-sandbox", "validate-pivot-core", "dev-smoke", "dev-construct",
            "dev-resource-plan", "dev-external-smoke", "freeze", "run-transitions", "analyze-transitions",
            "analyze-footprint", "analyze-all", "audit-terminal-states", "repair-manifests",
            "freeze-candidates", "run-promotion-replay", "analyze-promotion", "run-closed-loop",
            "analyze-closed-loop", "run-assessment", "run-pi-replication", "run-strategic",
            "run-ablations", "figures", "approve-figures", "reports", "finalize",
            "audit-numbers", "audit-claims", "audit-references", "audit-anonymity",
            "audit-language", "audit-reproducibility",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--external", action="store_true", help="run a pinned external-scaffold phase")
    parser.add_argument("--dev", action="store_true", help="keep an external phase DEV-only")
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--agent-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--verify-image", action="store_true")
    parser.add_argument("--agent-reviewer", action="store_true", help="enable the optional independent reviewer in DEV")
    args = parser.parse_args()
    root = args.root.resolve()
    external_commands = {
        "dev-external-smoke",
        "run-transitions",
        "run-promotion-replay",
        "run-closed-loop",
        "run-assessment",
        "run-pi-replication",
        "run-strategic",
        "run-ablations",
    }
    if args.command in external_commands and (args.external or args.command == "dev-external-smoke"):
        _reexec_external_command(root, sys.argv[1:])
    dispatch: dict[str, Callable[[], dict[str, Any]]] = {
        "snapshot": lambda: audit_repo(root),
        "audit-repo": lambda: audit_repo(root),
        "validate-inspect": lambda: validate_inspect(root),
        "validate-mini-swe": lambda: validate_mini_swe(root),
        "validate-pi": lambda: validate_pi(root),
        "validate-sandbox": lambda: validate_sandbox(root),
        "validate-pivot-core": lambda: validate_pivot_core(root),
        "dev-smoke": lambda: dev_smoke(root),
        "dev-construct": lambda: dev_construct(root),
        "dev-resource-plan": lambda: dev_resource_plan(root),
        "dev-external-smoke": lambda: dev_external_smoke(
            root,
            seed=args.seed,
            task_limit=args.task_limit or 1,
        ),
        "freeze": lambda: freeze(root),
        "run-transitions": lambda: (
            run_transitions(root)
            if not args.external
            else __import__("experiments.v15.external_study", fromlist=["run_transition_audit"]).run_transition_audit(
                root,
                confirmatory=not args.dev,
                trajectory_limit=args.trajectories,
                round_limit=args.rounds,
                candidates_per_round=args.candidates,
                task_limit=args.task_limit,
                agent_steps=args.agent_steps,
                verify_image=args.verify_image,
            )
        ),
        "analyze-transitions": lambda: analyze_transitions(root),
        "analyze-footprint": lambda: analyze_footprint(root),
        "analyze-all": lambda: analyze_all(root),
        "audit-terminal-states": lambda: audit_terminal_states(root),
        "repair-manifests": lambda: repair_manifests(root),
        "freeze-candidates": lambda: freeze_candidates(root),
        "run-promotion-replay": lambda: (
            run_promotion_replay(root)
            if not args.external
            else __import__("experiments.v15.external_promotion", fromlist=["run_external_promotion"]).run_external_promotion(
                root,
                confirmatory=not args.dev,
                task_limit=args.task_limit,
                agent_steps=args.agent_steps,
                budgets=(1, 2, 4),
                verify_image=args.verify_image,
            )
        ),
        "analyze-promotion": lambda: analyze_promotion(root),
        "run-closed-loop": lambda: (
            run_closed_loop(root)
            if not args.external
            else __import__("experiments.v15.external_closed_loop", fromlist=["run_external_closed_loop"]).run_external_closed_loop(
                root,
                confirmatory=not args.dev,
                trajectory_limit=args.trajectories,
                round_limit=args.rounds,
                candidates_per_round=args.candidates,
                task_limit=args.task_limit,
                assessment_limit=args.task_limit,
                agent_steps=args.agent_steps,
                verify_image=args.verify_image,
            )
        ),
        "analyze-closed-loop": lambda: analyze_closed_loop(root),
        "run-assessment": lambda: (
            run_assessment(root)
            if not args.external
            else __import__("experiments.v15.external_closed_loop", fromlist=["summarize_terminal_assessment"]).summarize_terminal_assessment(
                root,
                confirmatory=not args.dev,
            )
        ),
        "run-pi-replication": lambda: (
            run_pi_replication(root)
            if not args.external
            else __import__("experiments.v15.run_pi_replication", fromlist=["run_pi_replication"])
            .run_pi_replication(
                root,
                confirmatory=not args.dev,
                task_limit=args.task_limit,
                agent_steps=args.agent_steps,
            )
        ),
        "run-strategic": lambda: (
            run_strategic(root)
            if not args.external
            else __import__("experiments.v15.run_strategic", fromlist=["run_strategic"])
            .run_strategic(
                root,
                confirmatory=not args.dev,
                task_limit=args.task_limit,
                enable_agent_reviewer=args.agent_reviewer,
            )
        ),
        "run-ablations": lambda: (
            run_ablations(root)
            if not args.external
            else __import__("experiments.v15.run_ablations", fromlist=["run_ablations"])
            .run_ablations(
                root,
                confirmatory=not args.dev,
                verify_image=args.verify_image,
                task_limit=args.task_limit,
            )
        ),
        "figures": lambda: figures(root),
        "approve-figures": lambda: approve_figures(root),
        "reports": lambda: reports(root),
        "finalize": lambda: run_finalize(root),
        "audit-numbers": lambda: audit_numbers(root),
        "audit-claims": lambda: audit_claims(root),
        "audit-references": lambda: audit_references(root),
        "audit-anonymity": lambda: audit_anonymity(root),
        "audit-language": lambda: audit_language(root),
        "audit-reproducibility": lambda: audit_reproducibility(root),
    }
    result = dispatch[str(args.command)]()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
