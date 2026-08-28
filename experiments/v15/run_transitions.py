from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .commands import run_transitions
from .external_runtime import locked_runtime_python, running_under_locked_runtime
from .external_study import run_transition_audit


def _reexec_locked_runtime(root: Path, argv: list[str]) -> None:
    """Re-exec external runs with the immutable runtime from the lock."""

    if "--external" not in argv or os.getenv("PIVOT_V15_RUNTIME_REEXEC") == "1":
        return
    if running_under_locked_runtime(root):
        return
    try:
        runtime = locked_runtime_python(root)
    except (FileNotFoundError, ValueError):
        # The downstream resolver reports the missing runtime with its full
        # provenance context; do not substitute the project interpreter.
        return
    root = Path(root).resolve()
    env = os.environ.copy()
    env["PIVOT_V15_RUNTIME_REEXEC"] = "1"
    # The pinned environment owns external dependencies only.  Preserve the
    # project package import path when the caller starts from the project venv
    # or a different working directory.
    project_paths = (str(root), str(root / "src"))
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join((*project_paths, existing_path)) if existing_path else os.pathsep.join(project_paths)
    os.chdir(root)
    os.execve(str(runtime), [str(runtime), "-m", "experiments.v15.run_transitions", *argv], env)


def main() -> None:
    root_hint = Path.cwd().resolve()
    if "--root" in sys.argv:
        try:
            root_hint = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
        except (IndexError, ValueError):
            pass
    _reexec_locked_runtime(root_hint, sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run the locked transition audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--external", action="store_true", help="run the pinned external scaffold")
    parser.add_argument("--dev", action="store_true", help="keep the external run DEV-only")
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--agent-steps", type=int, default=None)
    parser.add_argument("--verify-image", action="store_true")
    args = parser.parse_args()
    if not args.external:
        result = run_transitions(args.root.resolve())
    else:
        result = run_transition_audit(
            args.root.resolve(),
            confirmatory=not args.dev,
            trajectory_limit=args.trajectories,
            round_limit=args.rounds,
            candidates_per_round=args.candidates,
            task_limit=args.task_limit,
            agent_steps=args.agent_steps,
            verify_image=args.verify_image,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
