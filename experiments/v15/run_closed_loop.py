from __future__ import annotations

import json
from pathlib import Path

from .commands import not_run


def run_closed_loop(root: Path) -> dict[str, object]:
    return not_run(root, "CLOSED_LOOP")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the locked modern-agent closed loop")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--agent-steps", type=int, default=None)
    parser.add_argument("--verify-image", action="store_true")
    args = parser.parse_args()
    if not args.external:
        result = run_closed_loop(args.root.resolve())
    else:
        from .external_closed_loop import run_external_closed_loop

        result = run_external_closed_loop(
            args.root.resolve(),
            confirmatory=not args.dev,
            trajectory_limit=args.trajectories,
            round_limit=args.rounds,
            candidates_per_round=args.candidates,
            task_limit=args.task_limit,
            assessment_limit=args.task_limit,
            agent_steps=args.agent_steps,
            verify_image=args.verify_image,
        )
    print(json.dumps(result, sort_keys=True))
