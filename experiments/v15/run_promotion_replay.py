from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import run_promotion_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay promotion methods on frozen candidates")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--external", action="store_true", help="run paired Inspect/mini-SWE gate evaluation")
    parser.add_argument("--dev", action="store_true", help="keep the external replay DEV-only")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--agent-steps", type=int, default=None)
    parser.add_argument("--verify-image", action="store_true")
    args = parser.parse_args()
    if not args.external:
        result = run_promotion_replay(args.root.resolve())
    else:
        from .external_promotion import run_external_promotion

        result = run_external_promotion(
            args.root.resolve(),
            confirmatory=not args.dev,
            task_limit=args.task_limit,
            agent_steps=args.agent_steps,
            verify_image=args.verify_image,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
