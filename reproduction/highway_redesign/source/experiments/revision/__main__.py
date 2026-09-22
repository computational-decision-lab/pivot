"""Local DEV-only revision CLI; no cloud resource operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_revision_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("e3c", "e5c"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--environment",
        choices=("controlled", "congestion_resource", "openspiel", "physical", "mpe2"),
        default="controlled",
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--world-config-json",
        default="{}",
        help="JSON object passed to the selected world's config",
    )
    args = parser.parse_args()
    try:
        world_config = json.loads(args.world_config_json)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --world-config-json: {error}")
    if not isinstance(world_config, dict):
        parser.error("--world-config-json must be an object")
    result = run_revision_smoke(
        args.experiment,
        root=args.root.resolve(),
        output=args.output.resolve(),
        seed=args.seed,
        resume=args.resume,
        environment=args.environment,
        environment_options=world_config,
        **({"methods": args.methods} if args.methods else {}),
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "status": result["status"],
                "methods": result["methods"],
                "result_rows": len(result["results"]),
                "output": str(args.output.resolve()),
                "hf_query_count": result["cost_ledger"]["hf_query_count"],
                "hf_cost": result["cost_ledger"]["hf_cost"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
