from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import e2c_operator_shift, e3c_closed_loop, e4c_learned_ood, e5c_efficiency, e7c_strategic
from .common import load_profile


def _dispatch(experiment: str) -> Callable[..., dict[str, Any]]:
    if experiment == "e2c":
        return e2c_operator_shift.run
    if experiment == "e3c":
        return e3c_closed_loop.run
    if experiment == "e4c":
        return e4c_learned_ood.run
    if experiment == "e5c":
        return e5c_efficiency.run
    if experiment == "e7c":
        return e7c_strategic.run
    raise ValueError(f"V9 experiment is not implemented yet: {experiment}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible PIVOT V9 experiment")
    parser.add_argument("--experiment", choices=["e2c", "e3c", "e4c", "e5c", "e7c"], required=True)
    parser.add_argument("--profile", choices=["smoke", "dev", "confirmatory"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    runner = _dispatch(args.experiment)
    decision = runner(
        root / f"configs/v9/{args.experiment}.yaml",
        profile=load_profile(root, args.profile),
        output=args.output,
        root=root,
        resume=args.resume,
    )
    print(json.dumps({"experiment": args.experiment, "profile": args.profile, "status": decision["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
