#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.analysis.registry import run_registered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen PIVOT registry")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    result = run_registered(args.registry, args.output, project_root=Path(__file__).resolve().parents[1], fail_fast=args.fail_fast)
    (args.output / "registry_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("experiment", "n_runs", "n_ok", "n_failed")}))
    if result["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
