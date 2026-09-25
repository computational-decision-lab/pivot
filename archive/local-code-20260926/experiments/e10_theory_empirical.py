#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pivot.theory.empirical import run_theory_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V6 empirical theory checks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e10-theory-empirical"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = run_theory_experiment(args.output, config)
    print(json.dumps({"output": str(args.output), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
