#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled PIVOT sweep")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--milestone", default="first", choices=["first"])
    parser.add_argument("--output", type=Path, default=Path("results/raw/controlled-first"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    world_config = PerformativeConfig(**config.get("world", {}))
    seeds = config.get("seeds", [1, 2, 3, 4, 5])
    scales = config.get("candidate_scales", [0.1, 0.2, 0.4])
    response_strengths = config.get("response_strengths")
    optimization_strengths = config.get("optimization_strengths")
    manifest = run_first_milestone(
        args.output,
        world_config,
        seeds,
        scales,
        response_strengths=response_strengths,
        optimization_strengths=optimization_strengths,
    )
    print(f"rows={manifest.row_count} storage={manifest.storage_format} output={args.output}")


if __name__ == "__main__":
    main()
