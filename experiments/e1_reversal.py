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
    parser = argparse.ArgumentParser(description="PIVOT E1 Improvement Reversal")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e1-reversal"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = run_first_milestone(
        args.output,
        PerformativeConfig(**payload.get("world", {})),
        payload.get("seeds", []),
        payload.get("candidate_scales", []),
        response_strengths=payload.get("response_strengths"),
        optimization_strengths=payload.get("optimization_strengths"),
    )
    print(f"E1 rows={manifest.row_count} output={args.output}")


if __name__ == "__main__":
    main()
