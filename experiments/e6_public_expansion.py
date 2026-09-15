#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.analysis.public_finance_expansion import run_public_finance_expansion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen public finance expansion")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_public_finance_expansion(args.config, args.output)
    print(
        json.dumps(
            {
                "grid_id": summary["grid_id"],
                "complete_grid": summary["complete_grid"],
                "n_rows": summary["n_rows"],
                "n_primary_sessions": summary["n_primary_sessions"],
                "n_failed_assets": summary["n_failed_assets"],
                "status": summary["status"],
            },
            sort_keys=True,
        )
    )
    if not summary["complete_grid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
