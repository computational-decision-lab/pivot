#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.analysis.public_finance import run_public_finance_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E6 public observational calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e6-public"))
    args = parser.parse_args()
    summary = run_public_finance_calibration(args.config, args.output)
    print(
        json.dumps(
            {
                "n_sessions": summary["n_sessions"],
                "n_rows": summary["n_rows"],
                "status": summary["status"],
                "gate_e_promoted": summary["gate_e_promoted"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
