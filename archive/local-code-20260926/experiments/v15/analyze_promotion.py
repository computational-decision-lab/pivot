from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import analyze_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze promotion replay artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(analyze_promotion(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
