from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import validate_mini_swe


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the mini-SWE-agent adapter")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate_mini_swe(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
