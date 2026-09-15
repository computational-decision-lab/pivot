from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import validate_inspect


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the Inspect control plane without running agents")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate_inspect(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
