from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the pre-outcome confirmatory protocol")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(freeze(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
