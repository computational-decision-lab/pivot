from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..commands import dev_construct


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V15 DEV construct check")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(dev_construct(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
