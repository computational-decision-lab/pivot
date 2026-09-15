from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..commands import dev_resource_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the V15 resource plan")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(dev_resource_plan(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
