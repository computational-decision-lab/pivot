from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the preserved pre-modern-agent snapshot")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
