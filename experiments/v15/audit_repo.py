from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import audit_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the V15 repository")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(audit_repo(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
