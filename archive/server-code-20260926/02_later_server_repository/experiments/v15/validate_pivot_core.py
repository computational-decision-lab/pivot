from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import validate_pivot_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate agent-agnostic PIVOT protocol components")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate_pivot_core(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
