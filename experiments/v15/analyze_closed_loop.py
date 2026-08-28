from __future__ import annotations

import json
from pathlib import Path

from .commands import analyze_closed_loop

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze closed-loop result schemas")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(analyze_closed_loop(args.root.resolve()), sort_keys=True))
