from __future__ import annotations

import json
from pathlib import Path

from .commands import not_run


def run_assessment(root: Path) -> dict[str, object]:
    return not_run(root, "SEALED_ASSESSMENT")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the untouched sealed assessment")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if not args.external:
        result = run_assessment(args.root.resolve())
    else:
        from .external_closed_loop import summarize_terminal_assessment

        result = summarize_terminal_assessment(args.root.resolve(), confirmatory=not args.dev)
    print(json.dumps(result, sort_keys=True))
