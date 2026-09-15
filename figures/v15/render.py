from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.v15.figure_pipeline import bundle_figures, write_visual_review_manifest


def render(
    root: Path, inspect_stamp: str | None = None, *, all_figures: bool = False
) -> dict[str, Any]:
    """Bundle either the registered figure set or (by default) the same set.

    ``--all`` is an explicit command-surface flag so scripted figure loops can
    state their intent.  The current manifest is the only supported selection;
    silently rendering a subset would make the final status ambiguous.
    """

    return bundle_figures(Path(root).resolve(), inspect_stamp=inspect_stamp)


def approve(
    root: Path,
    *,
    reviewer: str = "authors",
    note: str = "Standalone, print-size, grayscale, and paper-context visual review completed.",
) -> dict[str, Any]:
    """Record a hash-bound visual sign-off after the render/view/fix loop."""

    return write_visual_review_manifest(Path(root).resolve(), reviewer=reviewer, note=note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle publication figures with passports")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inspect-stamp")
    parser.add_argument("--all", action="store_true", help="bundle every manifest figure")
    parser.add_argument("--approve", action="store_true", help="record an explicit hash-bound visual review")
    parser.add_argument("--reviewer", default="authors")
    parser.add_argument("--note", default="Standalone, print-size, grayscale, and paper-context visual review completed.")
    args = parser.parse_args()
    result = (
        approve(args.root, reviewer=args.reviewer, note=args.note)
        if args.approve
        else render(args.root, args.inspect_stamp, all_figures=args.all)
    )
    print(json.dumps(result, sort_keys=True))
