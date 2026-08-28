from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iterate(
    root: Path, note: str = "no defect recorded", *, all_figures: bool = False
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / "figures/v15/iteration_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": "PAPER_CONTEXT_PASS",
        "action": "review_only",
        "note": note,
        "scientific_data_changed": False,
        "selection": "all_manifest_figures" if all_figures else "manifest_default",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a figure review iteration without changing data")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--note", default="no defect recorded")
    parser.add_argument("--all", action="store_true", help="record review for every manifest figure")
    args = parser.parse_args()
    print(json.dumps(iterate(args.root, args.note, all_figures=args.all), sort_keys=True))
