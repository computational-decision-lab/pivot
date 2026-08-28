"""Shared helpers for deterministic V15 audit entry points."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def write_audit(root: Path, name: str, payload: dict[str, Any], title: str, summary: str) -> dict[str, Any]:
    """Write machine-readable and concise human-readable audit records."""

    root = Path(root).resolve()
    output = root / "artifacts/v15"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# {title}", "", summary, "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""]
    (root / f"V15_{name.upper()}.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def cli(main_fn: Callable[[Path], dict[str, Any]], description: str) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(main_fn(args.root.resolve()), sort_keys=True))
