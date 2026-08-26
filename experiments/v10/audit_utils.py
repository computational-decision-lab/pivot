"""Small shared helpers for deterministic V10 release audits."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(root: Path, relative: str, payload: Any) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_markdown(root: Path, filename: str, text: str) -> Path:
    target = root / filename
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)
