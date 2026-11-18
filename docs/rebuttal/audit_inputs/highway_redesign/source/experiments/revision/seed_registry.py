"""Durable collision rejection for a coordinator's 32-bit rollout seed domain.

All methods/workers of a protocol must register at the same coordinator root,
or use a prevalidated, immutable exported plan. Separate local databases do not
provide global protection. Uniqueness is not a proof of statistical independence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SeedCollisionError(RuntimeError):
    """A different random context claimed an already registered rollout seed."""


class SeedRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)

    def register(self, records: list[tuple[int, dict[str, Any]]]) -> None:
        prepared = []
        for seed, identity in records:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("seed must be an unsigned 32-bit integer")
            prepared.append(
                (seed, json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False))
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS seeds (seed INTEGER PRIMARY KEY, identity TEXT NOT NULL)"
            )
            for seed, identity in prepared:
                row = connection.execute(
                    "SELECT identity FROM seeds WHERE seed = ?", (seed,)
                ).fetchone()
                if row is not None and row[0] != identity:
                    raise SeedCollisionError(
                        f"32-bit seed collision at {seed}; reject before paid evaluation; "
                        "use a new preregistered seed namespace"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO seeds (seed, identity) VALUES (?, ?)", (seed, identity)
                )
