"""Development-only command namespace.

The exports are lazy so ``python -m experiments.v15.dev.smoke`` does not load
the target module twice and emit a misleading runpy warning.
"""

from pathlib import Path
from typing import Any

from ..planes import SealedDataPlanes


def default_planes(manifest_path: Path | None = None) -> SealedDataPlanes:
    from .smoke import default_planes as implementation

    return implementation(manifest_path)


def run_smoke(output: Path, *, seed: int = 10001, candidates_per_operator: int = 2) -> dict[str, Any]:
    from .smoke import run_smoke as implementation

    return implementation(output, seed=seed, candidates_per_operator=candidates_per_operator)


__all__ = ["default_planes", "run_smoke"]
