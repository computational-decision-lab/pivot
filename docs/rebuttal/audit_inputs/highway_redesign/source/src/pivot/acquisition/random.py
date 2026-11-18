from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .common import candidate_id, validate_budget


def select_random(candidates: Sequence[Any], budget: int, seed: int = 0) -> list[str]:
    validate_budget(candidates, budget)
    ids = [candidate_id(candidate) for candidate in candidates]
    generator = random.Random(seed)
    generator.shuffle(ids)
    return ids[:budget]
