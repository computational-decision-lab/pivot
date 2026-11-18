from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecomposedEffects:
    direct: float | None
    actor: float | None
    strategic: float | None
    mechanical_effect: float | None
    competition_effect: float | None


def decompose_effects(
    direct: float | None, actor: float | None, strategic: float | None
) -> DecomposedEffects:
    mechanical = None if direct is None or actor is None else actor - direct
    competition = None if actor is None or strategic is None else strategic - actor
    return DecomposedEffects(direct, actor, strategic, mechanical, competition)
