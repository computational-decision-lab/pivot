"""IMPROVE-X platform facade for self-improvement fidelity research."""

__version__ = "0.1.0"

from .acquisition import score_decision_preservation, select_pivot_x
from .core.operator import CandidateBatch, ImprovementOperator
from .core.trajectory import ImprovementTrajectory
from .operators import EvolutionaryMutation

__all__ = [
    "CandidateBatch",
    "EvolutionaryMutation",
    "ImprovementOperator",
    "ImprovementTrajectory",
    "score_decision_preservation",
    "select_pivot_x",
]
