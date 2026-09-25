"""Stable public facade for the agent-agnostic PIVOT core.

The research implementation remains organized under :mod:`pivot` so the
historical experiments keep their import paths.  This facade gives adapters
and downstream users a small, stable surface for policy transitions, paired
evaluation, footprint extraction, and PIVOT-VOI acquisition.
"""

from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    expected_simple_regret,
    score_pivot_voi,
    select_pivot_voi,
    should_stop,
)
from pivot.algorithms.pivot import RoundResult, run_pivot_round, run_pivot_voi_round
from pivot.core.policy import Policy
from pivot.core.result import PairedEvaluation, RolloutContext, RolloutResult
from pivot.core.transition import PolicyTransition
from pivot.evaluation.paired import PairedEvaluator
from pivot.footprint.generic import Footprint, compute_update_footprint

__all__ = [
    "BayesianLinearDeltaPosterior",
    "Footprint",
    "PairedEvaluation",
    "PairedEvaluator",
    "Policy",
    "PolicyTransition",
    "RolloutContext",
    "RolloutResult",
    "RoundResult",
    "compute_update_footprint",
    "expected_simple_regret",
    "run_pivot_round",
    "run_pivot_voi_round",
    "score_pivot_voi",
    "select_pivot_voi",
    "should_stop",
]
