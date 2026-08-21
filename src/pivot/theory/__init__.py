"""Small, auditable numerical checks for the PIVOT theory claims."""

from .empirical import (
    evaluate_global_fidelity_case,
    evaluate_response_footprint_case,
    run_theory_experiment,
)

__all__ = [
    "evaluate_global_fidelity_case",
    "evaluate_response_footprint_case",
    "run_theory_experiment",
]
