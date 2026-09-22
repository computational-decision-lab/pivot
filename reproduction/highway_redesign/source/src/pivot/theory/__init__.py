"""Small, auditable numerical checks for the PIVOT theory claims."""

from .empirical import (
    evaluate_global_fidelity_case,
    evaluate_response_footprint_case,
    run_theory_experiment,
)
from .operator_shift import (
    chi_square_divergence,
    effective_sample_size,
    operator_shift_bound,
    operator_shift_summary,
)
from .sample_complexity import (
    best_update_error_bound,
    required_cluster_samples,
    required_subgaussian_samples,
)

__all__ = [
    "best_update_error_bound",
    "chi_square_divergence",
    "effective_sample_size",
    "evaluate_global_fidelity_case",
    "evaluate_response_footprint_case",
    "operator_shift_bound",
    "operator_shift_summary",
    "required_cluster_samples",
    "required_subgaussian_samples",
    "run_theory_experiment",
]
