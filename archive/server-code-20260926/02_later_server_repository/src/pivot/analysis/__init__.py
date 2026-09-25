"""Analysis helpers for registered PIVOT evidence runs."""

from .public_finance import run_public_finance_calibration
from .public_finance_expansion import aggregate_public_rows, run_public_finance_expansion
from .registered import (
    evaluate_gate_a_b,
    evaluate_gate_c,
    evaluate_gate_d,
    evaluate_gate_e,
    evaluate_gate_f,
    paired_bootstrap_mean_ci,
    summarize_ablation_runs,
    summarize_e4_runs,
    summarize_e5_runs,
    summarize_e6_runs,
    summarize_e7_runs,
    summarize_e8_runs,
    summarize_e9_runs,
    summarize_p2_runs,
)
from .registry import load_registry, materialize_seed_config, run_registered

__all__ = [
    "aggregate_public_rows",
    "evaluate_gate_a_b",
    "evaluate_gate_c",
    "evaluate_gate_d",
    "evaluate_gate_e",
    "evaluate_gate_f",
    "load_registry",
    "materialize_seed_config",
    "paired_bootstrap_mean_ci",
    "run_public_finance_calibration",
    "run_public_finance_expansion",
    "run_registered",
    "summarize_ablation_runs",
    "summarize_e4_runs",
    "summarize_e5_runs",
    "summarize_e6_runs",
    "summarize_e7_runs",
    "summarize_e8_runs",
    "summarize_e9_runs",
    "summarize_p2_runs",
]
