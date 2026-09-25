"""Empirical checks for the V6 global-fidelity and response-footprint claims.

The constructions in this module are deliberately small and transparent.  GFB
uses one adjacent rank swap in a finite policy family; RFS uses a scalar
Lipschitz response map whose constants are known exactly.  They are theorem
checks, not claims about a real market or a learned simulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pivot.metrics.improvement import compute_improvement_fidelity, compute_improvement_metrics
from pivot.transfer.global_value import spearman_rank_correlation


def evaluate_global_fidelity_case(
    n_policies: int,
    *,
    operator_samples: int = 64,
    seed: int = 0,
    epsilon: float | None = None,
) -> dict[str, Any]:
    """Evaluate the adjacent-swap construction for one policy-family size.

    True values are equally spaced on ``[0, 1]``.  The proxy swaps one
    adjacent pair.  The improvement operator repeatedly proposes the lower
    true-valued member as a replacement for the higher one, so every sampled
    operator transition is reversed even though the proxy error is concentrated
    on only two policies.
    """

    if n_policies < 4:
        raise ValueError("n_policies must be at least four")
    if operator_samples <= 0:
        raise ValueError("operator_samples must be positive")
    if epsilon is not None and (epsilon <= 0 or not math.isfinite(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    rng = np.random.default_rng(seed)
    true_values = np.linspace(0.0, 1.0, n_policies, dtype=np.float64)
    proxy_values = true_values.copy()
    swap_left = n_policies // 2 - 1
    proxy_values[swap_left], proxy_values[swap_left + 1] = (
        proxy_values[swap_left + 1],
        proxy_values[swap_left],
    )

    operator_rows: list[dict[str, Any]] = []
    for draw in range(operator_samples):
        draw_seed = int(rng.integers(0, 2**31 - 1))
        delta_proxy = float(proxy_values[swap_left] - proxy_values[swap_left + 1])
        delta_true = float(true_values[swap_left] - true_values[swap_left + 1])
        operator_rows.append(
            {
                "transition_id": f"gfb-n{n_policies}-draw{draw}",
                "round_id": 0,
                "seed": draw_seed,
                "delta_proxy": delta_proxy,
                "delta_true": delta_true,
                "selected": True,
            }
        )
    operator_metrics = compute_improvement_metrics(operator_rows)
    improvement_fidelity_ide = compute_improvement_fidelity(
        operator_rows, loss="absolute_delta_error"
    )
    improvement_fidelity_sign_error = compute_improvement_fidelity(
        operator_rows, loss="sign_error"
    )
    global_mae = float(np.mean(np.abs(proxy_values - true_values)))
    spearman = spearman_rank_correlation(proxy_values.tolist(), true_values.tolist())
    result: dict[str, Any] = {
        "policy_count": n_policies,
        "operator_sample_count": operator_samples,
        "seed": seed,
        "swapped_policy_indices": [swap_left, swap_left + 1],
        "q_a_name": "adjacent-swap-focused",
        "q_a_support_size": 1,
        "q_a_support_probability": 1.0,
        "q_a_incumbent_policy_index": swap_left + 1,
        "q_a_candidate_policy_index": swap_left,
        "global_mae": global_mae,
        "spearman": float(spearman),
        "spearman_deficit": float(1.0 - spearman),
        "operator_ide": _metric_float(operator_metrics, "ide"),
        "operator_improvement_sign_consistency": _metric_float(operator_metrics, "isc"),
        "operator_improvement_reversal_rate": _metric_float(operator_metrics, "irr"),
        "operator_positive_proxy_count": _metric_int(operator_metrics, "n_positive_proxy"),
        "operator_reversal_count": _metric_int(operator_metrics, "n_reversals"),
        "improvement_fidelity_ide": improvement_fidelity_ide,
        "improvement_fidelity_sign_error": improvement_fidelity_sign_error,
    }
    # Short aliases make the row directly comparable with the paper's metric
    # vocabulary while retaining the operator-conditioned names in artifacts.
    result["improvement_sign_consistency"] = result["operator_improvement_sign_consistency"]
    result["improvement_reversal_rate"] = result["operator_improvement_reversal_rate"]
    if epsilon is not None:
        result.update(
            {
                "epsilon": epsilon,
                "global_mae_pass": global_mae < epsilon,
                "spearman_pass": spearman > 1.0 - epsilon,
                "operator_reversal_pass": result["operator_improvement_reversal_rate"] == 1.0,
                "operator_sign_pass": result["operator_improvement_sign_consistency"] == 0.0,
            }
        )
        result["theorem_pass"] = all(
            bool(result[key])
            for key in (
                "global_mae_pass",
                "spearman_pass",
                "operator_reversal_pass",
                "operator_sign_pass",
            )
        )
    return result


def evaluate_response_footprint_case(
    response_strength: float,
    update_footprint: float,
    *,
    value_lipschitz: float = 1.0,
    seed: int = 0,
    incumbent_policy: float = 0.2,
) -> dict[str, Any]:
    """Evaluate the exact scalar response-footprint construction.

    ``M_lambda(x)=lambda*x`` has response Lipschitz constant ``lambda`` and
    ``J(x,m)=x-L_J*m`` is ``L_J``-Lipschitz in ``m``.  Consequently the
    response contribution reaches the proposed bound exactly for every
    positive footprint.
    """

    _validate_nonnegative_finite(response_strength, "response_strength")
    _validate_nonnegative_finite(update_footprint, "update_footprint")
    _validate_nonnegative_finite(value_lipschitz, "value_lipschitz")
    if update_footprint <= 0:
        raise ValueError("update_footprint must be positive")
    if incumbent_policy < 0 or not math.isfinite(incumbent_policy):
        raise ValueError("incumbent_policy must be finite and non-negative")

    candidate_policy = incumbent_policy + update_footprint
    incumbent_response = response_strength * incumbent_policy
    candidate_response = response_strength * candidate_policy
    direct_delta = candidate_policy - incumbent_policy
    actor_delta = (
        candidate_policy - value_lipschitz * candidate_response
        - (incumbent_policy - value_lipschitz * incumbent_response)
    )
    response_distance = abs(candidate_response - incumbent_response)
    observed_error = abs(actor_delta - direct_delta)
    bound = value_lipschitz * response_distance
    bound_slack = bound - observed_error
    bound_ratio = observed_error / bound if bound > 0 else 0.0
    return {
        "response_strength": response_strength,
        "update_footprint": update_footprint,
        "value_lipschitz": value_lipschitz,
        "seed": seed,
        "incumbent_policy": incumbent_policy,
        "candidate_policy": candidate_policy,
        "response_distance": response_distance,
        "delta_direct": direct_delta,
        "delta_actor": actor_delta,
        "observed_error": observed_error,
        "bound": bound,
        "bound_slack": bound_slack,
        "bound_ratio": bound_ratio,
        "bound_holds": bool(bound_slack >= -1e-12),
    }


def run_theory_experiment(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run both V6 checks and write hash-indexed, reproducible artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty theory run: {output}")
    policy_counts = _positive_int_grid(config.get("policy_counts", [16, 32, 64, 128, 256, 512, 1024]), "policy_counts")
    epsilons = _positive_float_grid(config.get("epsilons", [0.1, 0.01, 0.001]), "epsilons")
    response_strengths = _nonnegative_float_grid(config.get("response_strengths", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]), "response_strengths")
    footprints = _positive_float_grid(config.get("footprints", [0.01, 0.025, 0.05, 0.1, 0.2, 0.4]), "footprints")
    seeds = _positive_int_grid(config.get("seeds", [1, 2, 3, 4, 5]), "seeds", allow_zero=True)
    operator_samples = int(config.get("operator_samples", 128))
    if operator_samples <= 0:
        raise ValueError("operator_samples must be positive")
    value_lipschitz = float(config.get("value_lipschitz", 1.0))
    _validate_nonnegative_finite(value_lipschitz, "value_lipschitz")

    global_rows = [
        evaluate_global_fidelity_case(
            n_policies,
            operator_samples=operator_samples,
            seed=seed,
            epsilon=epsilon,
        )
        for n_policies in policy_counts
        for epsilon in epsilons
        for seed in seeds
    ]
    response_rows = [
        evaluate_response_footprint_case(
            response,
            footprint,
            value_lipschitz=value_lipschitz,
            seed=seed,
        )
        for response in response_strengths
        for footprint in footprints
        for seed in seeds
    ]
    metrics: dict[str, Any] = {
        "global_fidelity": _global_summary(global_rows),
        "response_footprint": _response_summary(response_rows),
    }
    metrics["global_fidelity_pass"] = bool(metrics["global_fidelity"]["all_theorem_gates_pass"])
    metrics["response_footprint_bound_pass"] = bool(
        metrics["response_footprint"]["all_bound_holds"]
    )
    config_snapshot = {
        "policy_counts": policy_counts,
        "epsilons": epsilons,
        "response_strengths": response_strengths,
        "footprints": footprints,
        "seeds": seeds,
        "operator_samples": operator_samples,
        "value_lipschitz": value_lipschitz,
        "dataset_version": "v6-theory-constructions-1",
        "q_a_definition": "empirical law induced by adjacent-swap-focused operator",
        "improvement_fidelity_losses": ["absolute_delta_error", "sign_error"],
    }
    _write_jsonl(output / "global_fidelity_rows.jsonl", global_rows)
    _write_jsonl(output / "response_footprint_rows.jsonl", response_rows)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "config_snapshot.json", config_snapshot)
    _write_json(output / "provenance.json", _provenance(output, config_snapshot))
    _write_summary_csv(output / "summary.csv", global_rows, response_rows)
    _write_figures(output, global_rows, response_rows)
    _write_readme(output)
    _write_manifest(output)
    return metrics


def _global_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("global fidelity rows must not be empty")
    epsilon_targets = sorted({float(row["epsilon"]) for row in rows})
    target_results: dict[str, Any] = {}
    smallest_passing_count: dict[str, int | None] = {}
    for epsilon in epsilon_targets:
        epsilon_rows = [row for row in rows if float(row["epsilon"]) == epsilon]
        policy_counts = sorted({int(row["policy_count"]) for row in epsilon_rows})
        passing_counts = [
            count
            for count in policy_counts
            if all(
                bool(row["theorem_pass"])
                for row in epsilon_rows
                if int(row["policy_count"]) == count
            )
        ]
        target_results[str(epsilon)] = bool(passing_counts)
        smallest_passing_count[str(epsilon)] = min(passing_counts, default=None)
    all_targets_covered = all(target_results.values())
    return {
        "n_rows": len(rows),
        "epsilon_targets": epsilon_targets,
        "epsilon_target_results": target_results,
        "smallest_passing_policy_count": smallest_passing_count,
        "all_epsilon_targets_covered": all_targets_covered,
        "all_theorem_gates_pass": all_targets_covered,
        "min_global_mae": min(float(row["global_mae"]) for row in rows),
        "max_global_mae": max(float(row["global_mae"]) for row in rows),
        "min_spearman": min(float(row["spearman"]) for row in rows),
        "max_spearman_deficit": max(float(row["spearman_deficit"]) for row in rows),
        "min_operator_irr": min(float(row["operator_improvement_reversal_rate"]) for row in rows),
        "max_operator_ide": max(float(row["operator_ide"]) for row in rows),
        "max_improvement_fidelity_ide": max(
            float(row["improvement_fidelity_ide"]) for row in rows
        ),
        "min_improvement_fidelity_sign_error": min(
            float(row["improvement_fidelity_sign_error"]) for row in rows
        ),
        "max_improvement_fidelity_sign_error": max(
            float(row["improvement_fidelity_sign_error"]) for row in rows
        ),
        "q_a_name": str(rows[0]["q_a_name"]),
    }


def _metric_float(metrics: Mapping[str, float | int | None], key: str) -> float:
    value = metrics.get(key)
    return 0.0 if value is None else float(value)


def _metric_int(metrics: Mapping[str, float | int | None], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"metric {key} is missing")
    return int(value)


def _response_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("response-footprint rows must not be empty")
    errors = [float(row["observed_error"]) for row in rows]
    bounds = [float(row["bound"]) for row in rows]
    ratios = [float(row["bound_ratio"]) for row in rows if float(row["bound"]) > 0]
    return {
        "n_rows": len(rows),
        "all_bound_holds": all(bool(row["bound_holds"]) for row in rows),
        "max_bound_violation": max(float(row["observed_error"]) - float(row["bound"]) for row in rows),
        "max_bound_ratio": max(ratios, default=0.0),
        "min_bound_ratio": min(ratios, default=0.0),
        "mean_observed_error": float(np.mean(errors)),
        "mean_bound": float(np.mean(bounds)),
    }


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and non-negative")


def _positive_int_grid(value: object, name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    values = tuple(int(item) for item in value)
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _positive_float_grid(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    values = tuple(float(item) for item in value)
    if not values or any(item <= 0 or not math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite positive values")
    return values


def _nonnegative_float_grid(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    values = tuple(float(item) for item in value)
    if not values or any(item < 0 or not math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _write_summary_csv(
    path: Path,
    global_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = [{"experiment": "global_fidelity", **dict(row)} for row in global_rows]
    rows.extend({"experiment": "response_footprint", **dict(row)} for row in response_rows)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _provenance(output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    repository_root = _repository_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root or output,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "run_id": output.name,
        "git_commit": commit,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "paired": False,
        "claim_boundary": "analytic constructions; not causal market evidence",
    }


def _repository_root() -> Path | None:
    """Find the checkout containing this module without recording its path."""

    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(root) if root else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(output: Path) -> None:
    files = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "v6-theory-empirical",
            "files": files,
            "file_count": len(files),
        },
    )


def _write_readme(output: Path) -> None:
    """Explain the registered construction without leaking local paths."""

    (output / "README.md").write_text(
        """# E10 V6 theory empirical check

This registered artifact checks two analytic claims used by PIVOT:

1. **Global Fidelity Blindness (GFB):** an adjacent rank swap in a growing
   finite policy family makes global MAE and rank deficit vanish while the
   operator-conditioned update reversal rate remains one. The sampled rows
   define `Q_A`; `improvement_fidelity_ide` and
   `improvement_fidelity_sign_error` are its absolute-delta and sign-error
   losses.
2. **Response-Footprint Sensitivity (RFS):** the scalar response map and
   Lipschitz value functional attain the bound
   `|Delta_actor - Delta_direct| <= L_J L_M d` exactly.

The rows and figures are generated by
`experiments/e10_theory_empirical.py` with
`configs/theory/v6_empirical.yaml`. These are transparent theorem checks, not
causal evidence about a live market or a learned world model.
""",
        encoding="utf-8",
    )


def _write_figures(
    output: Path,
    global_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt

        from scripts.figure_style import PALETTE, apply_publication_style, finalize_figure
    except ImportError as error:
        raise RuntimeError("matplotlib and the local figure-style adapter are required") from error
    apply_publication_style()
    grouped: dict[int, Mapping[str, Any]] = {}
    for row in global_rows:
        grouped[int(row["policy_count"])] = row
    counts = sorted(grouped)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    axes[0].loglog(counts, [float(grouped[n]["global_mae"]) for n in counts], "o-", color=PALETTE["blue_main"], label="global MAE")
    axes[0].loglog(counts, [float(grouped[n]["operator_ide"]) for n in counts], "s--", color=PALETTE["red_strong"], label="operator IDE")
    axes[0].set(xlabel="Policy-family size", ylabel="Error", title="Global error shrinks; local error persists")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogx(counts, [float(grouped[n]["spearman_deficit"]) for n in counts], "o-", color=PALETTE["teal"], label="1 − Spearman")
    axes[1].semilogx(counts, [1.0 - float(grouped[n]["operator_improvement_sign_consistency"]) for n in counts], "s--", color=PALETTE["red_strong"], label="1 − ISC")
    axes[1].set(xlabel="Policy-family size", ylabel="Deficit", title="Operator-conditioned blindness")
    axes[1].legend(frameon=False, fontsize=8)
    finalize_figure(figure, output / "global_fidelity_blindness.png", dpi=300)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    observed = np.asarray([float(row["observed_error"]) for row in response_rows])
    bounds = np.asarray([float(row["bound"]) for row in response_rows])
    axes[0].scatter(bounds, observed, s=18, color=PALETTE["blue_main"], alpha=0.75)
    limit = max(float(bounds.max()), float(observed.max()), 1e-12)
    axes[0].plot([0.0, limit], [0.0, limit], color=PALETTE["red_strong"], linewidth=1.2)
    axes[0].set(xlabel=r"$L_J L_M d$", ylabel=r"$|\Delta_{actor}-\Delta_{direct}|$", title="Response-footprint bound")
    ratios = [float(row["bound_ratio"]) for row in response_rows if float(row["bound"]) > 0]
    if max(ratios, default=0.0) - min(ratios, default=0.0) < 1e-12:
        value = ratios[0] if ratios else 0.0
        axes[1].bar([value], [len(ratios)], width=0.04, color=PALETTE["teal"], edgecolor="white")
    else:
        axes[1].hist(ratios, bins=8, color=PALETTE["teal"], edgecolor="white")
    axes[1].axvline(1.0, color=PALETTE["red_strong"], linewidth=1.2)
    axes[1].set(xlabel="Observed error / bound", ylabel="Count", title="Tightness")
    finalize_figure(figure, output / "response_footprint_bound.png", dpi=300)
