from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from pivot.evaluation.uncertainty import bootstrap_mean_ci


def paired_bootstrap_mean_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Return a bootstrap interval over within-pair ``left - right`` values."""

    if not left or not right:
        raise ValueError("paired samples must not be empty")
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    differences = [float(first) - float(second) for first, second in zip(left, right)]
    low, high = bootstrap_mean_ci(differences, seed=seed, n_bootstrap=n_bootstrap, alpha=alpha)
    return {"estimate": mean(differences), "ci_low": low, "ci_high": high, "n": len(differences)}


def _bootstrap(values: Sequence[float], *, seed: int) -> dict[str, float | int]:
    if not values:
        raise ValueError("values must not be empty")
    low, high = bootstrap_mean_ci(values, seed=seed)
    return {"estimate": mean(float(value) for value in values), "ci_low": low, "ci_high": high, "n": len(values)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _provenance(run: Path) -> dict[str, Any]:
    path = run / "provenance.json"
    payload: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded)
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded_manifest, dict):
            for key in ("run_id", "seeds", "config_sha256"):
                if key not in payload and key in loaded_manifest:
                    payload[key] = loaded_manifest[key]
    return payload


def summarize_p2_runs(
    run_dirs: Sequence[Path], *, min_response_strength: float = 0.3, min_footprint: float = 0.0
) -> dict[str, Any]:
    """Aggregate P2 at the independent-run level, preserving paired contrasts."""

    per_run: list[dict[str, Any]] = []
    config_hashes: list[str] = []
    seeds: list[int] = []
    for run_dir in run_dirs:
        rows = _jsonl(Path(run_dir) / "transitions.jsonl")
        provenance = _provenance(Path(run_dir))
        config_hash = provenance.get("config_sha256")
        if config_hash is not None:
            config_hashes.append(str(config_hash))
        seeds.extend(int(seed) for seed in provenance.get("seeds", []))
        eligible = [
            row
            for row in rows
            if float(row.get("delta_proxy", 0.0)) > 0
            and float(row.get("update_footprint", 0.0)) >= min_footprint
            and row.get("delta_true") is not None
        ]
        high = [row for row in eligible if float(row.get("response_strength", 0.0)) >= min_response_strength]
        low = [row for row in eligible if float(row.get("response_strength", 0.0)) < min_response_strength]
        per_run.append(
            {
                "run_id": str(provenance.get("run_id", Path(run_dir).name)),
                "n_rows": len(rows),
                "n_positive_proxy": len(eligible),
                "irr": _irr(eligible),
                "high_response_irr": _irr(high),
                "low_response_irr": _irr(low),
            }
        )
    valid = [row for row in per_run if row["irr"] is not None]
    by_run_id = {str(row["run_id"]): row for row in valid}
    high = [row for row in valid if row["high_response_irr"] is not None]
    contrasts = [
        float(row["high_response_irr"]) - float(row["low_response_irr"])
        for row in by_run_id.values()
        if row["high_response_irr"] is not None and row["low_response_irr"] is not None
    ]
    return {
        "run_count": len(run_dirs),
        "valid_run_count": len(valid),
        "run_ids": [row["run_id"] for row in per_run],
        "config_hashes": sorted(config_hashes),
        "seeds": sorted(set(seeds)),
        "per_run": per_run,
        "irr": _bootstrap([float(row["irr"]) for row in valid], seed=20260819) if valid else None,
        "high_response_irr": _bootstrap([float(row["high_response_irr"]) for row in high], seed=20260820) if high else None,
        "response_contrast": _bootstrap(contrasts, seed=20260821) if contrasts else None,
    }


def _irr(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["delta_true"]) < 0 for row in rows) / len(rows)


def summarize_e4_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    records = [_read_json(Path(run) / "comparison.json") for run in run_dirs]
    records = [record for record in records if record]
    metadata = [_provenance(Path(run)) for run in run_dirs]
    return {
        "run_count": len(run_dirs),
        "valid_run_count": len(records),
        "run_ids": [str(item.get("run_id", Path(run).name)) for run, item in zip(run_dirs, metadata)],
        "config_hashes": sorted(
            str(item["config_sha256"]) for item in metadata if item.get("config_sha256") is not None
        ),
        "seeds": sorted({int(seed) for item in metadata for seed in item.get("seeds", [])}),
        "policy_value_mae": _metric(records, "policy_value_mae", 20260830),
        "local_minus_global_isc": _paired_metric(records, "improvement_sign_consistency", "global_improvement_sign_consistency", 20260831),
        "local_minus_global_ide": _paired_metric(records, "improvement_differential_error", "global_improvement_differential_error", 20260832),
        "local_minus_global_isr": _paired_metric(records, "update_selection_regret", "global_update_selection_regret", 20260833),
    }


def summarize_e5_runs(run_dirs: Sequence[Path], *, target_budget: int) -> dict[str, Any]:
    by_run: list[dict[str, dict[str, dict[str, float]]]] = []
    for run in run_dirs:
        rows = _jsonl(Path(run) / "group_metrics.jsonl")
        grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in rows:
            if int(row["budget"]) == target_budget:
                method = str(row["method"])
                group = str(row["group_id"])
                grouped[method][group]["isr"].append(float(row["isr"]))
                grouped[method][group]["cti"].append(float(row["cti"]))
        method_means: dict[str, dict[str, dict[str, float]]] = {}
        for method, groups in grouped.items():
            method_means[method] = {
                group: {metric: mean(values) for metric, values in metrics.items()}
                for group, metrics in groups.items()
            }
        by_run.append(method_means)

    def paired(method_left: str, method_right: str, metric: str) -> dict[str, float | int] | None:
        run_differences: list[float] = []
        for methods in by_run:
            left = methods.get(method_left, {})
            right = methods.get(method_right, {})
            common = sorted(set(left) & set(right))
            if not common:
                continue
            if metric == "isr":
                run_differences.append(
                    mean(right[group][metric] - left[group][metric] for group in common)
                )
            else:
                run_differences.append(
                    mean(left[group][metric] - right[group][metric] for group in common)
                )
        return _bootstrap(run_differences, seed=20260840) if run_differences else None

    return {
        "run_count": len(run_dirs),
        "valid_run_count": sum(bool(methods) for methods in by_run),
        "run_ids": [str(_provenance(Path(run)).get("run_id", Path(run).name)) for run in run_dirs],
        "config_hashes": sorted(
            str(value)
            for run in run_dirs
            for value in [_provenance(Path(run)).get("config_sha256")]
            if value is not None
        ),
        "seeds": sorted(
            {
                int(seed)
                for run in run_dirs
                for seed in _provenance(Path(run)).get("seeds", [])
            }
        ),
        "target_budget": target_budget,
        "random_minus_pivot_isr": paired("pivot", "random_hf", "isr"),
        "top_proxy_minus_pivot_isr": paired("pivot", "top_proxy_hf", "isr"),
        "pivot_minus_random_cti": paired("pivot", "random_hf", "cti"),
        "pivot_minus_top_proxy_cti": paired("pivot", "top_proxy_hf", "cti"),
    }


def summarize_ablation_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Aggregate the twelve-ablation summaries over independent runs."""

    payloads = [_read_json(Path(run) / "ablation_summary.json") for run in run_dirs]
    valid = [payload for payload in payloads if payload.get("ablation_count") == 12]
    by_ablation: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for payload in valid:
        for ablation_id, ablation in payload.get("ablations", {}).items():
            for variant, metrics in ablation.get("variants", {}).items():
                for metric, summary in metrics.items():
                    if isinstance(summary, Mapping) and summary.get("estimate") is not None:
                        by_ablation[ablation_id][variant][metric].append(float(summary["estimate"]))
    aggregate: dict[str, Any] = {}
    for ablation_id, variants in sorted(by_ablation.items()):
        aggregate[ablation_id] = {
            variant: {
                metric: _bootstrap(values, seed=20260900 + index)
                for index, (metric, values) in enumerate(sorted(metrics.items()))
            }
            for variant, metrics in sorted(variants.items())
        }
    return {
        "run_count": len(run_dirs),
        "valid_run_count": len(valid),
        "ablation_count": len(aggregate),
        "run_ids": [str(_provenance(Path(run)).get("run_id", Path(run).name)) for run in run_dirs],
        "config_hashes": sorted(
            str(value)
            for run in run_dirs
            for value in [_provenance(Path(run)).get("config_sha256")]
            if value is not None
        ),
        "ablations": aggregate,
    }


def summarize_e6_runs(
    run_dirs: Sequence[Path], *, target_participation: float = 0.05, tolerance: float = 1e-12
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in run_dirs:
        rows = _json_rows(Path(run) / "finance_actor.json")
        zero = [
            float(row["delta_f2"]) - float(row["delta_f1"])
            for row in rows
            if abs(float(row["participation_rate"])) <= tolerance
        ]
        target = [
            float(row["delta_f2"]) - float(row["delta_f1"])
            for row in rows
            if abs(float(row["participation_rate"]) - target_participation) <= tolerance
        ]
        per_run.append(
            {
                "run_id": str(_provenance(Path(run)).get("run_id", Path(run).name)),
                "zero": mean(zero) if zero else None,
                "target": mean(target) if target else None,
                "n_rows": len(rows),
            }
        )
    zero = [float(row["zero"]) for row in per_run if row["zero"] is not None]
    target = [float(row["target"]) for row in per_run if row["target"] is not None]
    return {
        "run_count": len(run_dirs),
        "valid_run_count": sum(row["target"] is not None for row in per_run),
        "run_ids": [row["run_id"] for row in per_run],
        "zero_participation_f2_minus_f1": _bootstrap(zero, seed=20260850) if zero else None,
        "target_f2_minus_f1": _bootstrap(target, seed=20260851) if target else None,
        "target_participation": target_participation,
        "per_run": per_run,
    }


def summarize_e7_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in run_dirs:
        rows = _json_rows(Path(run) / "strategic_reversal.json")
        positive_actor = [row for row in rows if float(row.get("delta_actor", 0.0)) > 0]
        reversals = [float(row.get("delta_strategic", 0.0)) < 0 for row in positive_actor]
        effects = [
            float(row["competition_effect"])
            for row in rows
            if row.get("competition_effect") is not None
        ]
        per_run.append(
            {
                "run_id": str(_provenance(Path(run)).get("run_id", Path(run).name)),
                "sirr": sum(reversals) / len(reversals) if reversals else None,
                "competition_effect": mean(effects) if effects else None,
                "n_rows": len(rows),
            }
        )
    sirr = [float(row["sirr"]) for row in per_run if row["sirr"] is not None]
    effects = [
        float(row["competition_effect"])
        for row in per_run
        if row["competition_effect"] is not None
    ]
    return {
        "run_count": len(run_dirs),
        "valid_run_count": sum(row["sirr"] is not None for row in per_run),
        "run_ids": [row["run_id"] for row in per_run],
        "sirr": _bootstrap(sirr, seed=20260860) if sirr else None,
        "competition_effect": _bootstrap(effects, seed=20260861) if effects else None,
        "per_run": per_run,
    }


def summarize_e8_runs(run_dirs: Sequence[Path], *, mode: str = "adaptive") -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in run_dirs:
        rows = [
            row
            for row in _json_rows(Path(run) / "competition.json")
            if str(row.get("mode")) == mode
        ]
        positive_actor = [row for row in rows if float(row.get("delta_actor", 0.0)) > 0]
        effects = [
            float(row["competition_effect"])
            for row in rows
            if row.get("competition_effect") is not None
        ]
        sensitivities = sorted(
            {
                float(row["strategic_sensitivity"])
                for row in rows
                if row.get("strategic_sensitivity") is not None
            }
        )
        sensitivity_contrast = None
        if len(sensitivities) >= 2:
            low = sensitivities[0]
            high = sensitivities[-1]
            low_effects = [
                float(row["competition_effect"])
                for row in rows
                if float(row.get("strategic_sensitivity", low)) == low
            ]
            high_effects = [
                float(row["competition_effect"])
                for row in rows
                if float(row.get("strategic_sensitivity", high)) == high
            ]
            if low_effects and high_effects:
                sensitivity_contrast = mean(high_effects) - mean(low_effects)
        reversals = [float(row.get("delta_strategic", 0.0)) < 0 for row in positive_actor]
        per_run.append(
            {
                "run_id": str(_provenance(Path(run)).get("run_id", Path(run).name)),
                "competition_effect": mean(effects) if effects else None,
                "sirr": sum(reversals) / len(reversals) if reversals else None,
                "sensitivity_contrast": sensitivity_contrast,
                "n_rows": len(rows),
            }
        )
    effects = [
        float(row["competition_effect"])
        for row in per_run
        if row["competition_effect"] is not None
    ]
    sirr = [float(row["sirr"]) for row in per_run if row["sirr"] is not None]
    contrasts = [
        float(row["sensitivity_contrast"])
        for row in per_run
        if row["sensitivity_contrast"] is not None
    ]
    return {
        "run_count": len(run_dirs),
        "valid_run_count": sum(row["competition_effect"] is not None for row in per_run),
        "mode": mode,
        "run_ids": [row["run_id"] for row in per_run],
        "competition_effect": _bootstrap(effects, seed=20260870) if effects else None,
        "sirr": _bootstrap(sirr, seed=20260871) if sirr else None,
        "sensitivity_contrast": _bootstrap(contrasts, seed=20260872) if contrasts else None,
        "per_run": per_run,
    }


def summarize_e9_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in run_dirs:
        rows = _json_rows(Path(run) / "closed_loop.json")
        unknown_selected = sum(
            row.get("selected_delta_true") is None
            for row in rows
        )
        per_run.append(
            {
                "run_id": str(_provenance(Path(run)).get("run_id", Path(run).name)),
                "rounds": len(rows),
                "hf_queries": sum(int(row.get("hf_budget", 0)) for row in rows),
                "unknown_selected": unknown_selected,
                "cti": sum(
                    float(row["selected_delta_true"])
                    for row in rows
                    if row.get("selected_delta_true") is not None
                ),
            }
        )
    return {
        "run_count": len(run_dirs),
        "valid_run_count": sum(row["rounds"] > 0 for row in per_run),
        "run_ids": [row["run_id"] for row in per_run],
        "mean_rounds": _bootstrap([float(row["rounds"]) for row in per_run], seed=20260880) if per_run else None,
        "mean_hf_queries": _bootstrap([float(row["hf_queries"]) for row in per_run], seed=20260881) if per_run else None,
        "mean_cti": _bootstrap([float(row["cti"]) for row in per_run], seed=20260882) if per_run else None,
        "mean_unknown_selected": _bootstrap(
            [float(row["unknown_selected"]) for row in per_run], seed=20260883
        )
        if per_run
        else None,
        "per_run": per_run,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric(records: Sequence[Mapping[str, Any]], key: str, seed: int) -> dict[str, float | int] | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return _bootstrap(values, seed=seed) if values else None


def _paired_metric(
    records: Sequence[Mapping[str, Any]], left_key: str, right_key: str, seed: int
) -> dict[str, float | int] | None:
    left = [float(record[left_key]) for record in records if record.get(left_key) is not None and record.get(right_key) is not None]
    right = [float(record[right_key]) for record in records if record.get(left_key) is not None and record.get(right_key) is not None]
    return paired_bootstrap_mean_ci(left, right, seed=seed) if left else None


def evaluate_gate_a_b(summary: Mapping[str, Any], *, min_runs: int = 3) -> dict[str, Any]:
    enough = int(summary.get("valid_run_count", 0)) >= min_runs
    irr = summary.get("high_response_irr")
    contrast = summary.get("response_contrast")
    return {
        "A": _gate_status(enough and irr is not None and float(irr["ci_low"]) > 0.0),
        "B": _gate_status(enough and contrast is not None and float(contrast["ci_low"]) > 0.0),
        "criteria": {"min_runs": min_runs, "high_response": True, "ci_level": 0.95},
    }


def evaluate_gate_c(summary: Mapping[str, Any], *, min_runs: int = 3) -> dict[str, Any]:
    metric = summary.get("local_minus_global_isc")
    enough = int(summary.get("valid_run_count", 0)) >= min_runs
    return {"C": _gate_status(enough and metric is not None and float(metric["ci_low"]) > 0.0), "metric": metric}


def evaluate_gate_d(summary: Mapping[str, Any], *, min_runs: int = 3) -> dict[str, Any]:
    random = summary.get("random_minus_pivot_isr")
    top = summary.get("top_proxy_minus_pivot_isr")
    enough = int(summary.get("valid_run_count", 0)) >= min_runs
    passed = (
        enough
        and random is not None
        and top is not None
        and float(random["ci_low"]) > 0.0
        and float(top["ci_low"]) > 0.0
    )
    return {"D": _gate_status(passed), "random_minus_pivot_isr": random, "top_proxy_minus_pivot_isr": top}


def evaluate_gate_e(
    summary: Mapping[str, Any], *, min_runs: int = 3, tolerance: float = 1e-12
) -> dict[str, Any]:
    zero = summary.get("zero_participation_f2_minus_f1")
    target = summary.get("target_f2_minus_f1")
    enough = int(summary.get("valid_run_count", 0)) >= min_runs
    zero_ok = (
        zero is not None
        and abs(float(zero["ci_low"])) <= tolerance
        and abs(float(zero["ci_high"])) <= tolerance
    )
    target_differs = target is not None and (
        float(target["ci_high"]) < -tolerance or float(target["ci_low"]) > tolerance
    )
    return {
        "E": _gate_status(enough and zero_ok and target_differs),
        "zero_equivalence": zero,
        "target_effect": target,
    }


def evaluate_gate_f(
    e7_summary: Mapping[str, Any], e8_summary: Mapping[str, Any], *, min_runs: int = 3
) -> dict[str, Any]:
    e7_sirr = e7_summary.get("sirr")
    e7_effect = e7_summary.get("competition_effect")
    e8_sirr = e8_summary.get("sirr")
    e8_effect = e8_summary.get("competition_effect")
    e8_contrast = e8_summary.get("sensitivity_contrast")
    enough = min(
        int(e7_summary.get("valid_run_count", 0)),
        int(e8_summary.get("valid_run_count", 0)),
    ) >= min_runs
    passed = (
        enough
        and e7_sirr is not None
        and e7_effect is not None
        and e8_effect is not None
        and e8_contrast is not None
        and float(e7_sirr["ci_low"]) > 0.0
        and float(e7_effect["ci_high"]) < 0.0
        and float(e8_effect["ci_high"]) < 0.0
        and float(e8_contrast["ci_high"]) < 0.0
    )
    return {
        "F": _gate_status(passed),
        "e7_sirr": e7_sirr,
        "e7_competition_effect": e7_effect,
        "e8_sirr": e8_sirr,
        "e8_competition_effect": e8_effect,
        "e8_sensitivity_contrast": e8_contrast,
    }


def _gate_status(passed: bool) -> str:
    return "Pass" if passed else "Not run"
