"""Analyze whether registered pre-gate footprint features track transition risk.

The analysis is deliberately descriptive until a confirmatory transition archive
has been opened.  It treats ``run_id`` as the independent trajectory unit and
uses a cluster bootstrap so the many candidate rows generated within one
trajectory do not masquerade as independent replicates.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .audit_support import cli, write_audit
from .canonical import FOOTPRINT_COLUMNS
from .protocol import file_hash


def _float(value: object) -> float | None:
    if value is None or value == "" or str(value).casefold() in {"none", "null", "nan"}:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on", "y"}:
        return True
    if text in {"false", "0", "no", "off", "n", ""}:
        return False
    return default


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _cluster_bootstrap(
    clusters: Mapping[str, Sequence[float]],
    statistic: Callable[[Sequence[float]], float | None],
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float | None, float | None]:
    """Return percentile limits after resampling complete trajectory clusters."""

    keys = tuple(sorted(clusters))
    if not keys:
        return None, None
    if len(keys) == 1:
        value = statistic(tuple(clusters[keys[0]]))
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(100, int(draws))):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        values = [value for key in sampled for value in clusters[key]]
        estimate = statistic(values)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return None, None
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int(0.025 * (len(estimates) - 1))))
    high_index = max(0, min(len(estimates) - 1, int(0.975 * (len(estimates) - 1))))
    return estimates[low_index], estimates[high_index]


def _metric(
    values: Sequence[float], clusters: Mapping[str, Sequence[float]], *, seed: int
) -> dict[str, float | int | None]:
    low, high = _cluster_bootstrap(clusters, _mean, seed=seed)
    return {
        "estimate": _mean(values),
        "ci_low": low,
        "ci_high": high,
        "rows": len(values),
        "clusters": len(clusters),
    }


def analyze_footprint(root: Path) -> dict[str, Any]:
    """Produce the registered footprint-risk diagnostic and audit artifact."""

    root = Path(root).resolve()
    table = root / "results/v15/canonical/autonomous_transitions.csv"
    manifest_path = root / "results/v15/canonical/manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                manifest = dict(loaded)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    rows = _read_rows(table)
    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        proxy = _float(row.get("delta_proxy"))
        actor = _float(row.get("delta_actor"))
        if proxy is None or actor is None:
            continue
        valid_rows.append(
            {
                "row": row,
                "cluster": str(row.get("run_id") or row.get("transition_id") or f"row-{index}"),
                "error": abs(actor - proxy),
                "reversal": float(proxy > 0.0 and actor < 0.0),
            }
        )
    error_clusters: dict[str, list[float]] = defaultdict(list)
    reversal_clusters: dict[str, list[float]] = defaultdict(list)
    for item in valid_rows:
        error_clusters[item["cluster"]].append(float(item["error"]))
        reversal_clusters[item["cluster"]].append(float(item["reversal"]))
    errors = [float(item["error"]) for item in valid_rows]
    reversals = [float(item["reversal"]) for item in valid_rows]

    associations: list[dict[str, Any]] = []
    for feature in FOOTPRINT_COLUMNS:
        by_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in valid_rows:
            raw_feature = item["row"].get(f"footprint_{feature}")
            if raw_feature is None:
                nested = item["row"].get("footprint")
                raw_feature = nested.get(feature) if isinstance(nested, Mapping) else None
            feature_value = _float(raw_feature)
            if feature_value is not None:
                by_cluster[str(item["cluster"])].append((feature_value, float(item["error"])))
        cluster_values = {
            key: [sum(pair[0] for pair in pairs) / len(pairs)]
            for key, pairs in by_cluster.items()
            if pairs
        }
        cluster_targets = {
            key: [sum(pair[1] for pair in pairs) / len(pairs)]
            for key, pairs in by_cluster.items()
            if pairs
        }
        values = [items[0] for items in cluster_values.values()]
        targets = [cluster_targets[key][0] for key in cluster_values]
        associations.append(
            {
                "feature": feature,
                "n_rows": sum(len(items) for items in by_cluster.values()),
                "n_clusters": len(cluster_values),
                "pearson_error": _pearson(values, targets),
                "outcome_fields_used": [],
                "independent_unit": "trajectory_or_task_cluster",
            }
        )

    phase = str(manifest.get("phase", "DEV"))
    confirmatory = _bool(manifest.get("confirmatory"), default=False)
    status = "CONFIRMATORY" if confirmatory else ("DEV_ONLY" if valid_rows else "NOT_RUN")
    leakage_fields = {
        "delta_actor",
        "delta_strategic",
        "actor_reversal",
        "strategic_reversal",
        "deployment_score",
        "assessment_score",
    }
    decision_path = root / "artifacts/v15/transition_scientific_decision.json"
    decision: Mapping[str, Any] = {}
    if decision_path.is_file():
        try:
            loaded_decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if isinstance(loaded_decision, Mapping):
                decision = loaded_decision
        except (OSError, json.JSONDecodeError):
            decision = {}
    terminal_state = decision.get("terminal_state")
    result: dict[str, Any] = {
        "schema_version": "pivot-v15-footprint-analysis-1",
        "status": status,
        "phase": phase,
        "source": str(table.relative_to(root)) if table.is_relative_to(root) else str(table),
        "source_manifest": str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
        "source_hash": file_hash(table) if table.is_file() else None,
        "source_manifest_sha256": file_hash(manifest_path) if manifest_path.is_file() else None,
        "rows_read": len(rows),
        "rows_with_proxy_and_actor": len(valid_rows),
        "independent_trajectory_units": len(error_clusters),
        "independent_unit": "trajectory_or_task_cluster",
        "terminal_state": terminal_state,
        "analysis_decision_artifact": str(decision_path.relative_to(root)) if decision_path.is_relative_to(root) else str(decision_path),
        "feature_contract": list(FOOTPRINT_COLUMNS),
        "outcome_fields_excluded_from_features": sorted(leakage_fields),
        "leakage_detected": _bool(manifest.get("leakage_detected"), default=False),
        "outcome_chasing": _bool(manifest.get("outcome_chasing"), default=False),
        "transition_error": _metric(errors, error_clusters, seed=20260828),
        "improvement_reversal_rate": _metric(reversals, reversal_clusters, seed=20260829),
        "feature_associations": associations,
        "scientific_claim_allowed": confirmatory and bool(valid_rows) and terminal_state in {"HYPOTHESIS_SUPPORTED", "HYPOTHESIS_NOT_SUPPORTED"},
        "note": (
            "DEV_ONLY descriptive diagnostic; transition rows are nested within "
            "trajectory units and are not confirmatory evidence."
            if not confirmatory
            else "Confirmatory footprint analysis uses the frozen feature contract and cluster units."
        ),
    }
    return write_audit(
        root,
        "footprint_analysis",
        result,
        "Footprint Analysis",
        "Registered pre-gate footprint features are evaluated against transition error with trajectory-cluster bootstrap intervals.",
    )


if __name__ == "__main__":
    cli(analyze_footprint, "Analyze registered transition footprint features")
