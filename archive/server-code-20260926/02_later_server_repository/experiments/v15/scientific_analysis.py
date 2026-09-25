"""Outcome-blind scientific analysis for the V15 evidence contract.

The execution modules deliberately only produce immutable rows.  This module
is the small, shared analysis layer consumed by the command surface, reports,
and figure builders.  It has three design constraints:

* trajectory/task clusters, rather than raw transition rows, are the primary
  inferential unit;
* a development run can never be promoted to a confirmatory claim; and
* terminal states are assigned by fixed rules, including a valid powered null.

No function in this module chooses tasks, operators, or thresholds after
looking at an outcome.  The resulting JSON artifacts are therefore suitable
for number and provenance audits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .audit_support import write_audit
from .canonical import FOOTPRINT_COLUMNS
from .protocol import file_hash

ANALYSIS_SCHEMA_VERSION = "pivot-v15-scientific-analysis-1"
DEFAULT_MIN_CLUSTERS = 30
DEFAULT_BOOTSTRAP_DRAWS = 2000
SignAlternative = Literal["positive", "negative", "nonzero"]


def _finite(value: object) -> float | None:
    """Parse a scalar while treating missing/non-finite values as absent."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "nan"}:
        return None
    try:
        if isinstance(value, (bool, int, float)):
            parsed = float(value)
        else:
            parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_finite(row: Mapping[str, Any], *names: str) -> float | None:
    """Return the first finite value among backward-compatible field names."""

    for name in names:
        value = _finite(row.get(name))
        if value is not None:
            return value
    return None


def _sign(value: float | None, *, tolerance: float = 1e-12) -> int:
    if value is None or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties."""

    if len(left) != len(right) or len(left) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        output = [0.0] * len(values)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and values[order[end]] == values[order[cursor]]:
                end += 1
            rank = (cursor + 1 + end) / 2.0
            for position in order[cursor:end]:
                output[position] = rank
            cursor = end
        return output

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear quantile without a dependency on a particular NumPy version."""

    if not sorted_values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return a cluster-bootstrap mean and percentile interval.

    First the observations within each cluster are averaged, then complete
    clusters are resampled.  This avoids giving a long trajectory more weight
    merely because it generated more candidate rows.
    """

    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        parsed = _finite(value)
        if parsed is not None:
            grouped[str(cluster)].append(parsed)
    cluster_means = [sum(items) / len(items) for items in grouped.values() if items]
    estimate = _mean(cluster_means)
    result: dict[str, float | int | None] = {
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "n_rows": sum(len(items) for items in grouped.values()),
        "n_clusters": len(cluster_means),
    }
    if estimate is None:
        return result
    if len(cluster_means) == 1:
        result["ci_low"] = estimate
        result["ci_high"] = estimate
        return result
    rng = random.Random(int(seed))
    samples: list[float] = []
    draw_count = max(100, int(draws))
    for _ in range(draw_count):
        sampled = [cluster_means[rng.randrange(len(cluster_means))] for _ in cluster_means]
        samples.append(sum(sampled) / len(sampled))
    samples.sort()
    result["ci_low"] = _quantile(samples, alpha / 2.0)
    result["ci_high"] = _quantile(samples, 1.0 - alpha / 2.0)
    return result


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Parse booleans from JSON and CSV without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return default


def _cluster_p_value(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    alternative: SignAlternative = "nonzero",
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> float | None:
    """Compute a deterministic cluster sign-flip p-value around zero.

    This is deliberately conservative: observations are averaged within the
    registered cluster first, then signs are randomized.  It is used only for
    confirmatory summaries; DEV analyses report no p-values.
    """

    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        parsed = _finite(value)
        if parsed is not None:
            grouped[str(cluster)].append(parsed)
    means = [sum(items) / len(items) for items in grouped.values() if items]
    if not means:
        return None
    observed = sum(means) / len(means)
    rng = random.Random(int(seed))
    exceed = 0
    draw_count = max(100, int(draws))
    for _ in range(draw_count):
        null_mean = sum(value * (1.0 if rng.getrandbits(1) else -1.0) for value in means) / len(means)
        if alternative == "positive":
            extreme = null_mean >= observed
        elif alternative == "negative":
            extreme = null_mean <= observed
        else:
            extreme = abs(null_mean) >= abs(observed)
        exceed += int(extreme)
    # Add-one smoothing keeps a finite, reproducible p-value at small N.
    return (exceed + 1.0) / (draw_count + 1.0)


def _metric(values: Sequence[float], clusters: Sequence[str], *, seed: int, draws: int) -> dict[str, Any]:
    output = dict(cluster_bootstrap(values, clusters, seed=seed, draws=draws))
    output["independent_unit"] = "trajectory_or_task_cluster"
    return output


def _attach_p_value(
    metric: dict[str, Any],
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    confirmatory: bool,
    seed: int,
    alternative: SignAlternative,
    draws: int,
) -> dict[str, Any]:
    """Attach a cluster sign-flip p-value only to frozen confirmation data."""

    if confirmatory:
        p_value = _cluster_p_value(values, clusters, seed=seed, alternative=alternative, draws=draws)
        if p_value is not None:
            metric["p_value"] = p_value
    return metric


def _paired_cluster_bootstrap(
    values: Mapping[str, float], *, seed: int, draws: int
) -> dict[str, Any]:
    keys = sorted(values)
    return cluster_bootstrap(
        [values[key] for key in keys],
        keys,
        seed=seed,
        draws=draws,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down adjustment with deterministic tie handling."""

    clean = {
        str(name): max(0.0, min(1.0, float(value)))
        for name, value in p_values.items()
        if math.isfinite(float(value))
    }
    ordered = sorted(clean.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def classify_terminal_state(
    *,
    attempted: bool,
    confirmatory: bool,
    design_valid: bool,
    implementation_failures: int,
    n_clusters: int,
    estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
    alternative: SignAlternative = "positive",
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
) -> str | None:
    """Assign exactly one closed scientific terminal state when applicable.

    Development artifacts intentionally terminate as ``UNDERPOWERED`` even
    when their point estimate has the expected sign.  A confirmatory null is a
    valid terminal state, not an implementation failure.
    """

    if not attempted:
        return None
    if int(implementation_failures) > 0:
        return "IMPLEMENTATION_FAILURE"
    if not design_valid:
        return "DESIGN_INVALID"
    if not confirmatory or int(n_clusters) < int(min_clusters):
        return "UNDERPOWERED"
    if estimate is None or ci_low is None or ci_high is None:
        return "HYPOTHESIS_NOT_SUPPORTED"
    if alternative == "positive":
        supported = ci_low > 0.0
    elif alternative == "negative":
        supported = ci_high < 0.0
    else:
        supported = ci_low > 0.0 or ci_high < 0.0
    return "HYPOTHESIS_SUPPORTED" if supported else "HYPOTHESIS_NOT_SUPPORTED"


def _read_json(path: Path, fallback: Any = None) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _phase_source(
    root: Path,
    *,
    phase_name: str,
    jsonl_name: str,
    csv_name: str,
    confirmatory: bool | None,
    source_dir: Path | None,
    canonical_fallback: bool = False,
) -> tuple[Path | None, dict[str, Any], list[dict[str, Any]], bool]:
    """Choose a source without allowing DEV fallback for confirmation."""

    root = Path(root).resolve()
    if source_dir is not None:
        candidates = [Path(source_dir).resolve()]
    elif confirmatory is True:
        candidates = [root / "results/v15" / phase_name]
    elif confirmatory is False:
        candidates = [root / "results/v15" / f"dev-{phase_name}"]
    else:
        candidates = [root / "results/v15" / phase_name, root / "results/v15" / f"dev-{phase_name}"]
        if canonical_fallback:
            # Canonical tables are a compatibility fallback for small audit
            # fixtures. A real phase directory always wins, and an explicit
            # confirmatory request can never reach this branch.
            candidates.append(root / "results/v15" / "canonical")
    for directory in candidates:
        manifest = _read_json(directory / "manifest.json", {})
        manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
        is_confirmatory = _as_bool(manifest.get("confirmatory"), default=False)
        if confirmatory is True and not is_confirmatory:
            continue
        path = directory / jsonl_name
        rows = _read_jsonl(path) if path.is_file() else _read_csv(directory / csv_name)
        has_source = path.is_file() or (directory / csv_name).is_file()
        if not has_source and not (directory / "manifest.json").is_file():
            continue
        # An explicitly confirmatory analysis may only consume the frozen
        # confirmatory directory; this check prevents a stale DEV row from
        # being silently re-labelled.
        if confirmatory is True and not is_confirmatory:
            continue
        # A canonical manifest without the requested table is not an attempted
        # optional phase.  This distinction keeps H4/H6 ``NOT_RUN`` instead of
        # relabelling an empty compatibility directory as data.
        if not has_source and directory.name == "canonical":
            continue
        return directory, manifest, rows, is_confirmatory
    return None, {}, [], False


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_result(
    root: Path,
    *,
    artifact_name: str,
    title: str,
    summary: str,
    result: dict[str, Any],
    source_dir: Path | None,
) -> dict[str, Any]:
    """Write the analysis artifact and a phase-local scientific decision."""

    root = Path(root).resolve()
    result = dict(result)
    result.setdefault("schema_version", ANALYSIS_SCHEMA_VERSION)
    payload = write_audit(root, artifact_name, result, title, summary)
    decision = {
        "schema_version": "pivot-v15-scientific-decision-1",
        "analysis_artifact": f"artifacts/v15/{artifact_name}.json",
        "hypothesis": result.get("primary_hypothesis"),
        "terminal_state": result.get("terminal_state"),
        "status": result.get("status"),
        "confirmatory": bool(result.get("confirmatory", False)),
        "independent_unit": result.get("independent_unit", "trajectory_or_task_cluster"),
        "independent_n": result.get("independent_n", 0),
        "outcome_chasing": bool(result.get("outcome_chasing", False)),
        "source_hash": result.get("source_hash"),
        "criterion": result.get("criterion"),
    }
    decision_path = root / "artifacts/v15" / f"{artifact_name.removesuffix('_analysis')}_scientific_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_dir is not None:
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "scientific_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    payload["decision_artifact"] = _relative(decision_path, root)
    return payload


def _manifest_design_valid(manifest: Mapping[str, Any]) -> bool:
    return (not _as_bool(manifest.get("outcome_chasing"), default=False)) and str(manifest.get("status", "")) not in {
        "DESIGN_INVALID",
        "IMPLEMENTATION_FAILURE",
    }


def _transition_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        proxy = _finite(row.get("delta_proxy", row.get("proxy_delta")))
        actor = _finite(row.get("delta_actor", row.get("delta_true")))
        if proxy is None or actor is None:
            continue
        cluster = str(row.get("run_id") or row.get("trajectory_id") or row.get("transition_id") or f"row-{index}")
        output.append(
            {
                **row,
                "cluster": cluster,
                "proxy": proxy,
                "actor": actor,
                "error": abs(actor - proxy),
                "sign_match": float(_sign(proxy) != 0 and _sign(proxy) == _sign(actor)),
                "comparable": float(_sign(proxy) != 0 and _sign(actor) != 0),
                "reversal": float(proxy > 0.0 and actor < 0.0),
            }
        )
    return output


def _transition_metric_bundle(rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> dict[str, Any]:
    clusters = [str(row["cluster"]) for row in rows]
    ide_values = [float(row["error"]) for row in rows]
    metrics: dict[str, Any] = {"IDE": _metric(ide_values, clusters, seed=seed, draws=draws)}
    comparable = [row for row in rows if row["comparable"]]
    metrics["ISC"] = _metric(
        [float(row["sign_match"]) for row in comparable],
        [str(row["cluster"]) for row in comparable],
        seed=seed + 1,
        draws=draws,
    )
    positive = [row for row in rows if float(row["proxy"]) > 0.0]
    metrics["IRR"] = _metric(
        [float(row["actor"]) < 0.0 for row in positive],
        [str(row["cluster"]) for row in positive],
        seed=seed + 2,
        draws=draws,
    )
    actor_positive = [
        row for row in rows if float(row["actor"]) > 0.0 and _finite(row.get("delta_strategic")) is not None
    ]
    strategic_values = [_finite(row.get("delta_strategic")) for row in actor_positive]
    strategic = [
        (float(value), str(row["cluster"]))
        for value, row in zip(strategic_values, actor_positive)
        if value is not None
    ]
    metrics["SIRR"] = _metric(
        [float(value < 0.0) for value, _ in strategic],
        [cluster for _, cluster in strategic],
        seed=seed + 3,
        draws=draws,
    )
    metrics["strategic_effect"] = _metric(
        [
            float(value - float(row["actor"]))
            for value, row in zip(strategic_values, actor_positive)
            if value is not None
        ],
        [str(row["cluster"]) for value, row in zip(strategic_values, actor_positive) if value is not None],
        seed=seed + 4,
        draws=draws,
    )
    return metrics


def _group_transition_metrics(
    rows: Sequence[Mapping[str, Any]], *, key: str, seed: int, draws: int
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        name: _transition_metric_bundle(group, seed=seed + index * 17, draws=draws)
        for index, (name, group) in enumerate(sorted(grouped.items()))
    }


def _policy_rank_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one rank-fidelity observation per candidate batch.

    The level scores are optional for backward compatibility.  A batch is
    usable only when every candidate has both proxy and actor level scores;
    otherwise it is reported as unavailable rather than reconstructed from a
    delta (which would collapse the global and transition estimands).
    """

    batches: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        batches[(str(row.get("run_id", "")), str(row.get("round", 0)))].append(row)
    output: list[dict[str, Any]] = []
    for (run_id, round_id), batch in sorted(batches.items()):
        values: list[tuple[float, float]] = []
        for row in batch:
            proxy = _finite(row.get("proxy_candidate_score"))
            actor = _finite(row.get("actor_candidate_score"))
            if proxy is None or actor is None:
                values = []
                break
            values.append((proxy, actor))
        if len(values) < 2:
            continue
        output.append(
            {
                "cluster": run_id,
                "round": round_id,
                "operator": str(batch[0].get("operator", "unknown")),
                "task_family": str(batch[0].get("task_family", "unknown")),
                "rank_fidelity": _rank_correlation(
                    [proxy for proxy, _ in values], [actor for _, actor in values]
                ),
                "candidate_count": len(values),
            }
        )
    return [row for row in output if row["rank_fidelity"] is not None]


def _rank_metric(rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> dict[str, Any]:
    rank_rows = _policy_rank_rows(rows)
    return _metric(
        [float(row["rank_fidelity"]) for row in rank_rows],
        [str(row["cluster"]) for row in rank_rows],
        seed=seed,
        draws=draws,
    )


def _cluster_correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Estimate a Pearson association after reducing observations by cluster."""

    if len(x_values) != len(y_values) or len(x_values) != len(clusters):
        raise ValueError("association inputs must have equal length")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for x, y, cluster in zip(x_values, y_values, clusters):
        if _finite(x) is not None and _finite(y) is not None:
            grouped[str(cluster)].append((float(x), float(y)))
    points = [
        (sum(x for x, _ in values) / len(values), sum(y for _, y in values) / len(values))
        for values in grouped.values()
        if values
    ]

    def corr(sample: Sequence[tuple[float, float]]) -> float | None:
        if len(sample) < 2:
            return None
        x_mean = sum(x for x, _ in sample) / len(sample)
        y_mean = sum(y for _, y in sample) / len(sample)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in sample)
        x_scale = math.sqrt(sum((x - x_mean) ** 2 for x, _ in sample))
        y_scale = math.sqrt(sum((y - y_mean) ** 2 for _, y in sample))
        return numerator / (x_scale * y_scale) if x_scale and y_scale else 0.0

    estimate = corr(points)
    output: dict[str, Any] = {
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "n_rows": sum(len(values) for values in grouped.values()),
        "n_clusters": len(points),
        "independent_unit": "trajectory_or_task_cluster",
    }
    if estimate is None or len(points) < 2:
        return output
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(max(100, int(draws))):
        resampled = [points[rng.randrange(len(points))] for _ in points]
        value = corr(resampled)
        if value is not None:
            samples.append(value)
    samples.sort()
    if samples:
        output["ci_low"] = _quantile(samples, 0.025)
        output["ci_high"] = _quantile(samples, 0.975)
    return output


def _footprint_value(row: Mapping[str, Any]) -> float:
    raw = row.get("footprint")
    if isinstance(raw, Mapping):
        values = [_finite(raw.get(name)) for name in FOOTPRINT_COLUMNS]
    else:
        values = [_finite(row.get(f"footprint_{name}")) for name in FOOTPRINT_COLUMNS]
    return sum(abs(value) for value in values if value is not None)


def analyze_transition_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    source_dir: Path | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Analyze observer/deployment transition fidelity and write its contract."""

    root = Path(root).resolve()
    directory, manifest, raw_rows, is_confirmatory = _phase_source(
        root,
        phase_name="external-transition-audit",
        jsonl_name="autonomous_transitions.jsonl",
        csv_name="autonomous_transitions.csv",
        confirmatory=confirmatory,
        source_dir=source_dir,
        canonical_fallback=True,
    )
    rows = _transition_rows(raw_rows)
    attempted = directory is not None
    failures = int(_finite(manifest.get("execution_failure_count")) or 0)
    outcome_chasing = _as_bool(manifest.get("outcome_chasing"), default=False)
    leakage_detected = _as_bool(manifest.get("leakage_detected"), default=False)
    design_valid = _manifest_design_valid(manifest) and not leakage_detected and bool(rows or not attempted)
    metrics = _transition_metric_bundle(rows, seed=20260828, draws=draws) if rows else {
        name: _metric([], [], seed=20260828 + index, draws=draws)
        for index, name in enumerate(("IDE", "ISC", "IRR", "SIRR", "strategic_effect"))
    }
    if rows:
        _attach_p_value(
            metrics["IDE"],
            [float(row["error"]) for row in rows],
            [str(row["cluster"]) for row in rows],
            confirmatory=is_confirmatory,
            seed=20501,
            alternative="positive",
            draws=draws,
        )
    ide = metrics["IDE"]
    terminal = classify_terminal_state(
        attempted=attempted,
        confirmatory=is_confirmatory,
        design_valid=design_valid,
        implementation_failures=failures,
        n_clusters=int(ide.get("n_clusters", 0)),
        estimate=_finite(ide.get("estimate")),
        ci_low=_finite(ide.get("ci_low")),
        ci_high=_finite(ide.get("ci_high")),
        alternative="nonzero",
        min_clusters=min_clusters,
    )
    source_path = directory / "autonomous_transitions.jsonl" if directory else None
    if source_path is None or not source_path.is_file():
        source_path = directory / "autonomous_transitions.csv" if directory else None
    source_hash = file_hash(source_path) if source_path and source_path.is_file() else None
    operator_metrics = _group_transition_metrics(rows, key="operator", seed=20300, draws=draws) if rows else {}
    task_metrics = _group_transition_metrics(rows, key="task_family", seed=20400, draws=draws) if rows else {}
    rank_rows = _policy_rank_rows(rows)
    global_rank = _rank_metric(rows, seed=20500, draws=draws)
    rank_by_operator = {
        operator: _rank_metric(
            [row for row in rows if str(row.get("operator", "unknown")) == operator],
            seed=20550 + index * 17,
            draws=draws,
        )
        for index, operator in enumerate(sorted({str(row.get("operator", "unknown")) for row in rank_rows}))
    }
    footprint_values = [_footprint_value(row) for row in rows]
    footprint_clusters = [str(row["cluster"]) for row in rows]
    y_values = [float(row["error"]) for row in rows]
    footprint_error_association = _cluster_correlation(
        footprint_values,
        y_values,
        footprint_clusters,
        seed=20575,
        draws=draws,
    )
    footprint_error_association.update(
        {
            "feature": "registered_footprint_l1_norm",
            "outcome_fields_used": [],
        }
    )
    global_rank_estimate = _finite(global_rank.get("estimate"))
    rank_relative_rows = [
        row
        for row in rank_rows
        if global_rank_estimate is not None and _finite(row.get("rank_fidelity")) is not None
    ]
    rank_relative_values = [
        float(value) - global_rank_estimate
        for row in rank_relative_rows
        for value in [_finite(row.get("rank_fidelity"))]
        if value is not None and global_rank_estimate is not None
    ]
    rank_relative_clusters = [str(row["cluster"]) for row in rank_relative_rows]
    rank_relative_metric = _metric(
        rank_relative_values,
        rank_relative_clusters,
        seed=20576,
        draws=draws,
    )
    if is_confirmatory and rank_relative_rows:
        _attach_p_value(
            rank_relative_metric,
            rank_relative_values,
            rank_relative_clusters,
            confirmatory=True,
            seed=20577,
            alternative="nonzero",
            draws=draws,
        )
    h5_metric = footprint_error_association
    h2_terminal = classify_terminal_state(
        attempted=attempted,
        confirmatory=is_confirmatory,
        design_valid=design_valid and (bool(rank_rows) or not is_confirmatory),
        implementation_failures=failures,
        n_clusters=int(rank_relative_metric.get("n_clusters", 0)),
        estimate=_finite(rank_relative_metric.get("estimate")),
        ci_low=_finite(rank_relative_metric.get("ci_low")),
        ci_high=_finite(rank_relative_metric.get("ci_high")),
        alternative="nonzero",
        min_clusters=min_clusters,
    )
    h5_terminal = classify_terminal_state(
        attempted=attempted,
        confirmatory=is_confirmatory,
        design_valid=design_valid and (bool(rows) or not is_confirmatory),
        implementation_failures=failures,
        n_clusters=int(h5_metric.get("n_clusters", 0)),
        estimate=_finite(h5_metric.get("estimate")),
        ci_low=_finite(h5_metric.get("ci_low")),
        ci_high=_finite(h5_metric.get("ci_high")),
        alternative="nonzero",
        min_clusters=min_clusters,
    )
    hypotheses: dict[str, dict[str, Any]] = {
        "H1": {
            "statement": "autonomously generated updates have measurable observer/deployment error",
            "primary_metric": "IDE",
            "terminal_state": terminal,
            "metric": ide,
            "criterion": "95% cluster-bootstrap interval excludes zero after the frozen independent-N rule",
        },
        "H2": {
            "statement": "operator-conditioned transition fidelity differs from global fidelity",
            "primary_metric": "operator_relative_rank_fidelity",
            "terminal_state": h2_terminal,
            "metric": rank_relative_metric,
            "criterion": "predeclared operator strata are compared with global policy rank fidelity; no delta reconstruction is used",
        },
        "H5": {
            "statement": "registered update footprint predicts transition error",
            "primary_metric": "registered_footprint_l1_norm_association",
            "terminal_state": h5_terminal,
            "metric": h5_metric,
            "criterion": "association is descriptive unless a frozen confirmatory split and interval are available",
        },
    }
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "CONFIRMATORY_ANALYZED" if is_confirmatory and attempted else ("DEV_ONLY" if attempted else "NOT_RUN"),
        "phase": str(manifest.get("phase", "NOT_RUN" if not attempted else "DEV")),
        "confirmatory": is_confirmatory,
        "primary_hypothesis": "H1",
        "terminal_state": terminal,
        "criterion": str(hypotheses["H1"]["criterion"]),
        "independent_unit": "trajectory_or_task_cluster",
        "independent_n": int(ide.get("n_clusters", 0)),
        "rows_read": len(raw_rows),
        "rows_analyzed": len(rows),
        "source": _relative(source_path, root) if source_path else None,
        "source_hash": source_hash,
        "source_manifest_hash": file_hash(directory / "manifest.json") if directory and (directory / "manifest.json").is_file() else None,
        "metrics": metrics,
        "by_operator": operator_metrics,
        "by_task_family": task_metrics,
        "policy_level": {
            "available": bool(rank_rows),
            "global_rank_fidelity": global_rank,
            "rank_fidelity_by_operator": rank_by_operator,
            "rank_observation_count": len(rank_rows),
            "note": "Policy-level rank fidelity requires recorded level scores for all candidates in a batch.",
        },
        "footprint": {
            "registered_features": list(FOOTPRINT_COLUMNS),
            "l1_error_association": footprint_error_association,
            "outcome_fields_used": [],
        },
        "hypotheses": hypotheses,
        "implementation_failures": failures,
        "design_valid": design_valid,
        "outcome_chasing": outcome_chasing,
        "global_policy_value_fidelity_available": bool(rank_rows),
        "leakage_detected": leakage_detected,
        "note": (
            "DEV-only descriptive result; no scientific claim is allowed until the frozen confirmatory archive is opened."
            if not is_confirmatory
            else "Confirmatory transition analysis uses trajectory-cluster inference and the frozen feature contract."
        ),
    }
    return _write_result(
        root,
        artifact_name="transition_analysis",
        title="Transition Analysis",
        summary="Cluster-bootstrap analysis of observer versus deployment improvement transitions.",
        result=result,
        source_dir=directory,
    )


def _promotion_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('run_id', 'unknown')}::{row.get('round', 0)}::{row.get('hf_budget', row.get('budget', 0))}"


def _promotion_budget(row: Mapping[str, Any]) -> int:
    value = _finite(row.get("hf_budget", row.get("budget", 0)))
    return int(value or 0)


def analyze_promotion_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    source_dir: Path | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    target_budget: int | None = None,
) -> dict[str, Any]:
    """Analyze promotion regret with the fixed Proxy-minus-PIVOT orientation."""

    root = Path(root).resolve()
    directory, manifest, raw_rows, is_confirmatory = _phase_source(
        root,
        phase_name="external-promotion",
        jsonl_name="promotion_results.jsonl",
        csv_name="promotion_results.csv",
        confirmatory=confirmatory,
        source_dir=source_dir,
        canonical_fallback=True,
    )
    attempted = directory is not None
    query_path = directory / "hf_queries.jsonl" if directory else None
    query_rows = _read_jsonl(query_path) if query_path and query_path.is_file() else (
        _read_csv(directory / "hf_queries.csv") if directory else []
    )
    budgets = sorted({_promotion_budget(row) for row in raw_rows})
    chosen_budget = int(target_budget if target_budget is not None else (max(budgets) if budgets else 0))
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        method = str(row.get("method", ""))
        budget = _promotion_budget(row)
        value = _finite(row.get("ISR", row.get("isr")))
        if method and value is not None:
            grouped[(method, budget)].append(row)
    method_metrics: dict[str, Any] = {}
    for index, ((method, budget), rows) in enumerate(sorted(grouped.items())):
        metric_values = [float(_finite(row.get("ISR", row.get("isr"))) or 0.0) for row in rows]
        clusters = [_promotion_key(row) for row in rows]
        method_metrics.setdefault(method, {})[str(budget)] = _metric(
            metric_values, clusters, seed=20600 + index * 13, draws=draws
        )
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in raw_rows:
        if _promotion_budget(row) != chosen_budget:
            continue
        method = str(row.get("method", ""))
        value = _finite(row.get("ISR", row.get("isr")))
        if value is not None:
            paired[_promotion_key(row)][method] = value
    differences: dict[str, float] = {}
    for key, by_method in paired.items():
        if "Proxy Only" in by_method and "PIVOT-VOI" in by_method:
            differences[key] = float(by_method["Proxy Only"] - by_method["PIVOT-VOI"])
    effect = _paired_cluster_bootstrap(differences, seed=20701, draws=draws)
    effect["direction"] = "proxy_minus_pivot; positive_favors_pivot"
    if differences:
        _attach_p_value(
            effect,
            list(differences.values()),
            list(differences),
            confirmatory=is_confirmatory,
            seed=20702,
            alternative="positive",
            draws=draws,
        )
    batches_by_method: dict[str, set[str]] = defaultdict(set)
    for row in raw_rows:
        batch = str(row.get("candidate_batch_hash") or _promotion_key(row))
        batches_by_method[str(row.get("method", ""))].add(batch)
    query_accounting = {
        "logical_hf_queries": sum(_as_bool(row.get("logical_hf_query"), default=True) for row in query_rows),
        "physical_pair_evaluations": sum(
            _as_bool(row.get("physical_pair_evaluation"), default=True) for row in query_rows
        ),
        "cache_hit_rows": sum(_as_bool(row.get("cache_hit"), default=False) for row in query_rows),
        "query_rows": len(query_rows),
        "legacy_fields_missing": any("logical_hf_query" not in row for row in query_rows),
        "post_decision_truth_excluded": True,
    }
    failures = int(_finite(manifest.get("execution_failure_count")) or 0)
    outcome_chasing = _as_bool(manifest.get("outcome_chasing"), default=False)
    leakage_detected = _as_bool(manifest.get("leakage_detected"), default=False)
    design_valid = _manifest_design_valid(manifest) and not leakage_detected
    terminal = classify_terminal_state(
        attempted=attempted,
        confirmatory=is_confirmatory,
        design_valid=design_valid,
        implementation_failures=failures,
        n_clusters=int(_finite(effect.get("n_clusters")) or 0),
        estimate=_finite(effect.get("estimate")),
        ci_low=_finite(effect.get("ci_low")),
        ci_high=_finite(effect.get("ci_high")),
        alternative="positive",
        min_clusters=min_clusters,
    )
    source_path = directory / "promotion_results.jsonl" if directory else None
    if source_path is None or not source_path.is_file():
        source_path = directory / "promotion_results.csv" if directory else None
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "CONFIRMATORY_ANALYZED" if is_confirmatory and attempted else ("DEV_ONLY" if attempted else "NOT_RUN"),
        "phase": str(manifest.get("phase", "NOT_RUN" if not attempted else "DEV")),
        "confirmatory": is_confirmatory,
        "primary_hypothesis": "H3",
        "terminal_state": terminal,
        "criterion": "Proxy Only ISR minus PIVOT-VOI ISR has a positive 95% cluster-bootstrap lower bound at the frozen target budget",
        "independent_unit": "candidate_batch_cluster",
        "independent_n": int(_finite(effect.get("n_clusters")) or 0),
        "target_budget": chosen_budget,
        "budgets": budgets,
        "source": _relative(source_path, root) if source_path else None,
        "source_hash": file_hash(source_path) if source_path and source_path.is_file() else None,
        "source_manifest_hash": file_hash(directory / "manifest.json") if directory and (directory / "manifest.json").is_file() else None,
        "method_metrics": method_metrics,
        "paired_effect": effect,
        "query_accounting": query_accounting,
        "fairness": {
            "candidate_batch_hashes_by_method": {name: sorted(values) for name, values in sorted(batches_by_method.items())},
            "same_candidate_batches_observed": len({frozenset(values) for values in batches_by_method.values()}) <= 1,
            "all_hf_oracle_is_reference_only": True,
        },
        "implementation_failures": failures,
        "design_valid": design_valid,
        "outcome_chasing": outcome_chasing,
        "leakage_detected": leakage_detected,
        "note": "ISR orientation is fixed as Proxy Only minus PIVOT-VOI; positive values always favor PIVOT.",
    }
    return _write_result(
        root,
        artifact_name="promotion_analysis",
        title="Promotion Analysis",
        summary="Cluster-bootstrap promotion regret and paired high-fidelity query accounting.",
        result=result,
        source_dir=directory,
    )


def _assessment_rows(directory: Path | None) -> list[dict[str, Any]]:
    if directory is None:
        return []
    path = directory / "assessment_results.jsonl"
    return _read_jsonl(path) if path.is_file() else _read_csv(directory / "assessment_results.csv")


def analyze_closed_loop_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    source_dir: Path | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Analyze terminal deployment transfer without re-querying assessment data."""

    root = Path(root).resolve()
    directory, manifest, raw_rows, is_confirmatory = _phase_source(
        root,
        phase_name="external-closed-loop",
        jsonl_name="closed_loop_results.jsonl",
        csv_name="closed_loop_results.csv",
        confirmatory=confirmatory,
        source_dir=source_dir,
        canonical_fallback=True,
    )
    assessments = _assessment_rows(directory)
    attempted = directory is not None
    failures = int(_finite(manifest.get("execution_failure_count")) or 0)
    outcome_chasing = _as_bool(manifest.get("outcome_chasing"), default=False)
    leakage_detected = _as_bool(manifest.get("leakage_detected"), default=False)
    endpoint_by_method: dict[str, dict[str, float]] = defaultdict(dict)
    for row in assessments:
        method = str(row.get("method", ""))
        run_id = str(row.get("run_id", ""))
        value = _finite(row.get("assessment_score", row.get("assessment_score_if_terminal")))
        if method and run_id and value is not None:
            endpoint_by_method[method][run_id] = value
    endpoint_differences = {
        run_id: values["PIVOT-VOI"] - values["Proxy Only"]
        for run_id, values in (
            {
                run_id: {method: scores[run_id] for method, scores in endpoint_by_method.items() if run_id in scores}
                for run_id in sorted({run for scores in endpoint_by_method.values() for run in scores})
            }
        ).items()
        if "PIVOT-VOI" in values and "Proxy Only" in values
    }
    endpoint_effect = _paired_cluster_bootstrap(endpoint_differences, seed=20801, draws=draws)
    if endpoint_differences:
        _attach_p_value(
            endpoint_effect,
            list(endpoint_differences.values()),
            list(endpoint_differences),
            confirmatory=is_confirmatory,
            seed=20803,
            alternative="positive",
            draws=draws,
        )
    cisr_by_method: dict[str, dict[str, float]] = defaultdict(dict)
    for row in raw_rows:
        method = str(row.get("method", ""))
        run_id = str(row.get("run_id", ""))
        value = _finite(row.get("CISR", row.get("cisr")))
        if method and run_id and value is not None:
            cisr_by_method[method][run_id] = value
    cisr_differences = {
        run_id: values["Proxy Only"] - values["PIVOT-VOI"]
        for run_id, values in (
            {
                run_id: {method: scores[run_id] for method, scores in cisr_by_method.items() if run_id in scores}
                for run_id in sorted({run for scores in cisr_by_method.values() for run in scores})
            }
        ).items()
        if "PIVOT-VOI" in values and "Proxy Only" in values
    }
    cisr_effect = _paired_cluster_bootstrap(cisr_differences, seed=20802, draws=draws)
    cisr_effect["direction"] = "proxy_minus_pivot; positive_favors_pivot"
    sealed = _as_bool(manifest.get("assessment_sealed_until_terminal"), default=False) and _as_bool(
        manifest.get("terminal_assessment_exactly_once"), default=False
    )
    assessment_audit = {
        "sealed": sealed,
        "assessment_rows": len(assessments),
        "unique_terminal_pairs": len(endpoint_differences),
        "all_rows_terminal_role": all(str(row.get("role", "")) == "terminal_assessor" for row in assessments),
        "all_rows_queried_once": all(_as_bool(row.get("queried_once"), default=False) for row in assessments),
        "outcomes_returned_to_operator": False,
    }
    design_valid = bool(
        _manifest_design_valid(manifest)
        and not leakage_detected
        and sealed
        and bool(assessment_audit["all_rows_terminal_role"])
    )
    terminal = classify_terminal_state(
        attempted=attempted,
        confirmatory=is_confirmatory,
        design_valid=design_valid,
        implementation_failures=failures,
        n_clusters=int(_finite(endpoint_effect.get("n_clusters")) or 0),
        estimate=_finite(endpoint_effect.get("estimate")),
        ci_low=_finite(endpoint_effect.get("ci_low")),
        ci_high=_finite(endpoint_effect.get("ci_high")),
        alternative="positive",
        min_clusters=min_clusters,
    )
    source_path = directory / "closed_loop_results.jsonl" if directory else None
    if source_path is None or not source_path.is_file():
        source_path = directory / "closed_loop_results.csv" if directory else None
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "CONFIRMATORY_ANALYZED" if is_confirmatory and attempted else ("DEV_ONLY" if attempted else "NOT_RUN"),
        "phase": str(manifest.get("phase", "NOT_RUN" if not attempted else "DEV")),
        "confirmatory": is_confirmatory,
        "primary_hypothesis": "H3_closed_loop_transfer",
        "terminal_state": terminal,
        "criterion": "PIVOT-VOI minus Proxy Only terminal assessment score has a positive 95% paired cluster-bootstrap lower bound",
        "independent_unit": "trajectory_cluster",
        "independent_n": int(_finite(endpoint_effect.get("n_clusters")) or 0),
        "source": _relative(source_path, root) if source_path else None,
        "source_hash": file_hash(source_path) if source_path and source_path.is_file() else None,
        "source_manifest_hash": file_hash(directory / "manifest.json") if directory and (directory / "manifest.json").is_file() else None,
        "endpoint_by_method": {
            method: _metric(list(scores.values()), list(scores), seed=20900 + index * 11, draws=draws)
            for index, (method, scores) in enumerate(sorted(endpoint_by_method.items()))
        },
        "endpoint_effect": endpoint_effect,
        "cisr_effect": cisr_effect,
        "assessment_audit": assessment_audit,
        "implementation_failures": failures,
        "design_valid": design_valid,
        "outcome_chasing": outcome_chasing,
        "leakage_detected": leakage_detected,
        "note": "Terminal assessment is summarized from existing rows and is never re-queried by analysis.",
    }
    return _write_result(
        root,
        artifact_name="closed_loop_analysis",
        title="Closed-Loop Analysis",
        summary="Paired terminal deployment transfer and cumulative selection-regret analysis.",
        result=result,
        source_dir=directory,
    )


def _analyze_optional_phase(
    root: Path,
    *,
    phase_name: str,
    jsonl_name: str,
    csv_name: str,
    artifact_name: str,
    title: str,
    hypothesis: str,
    confirmatory: bool | None,
    min_clusters: int,
    draws: int,
    alternate_jsonl_names: Sequence[str] = (),
) -> tuple[dict[str, Any], Path | None, list[dict[str, Any]], dict[str, Any]]:
    directory, manifest, rows, is_confirmatory = _phase_source(
        root,
        phase_name=phase_name,
        jsonl_name=jsonl_name,
        csv_name=csv_name,
        confirmatory=confirmatory,
        source_dir=None,
    )
    attempted = directory is not None
    failures = int(_finite(manifest.get("execution_failure_count")) or 0)
    source_path = directory / jsonl_name if directory else None
    if (source_path is None or not source_path.is_file()) and directory is not None:
        for alternate in alternate_jsonl_names:
            candidate = directory / alternate
            if candidate.is_file():
                source_path = candidate
                rows = _read_jsonl(candidate)
                break
    if source_path is None or not source_path.is_file():
        source_path = directory / csv_name if directory else None
    base = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "CONFIRMATORY_ANALYZED" if is_confirmatory and attempted else ("DEV_ONLY" if attempted else "NOT_RUN"),
        "phase": str(manifest.get("phase", "NOT_RUN" if not attempted else "DEV")),
        "confirmatory": is_confirmatory,
        "primary_hypothesis": hypothesis,
        "source": _relative(source_path, root) if source_path else None,
        "source_hash": file_hash(source_path) if source_path and source_path.is_file() else None,
        "source_manifest_hash": file_hash(directory / "manifest.json") if directory and (directory / "manifest.json").is_file() else None,
        "rows_read": len(rows),
        "implementation_failures": failures,
        "outcome_chasing": _as_bool(manifest.get("outcome_chasing"), default=False),
        "leakage_detected": _as_bool(manifest.get("leakage_detected"), default=False),
        "design_valid": _manifest_design_valid(manifest) and not _as_bool(manifest.get("leakage_detected"), default=False),
    }
    return base, directory, rows, manifest


def analyze_ablations_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Summarize registered ablations without treating overlap as power."""

    base, directory, rows, manifest = _analyze_optional_phase(
        Path(root).resolve(),
        phase_name="external-ablations",
        jsonl_name="ablation_results.jsonl",
        csv_name="ablation_results.csv",
        artifact_name="ablation_analysis",
        title="Ablation Analysis",
        hypothesis="H4",
        confirmatory=confirmatory,
        min_clusters=min_clusters,
        draws=draws,
    )
    value_pairs = [
        (
            float(value),
            f"{row.get('run_id', row.get('independent_unit_count', index))}::{row.get('ablation', 'unknown')}",
        )
        for index, row in enumerate(rows)
        for value in [_finite(row.get("value"))]
        if value is not None
    ]
    values = [value for value, _ in value_pairs]
    clusters = [cluster for _, cluster in value_pairs]
    overlap_metric = _metric(values, clusters, seed=21001, draws=draws)
    no_pairing = [row for row in rows if row.get("ablation") == "no_pairing"]
    pairing_values = [value for value in (_finite(row.get("unpaired_delta")) for row in no_pairing) if value is not None]
    pairing_metric = _metric(
        pairing_values,
        [str(row.get("run_id", index)) for index, row in enumerate(no_pairing) if _finite(row.get("unpaired_delta")) is not None],
        seed=21002,
        draws=draws,
    )
    independent_n = len({str(row.get("run_id", row.get("independent_unit_count", index))) for index, row in enumerate(rows)})
    terminal = classify_terminal_state(
        attempted=directory is not None,
        confirmatory=bool(base["confirmatory"]),
        design_valid=bool(base["design_valid"]),
        implementation_failures=int(base["implementation_failures"]),
        n_clusters=independent_n,
        estimate=None,
        ci_low=None,
        ci_high=None,
        alternative="nonzero",
        min_clusters=min_clusters,
    )
    base.update(
        {
            "terminal_state": terminal,
            "criterion": "paired and unpaired evidence are compared on the frozen archive; no post-outcome feature selection",
            "independent_unit": "trajectory_or_task_cluster",
            "independent_n": independent_n,
            "registered_ablation_families": sorted({str(row.get("ablation", "unknown")) for row in rows}),
            "all_rows": _metric(values, clusters, seed=21003, draws=draws),
            "no_pairing_diagnostic": pairing_metric,
            "query_overlap_diagnostic": overlap_metric,
            "assessment_accessed": _as_bool(manifest.get("assessment_accessed"), default=False),
            "note": "DEV ablation rows are diagnostic; a null or overlap does not justify redesigning the protocol.",
        }
    )
    return _write_result(
        Path(root).resolve(),
        artifact_name="ablation_analysis",
        title="Ablation Analysis",
        summary="Registered pairing, footprint, and acquisition ablation diagnostics.",
        result=base,
        source_dir=directory,
    )


def analyze_strategic_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Analyze the frozen identity-blind response layer."""

    base, directory, rows, manifest = _analyze_optional_phase(
        Path(root).resolve(),
        phase_name="external-strategic-response",
        jsonl_name="response_audits.jsonl",
        csv_name="response_audits.csv",
        artifact_name="strategic_analysis",
        title="Strategic Response Analysis",
        hypothesis="H3_strategic_response",
        confirmatory=confirmatory,
        min_clusters=min_clusters,
        draws=draws,
    )
    valid = [
        row
        for row in rows
        if _first_finite(row, "delta_strategic", "delta_response_utility") is not None
        and _finite(row.get("delta_actor")) is not None
    ]
    effects = [
        float(_first_finite(row, "delta_strategic", "delta_response_utility") or 0.0)
        - float(row["delta_actor"])
        for row in valid
    ]
    clusters = [str(row.get("run_id") or row.get("transition_id") or index) for index, row in enumerate(valid)]
    effect_metric = _metric(effects, clusters, seed=21101, draws=draws)
    if effects:
        _attach_p_value(
            effect_metric,
            effects,
            clusters,
            confirmatory=bool(base["confirmatory"]),
            seed=21103,
            alternative="negative",
            draws=draws,
        )
    actor_positive = [row for row in valid if float(row["delta_actor"]) > 0.0]
    reversal_metric = _metric(
        [float(row["delta_strategic"]) < 0.0 for row in actor_positive],
        [str(row.get("run_id") or row.get("transition_id") or index) for index, row in enumerate(actor_positive)],
        seed=21102,
        draws=draws,
    )
    independent_n = len(set(clusters))
    terminal = classify_terminal_state(
        attempted=directory is not None,
        confirmatory=bool(base["confirmatory"]),
        design_valid=bool(base["design_valid"]) and _as_bool(manifest.get("identity_blind"), default=False),
        implementation_failures=int(base["implementation_failures"]),
        n_clusters=independent_n,
        estimate=_finite(effect_metric.get("estimate")),
        ci_low=_finite(effect_metric.get("ci_low")),
        ci_high=_finite(effect_metric.get("ci_high")),
        alternative="negative",
        min_clusters=min_clusters,
    )
    base.update(
        {
            "terminal_state": terminal,
            "criterion": "strategic effect is the registered response difference; no candidate identity or outcome is an input",
            "independent_unit": "trajectory_or_task_cluster",
            "independent_n": independent_n,
            "strategic_effect": effect_metric,
            "strategic_reversal_rate": reversal_metric,
            "identity_blind": _as_bool(manifest.get("identity_blind"), default=False),
            "response_families": manifest.get("response_families", {}),
            "outcomes_used_for_response": [],
        }
    )
    return _write_result(
        Path(root).resolve(),
        artifact_name="strategic_analysis",
        title="Strategic Response Analysis",
        summary="Identity-blind adaptive-response effect and reversal diagnostics.",
        result=base,
        source_dir=directory,
    )


def analyze_pi_artifact(
    root: Path,
    *,
    confirmatory: bool | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Analyze the second-scaffold transition archive with the same estimand."""

    base, directory, rows, manifest = _analyze_optional_phase(
        Path(root).resolve(),
        phase_name="pi-replication",
        jsonl_name="autonomous_transitions.jsonl",
        csv_name="autonomous_transitions.csv",
        artifact_name="pi_analysis",
        title="Pi Replication Analysis",
        hypothesis="H6",
        confirmatory=confirmatory,
        min_clusters=min_clusters,
        draws=draws,
        alternate_jsonl_names=("pi_transition_pairs.jsonl", "pi_transitions.jsonl"),
    )
    if rows and not any(_finite(row.get("delta_proxy", row.get("proxy_delta"))) is not None for row in rows) and directory is not None:
        # Legacy Pi DEV smoke artifacts stored paired actor rows separately
        # from proxy execution rows.  Reconstruct only the identifiable proxy
        # difference from those recorded proxy successes; never infer it from
        # the actor delta itself.
        execution_rows = _read_jsonl(directory / "pi_transitions.jsonl")
        by_policy: dict[str, list[float]] = defaultdict(list)
        for execution in execution_rows:
            policy_hash = str(execution.get("policy_hash", ""))
            success = _finite(execution.get("success"))
            if policy_hash and success is not None:
                by_policy[policy_hash].append(success)
        reconstructed: list[dict[str, Any]] = []
        for row in rows:
            incumbent_hash = str(row.get("incumbent_policy_hash", row.get("incumbent_hash", "")))
            candidate_hash = str(row.get("candidate_policy_hash", row.get("candidate_hash", "")))
            incumbent_values = by_policy.get(incumbent_hash, [])
            candidate_values = by_policy.get(candidate_hash, [])
            if not incumbent_values or not candidate_values:
                continue
            merged = dict(row)
            merged["delta_proxy"] = sum(candidate_values) / len(candidate_values) - sum(incumbent_values) / len(incumbent_values)
            merged["proxy_incumbent_score"] = sum(incumbent_values) / len(incumbent_values)
            merged["proxy_candidate_score"] = sum(candidate_values) / len(candidate_values)
            merged["actor_incumbent_score"] = _finite(row.get("incumbent_success"))
            merged["actor_candidate_score"] = _finite(row.get("candidate_success"))
            merged["proxy_inferred_from_execution_archive"] = True
            reconstructed.append(merged)
        rows = reconstructed
    transition_rows = _transition_rows(rows)
    metrics = _transition_metric_bundle(transition_rows, seed=21201, draws=draws) if transition_rows else {}
    independent_n = len({str(row["cluster"]) for row in transition_rows})
    ide = metrics.get("IDE", {})
    if transition_rows and isinstance(ide, dict):
        _attach_p_value(
            ide,
            [float(row["error"]) for row in transition_rows],
            [str(row["cluster"]) for row in transition_rows],
            confirmatory=bool(base["confirmatory"]),
            seed=21202,
            alternative="positive",
            draws=draws,
        )
    terminal = classify_terminal_state(
        attempted=directory is not None,
        confirmatory=bool(base["confirmatory"]),
        design_valid=bool(base["design_valid"]) and str(manifest.get("scaffold", "Pi")) == "Pi",
        implementation_failures=int(base["implementation_failures"]),
        n_clusters=independent_n,
        estimate=_finite(ide.get("estimate")),
        ci_low=_finite(ide.get("ci_low")),
        ci_high=_finite(ide.get("ci_high")),
        alternative="nonzero",
        min_clusters=min_clusters,
    )
    base.update(
        {
            "terminal_state": terminal,
            "criterion": "the frozen transition estimand is evaluated on Pi without retuning PIVOT after primary outcomes",
            "independent_unit": "task_cluster",
            "independent_n": independent_n,
            "metrics": metrics,
            "scaffold": manifest.get("scaffold", "Pi"),
            "assessment_accessed": _as_bool(manifest.get("assessment_accessed"), default=False),
            "candidate_archive_frozen": True,
        }
    )
    return _write_result(
        Path(root).resolve(),
        artifact_name="pi_analysis",
        title="Pi Replication Analysis",
        summary="Second-scaffold transition-fidelity diagnostic with the primary estimand unchanged.",
        result=base,
        source_dir=directory,
    )


def analyze_all(
    root: Path,
    *,
    confirmatory: bool | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Run every registered analysis and emit one H1--H6 decision ledger."""

    root = Path(root).resolve()
    transition = analyze_transition_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    promotion = analyze_promotion_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    closed_loop = analyze_closed_loop_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    ablations = analyze_ablations_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    strategic = analyze_strategic_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    pi = analyze_pi_artifact(root, confirmatory=confirmatory, min_clusters=min_clusters, draws=draws)
    terminal_states = {
        "H1": transition.get("terminal_state"),
        "H2": (transition.get("hypotheses") or {}).get("H2", {}).get("terminal_state"),
        "H3": promotion.get("terminal_state"),
        "H4": ablations.get("terminal_state"),
        "H5": (transition.get("hypotheses") or {}).get("H5", {}).get("terminal_state"),
        "H6": pi.get("terminal_state"),
    }
    # Holm is included as an explicit audit field.  We only populate p-values
    # when a confirmatory analysis provides them; DEV rows remain underpowered.
    p_values: dict[str, float] = {}
    transition_hypotheses = transition.get("hypotheses", {})
    if isinstance(transition_hypotheses, Mapping):
        for name in ("H1", "H2"):
            metric = transition_hypotheses.get(name, {}).get("metric") if isinstance(transition_hypotheses.get(name), Mapping) else None
            if isinstance(metric, Mapping) and _finite(metric.get("p_value")) is not None:
                p_values[name] = float(metric["p_value"])
        h5_metric = transition_hypotheses.get("H5", {}).get("metric") if isinstance(transition_hypotheses.get("H5"), Mapping) else None
        if isinstance(h5_metric, Mapping) and _finite(h5_metric.get("p_value")) is not None:
            p_values["H5"] = float(h5_metric["p_value"])
    for name, analysis in (("H3", promotion), ("H3_closed_loop", closed_loop), ("H6", pi), ("H3_strategic_response", strategic)):
        metric_candidates = (
            analysis.get("paired_effect"),
            analysis.get("endpoint_effect"),
            analysis.get("strategic_effect"),
        )
        metric = next((candidate for candidate in metric_candidates if isinstance(candidate, Mapping)), None)
        if isinstance(metric, Mapping) and _finite(metric.get("p_value")) is not None:
            p_values[name] = float(metric["p_value"])
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        if name in {"H1", "H2"} and isinstance(transition_hypotheses, Mapping):
            hypothesis = transition_hypotheses.get(name)
            if isinstance(hypothesis, dict) and isinstance(hypothesis.get("metric"), dict):
                hypothesis["metric"]["p_value_holm"] = value
    summary = {
        "schema_version": "pivot-v15-scientific-summary-1",
        "status": "CONFIRMATORY_ANALYZED" if any(item.get("confirmatory") for item in (transition, promotion, closed_loop, ablations, strategic, pi)) else "DEV_ONLY",
        "confirmatory": any(item.get("confirmatory") for item in (transition, promotion, closed_loop, ablations, strategic, pi)),
        "hypotheses": terminal_states,
        "analysis_artifacts": {
            "transition": "artifacts/v15/transition_analysis.json",
            "promotion": "artifacts/v15/promotion_analysis.json",
            "closed_loop": "artifacts/v15/closed_loop_analysis.json",
            "ablations": "artifacts/v15/ablation_analysis.json",
            "strategic": "artifacts/v15/strategic_analysis.json",
            "pi": "artifacts/v15/pi_analysis.json",
        },
        "outcome_chasing": False,
        "multiple_testing": {"method": "Holm", "raw_p_values": p_values, "adjusted_p_values": adjusted, "status": "NO_CONFIRMATORY_P_VALUES"},
        "independent_units": {
            "transition": transition.get("independent_n", 0),
            "promotion": promotion.get("independent_n", 0),
            "closed_loop": closed_loop.get("independent_n", 0),
            "ablations": ablations.get("independent_n", 0),
            "strategic": strategic.get("independent_n", 0),
            "pi": pi.get("independent_n", 0),
        },
        "levels": {
            "modern_agent_evidence": "LEVEL_E" if not any(item.get("confirmatory") for item in (transition, promotion, closed_loop, pi)) else "PENDING_CLASSIFICATION",
            "nulls_preserved": True,
        },
    }
    path = root / "artifacts/v15/scientific_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V15 Scientific Summary",
        "",
        f"Status: **{summary['status']}**",
        "",
        "| Hypothesis | Terminal state |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {state or 'NOT_RUN'} |" for name, state in sorted(terminal_states.items()))
    lines.extend(
        [
            "",
            "All decisions use the frozen terminal-state vocabulary. Development results remain underpowered and no unfavorable outcome triggers protocol changes.",
        ]
    )
    (root / "V15_SCIENTIFIC_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def analysis_artifact_hash(path: Path) -> str:
    """Hash an analysis artifact for manifests and release provenance."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_MIN_CLUSTERS",
    "analysis_artifact_hash",
    "analyze_ablations_artifact",
    "analyze_all",
    "analyze_closed_loop_artifact",
    "analyze_pi_artifact",
    "analyze_promotion_artifact",
    "analyze_strategic_artifact",
    "analyze_transition_artifact",
    "classify_terminal_state",
    "cluster_bootstrap",
    "holm_adjust",
]
