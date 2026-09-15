"""Candidate promotion and paired high-fidelity query replay.

This module contains the decision layer of PIVOT.  Candidate generation is
deliberately outside the module: every selector consumes the same immutable
candidate rows and only receives a hidden outcome when that candidate is
queried.  The ``true_deltas`` argument is therefore an evaluator-side input
used to replay a sealed query, never a field used by candidate generation.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import content_hash, file_hash, write_jsonl, write_table

METHODS = (
    "Proxy Only",
    "Random HF",
    "Paired LUCB",
    "Global-VOI",
    "PIVOT-H",
    "PIVOT-VOI",
    "All-HF Oracle",
)


def _group_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('run_id')}::{int(row.get('round', row.get('round_index', 0)))}"


def _candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id", row.get("candidate_hash"))
    if value is None:
        raise ValueError("candidate row requires candidate_id or candidate_hash")
    return str(value)


def _proxy_delta(row: Mapping[str, Any]) -> float:
    value = row.get("proxy_delta", row.get("delta_proxy"))
    if value is None:
        raise ValueError("candidate row requires proxy_delta")
    return float(value)


def _ordered_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    return [
        sorted(group, key=lambda item: (int(item.get("candidate_index", 0)), _candidate_id(item)))
        for _, group in sorted(groups.items())
    ]


def _batch_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return content_hash([str(row.get("candidate_hash", _candidate_id(row))) for row in rows])


def _footprint_norm(row: Mapping[str, Any]) -> float:
    value = row.get("footprint", {})
    if not isinstance(value, Mapping):
        return 0.0
    total = 0.0
    for item in value.values():
        try:
            total += abs(float(item))
        except (TypeError, ValueError):
            continue
    return total


def _truth_for(row: Mapping[str, Any], true_deltas: Mapping[str, float]) -> float:
    keys = (_candidate_id(row), str(row.get("candidate_hash", "")))
    for key in keys:
        if key in true_deltas:
            return float(true_deltas[key])
    raise KeyError(f"missing sealed paired delta for candidate {_candidate_id(row)}")


@dataclass
class _Posterior:
    """Small evaluator-side posterior used only for method replay."""

    means: dict[str, float]
    scales: dict[str, float]
    observed: dict[str, float]

    def value(self, row: Mapping[str, Any]) -> float:
        return self.observed.get(_candidate_id(row), self.means[_candidate_id(row)])

    def uncertainty(self, row: Mapping[str, Any]) -> float:
        key = _candidate_id(row)
        return 0.0 if key in self.observed else self.scales[key]

    def observe(self, row: Mapping[str, Any], value: float) -> None:
        key = _candidate_id(row)
        self.observed[key] = float(value)
        self.means[key] = float(value)
        self.scales[key] = 0.0


def _posterior_regret(
    rows: Sequence[Mapping[str, Any]],
    posterior: _Posterior,
    *,
    rng: random.Random,
    samples: int = 256,
) -> float:
    """Monte-Carlo expected simple regret under the evaluator-side posterior."""

    if samples <= 0:
        raise ValueError("posterior samples must be positive")
    if not rows:
        return 0.0
    means = [posterior.value(row) for row in rows]
    scales = [posterior.uncertainty(row) for row in rows]
    selected = max(range(len(rows)), key=lambda index: (means[index], -int(rows[index].get("candidate_index", index))))
    regrets: list[float] = []
    for _ in range(samples):
        draw = [rng.gauss(mean, scale) if scale > 0.0 else mean for mean, scale in zip(means, scales)]
        regrets.append(max(draw) - draw[selected])
    return max(0.0, sum(regrets) / len(regrets))


def expected_evsi(
    rows: Sequence[Mapping[str, Any]],
    posterior: _Posterior,
    candidate: Mapping[str, Any],
    *,
    seed: int = 0,
    fantasies: int = 128,
    posterior_samples: int = 256,
) -> float:
    """Estimate pre-query EVSI as expected simple-regret reduction.

    The unknown paired delta is modeled with the current independent normal
    arm posterior.  This is deliberately a transparent replay posterior; a
    calibrated feature posterior can replace it without changing the ledger
    contract.  Crucially, this function uses only pre-query proxy/footprint
    state and never reads a sealed outcome.
    """

    if fantasies <= 0 or posterior_samples <= 0:
        raise ValueError("fantasies and posterior_samples must be positive")
    key = _candidate_id(candidate)
    if key in posterior.observed:
        return 0.0
    rng = random.Random(seed)
    current = _posterior_regret(rows, posterior, rng=rng, samples=posterior_samples)
    mean = posterior.value(candidate)
    scale = posterior.uncertainty(candidate)
    if scale <= 0.0:
        return 0.0
    post_regrets: list[float] = []
    for _ in range(fantasies):
        fantasy = rng.gauss(mean, scale)
        updated = _Posterior(dict(posterior.means), dict(posterior.scales), dict(posterior.observed))
        updated.observe(candidate, fantasy)
        post_regrets.append(_posterior_regret(rows, updated, rng=rng, samples=posterior_samples))
    return max(0.0, current - sum(post_regrets) / len(post_regrets))


def _initial_posterior(rows: Sequence[Mapping[str, Any]]) -> _Posterior:
    means = {_candidate_id(row): _proxy_delta(row) for row in rows}
    # Footprint is available before a sealed query.  The scale is a heuristic
    # acquisition signal, not a claim of calibrated predictive uncertainty.
    scales = {
        _candidate_id(row): 0.05 + 0.05 * min(_footprint_norm(row), 10.0)
        for row in rows
    }
    return _Posterior(means=means, scales=scales, observed={})


def _select_max(rows: Sequence[Mapping[str, Any]], posterior: _Posterior) -> Mapping[str, Any]:
    return max(rows, key=lambda row: (posterior.value(row), -int(row.get("candidate_index", 0))))


def _query_order(
    method: str,
    rows: Sequence[Mapping[str, Any]],
    posterior: _Posterior,
    budget: int,
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    if budget <= 0 or method == "Proxy Only":
        return []
    if method == "All-HF Oracle":
        return list(rows)
    available = list(rows)
    if method == "Random HF":
        rng.shuffle(available)
        return available[: min(budget, len(available))]
    if method == "Paired LUCB":
        # A paired LUCB-style arm schedule: alternate the empirical leader and
        # the strongest challenger so every observation remains paired.
        leader = _select_max(available, posterior)
        challenger_pool = [row for row in available if _candidate_id(row) != _candidate_id(leader)]
        challenger = max(
            challenger_pool,
            key=lambda row: (posterior.value(row) + posterior.uncertainty(row), -int(row.get("candidate_index", 0))),
            default=None,
        )
        ordered = [leader] + ([challenger] if challenger is not None else [])
        ordered.extend(
            sorted(
                [row for row in available if _candidate_id(row) not in {_candidate_id(item) for item in ordered}],
                key=lambda row: (-posterior.value(row), int(row.get("candidate_index", 0))),
            )
        )
        return ordered[: min(budget, len(ordered))]
    if method == "Global-VOI":
        # Global value acquisition uses only proxy distance from the current
        # leader; it intentionally ignores update footprint.
        leader_value = max(posterior.value(row) for row in available)
        ranked = sorted(
            available,
            key=lambda row: (abs(posterior.value(row) - leader_value) + posterior.uncertainty(row), -int(row.get("candidate_index", 0))),
            reverse=True,
        )
        return ranked[: min(budget, len(ranked))]
    if method == "PIVOT-H":
        ranked = sorted(
            available,
            key=lambda row: (posterior.uncertainty(row), -int(row.get("candidate_index", 0))),
            reverse=True,
        )
        return ranked[: min(budget, len(ranked))]
    if method == "PIVOT-VOI":
        # Compute an explicit expected-regret reduction before ranking.  The
        # footprint tie-break keeps the acquisition interpretable when two
        # candidates have indistinguishable Monte-Carlo scores.
        evsi = {
            _candidate_id(row): expected_evsi(
                available,
                posterior,
                row,
                seed=rng.randrange(2**31),
                fantasies=32,
                posterior_samples=64,
            )
            for row in available
        }
        ranked = sorted(
            available,
            key=lambda row: (
                evsi[_candidate_id(row)],
                _footprint_norm(row),
                posterior.uncertainty(row),
                -int(row.get("candidate_index", 0)),
            ),
            reverse=True,
        )
        return ranked[: min(budget, len(ranked))]
    raise ValueError(f"unknown promotion method: {method}")


def _replay_group(
    rows: Sequence[Mapping[str, Any]],
    true_deltas: Mapping[str, float],
    *,
    budget: int,
    method: str,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("candidate batch must not be empty")
    posterior = _initial_posterior(rows)
    batch_hash = _batch_hash(rows)
    truth = {_candidate_id(row): _truth_for(row, true_deltas) for row in rows}
    true_best = max(rows, key=lambda row: (truth[_candidate_id(row)], -int(row.get("candidate_index", 0))))
    query_rows = _query_order(method, rows, posterior, budget, rng)
    query_ledger: list[dict[str, Any]] = []
    for query_index, row in enumerate(query_rows):
        key = _candidate_id(row)
        before = posterior.value(row)
        evsi = expected_evsi(
            rows,
            posterior,
            row,
            seed=rng.randrange(2**31),
            fantasies=128,
            posterior_samples=256,
        )
        observed = truth[key]
        posterior.observe(row, observed)
        query_ledger.append(
            {
                "phase": "DEV",
                "method": method,
                "run_id": str(row.get("run_id")),
                "round": int(row.get("round", row.get("round_index", 0))),
                "candidate_id": key,
                "candidate_hash": str(row.get("candidate_hash", key)),
                "query_index": query_index,
                "task_id": f"{_group_key(row)}::{key}",
                "paired_delta": observed,
                "cost": 1.0,
                "posterior_before": before,
                "posterior_after": observed,
                "EVSI": evsi,
                "observed_information_gain": abs(observed - before),
                "candidate_batch_hash": batch_hash,
                "logical_hf_query": True,
                "physical_pair_evaluation": True,
                "cache_hit": False,
                "outcome_chasing": False,
            }
        )
    selected = _select_max(rows, posterior)
    selected_key = _candidate_id(selected)
    max_true = truth[_candidate_id(true_best)]
    result = {
        "phase": "DEV",
        "method": method,
        "run_id": str(selected.get("run_id")),
        "round": int(selected.get("round", selected.get("round_index", 0))),
        "hf_budget": int(budget),
        "selected_candidate": selected_key,
        "selected_candidate_hash": str(selected.get("candidate_hash", selected_key)),
        "true_best_candidate": _candidate_id(true_best),
        "true_best_candidate_hash": str(true_best.get("candidate_hash", _candidate_id(true_best))),
        "ISR": max_true - truth[selected_key],
        "hf_cost": float(len(query_rows)),
        "candidate_count": len(rows),
        "candidate_batch_hash": batch_hash,
        "outcome_chasing": False,
    }
    return result, query_ledger


def replay_methods(
    candidate_rows: Sequence[Mapping[str, Any]],
    true_deltas: Mapping[str, float],
    *,
    budgets: Sequence[int] = (1, 2, 4),
    seed: int = 101,
) -> dict[str, list[dict[str, Any]]]:
    """Replay registered promotion selectors on one immutable archive.

    ``true_deltas`` represents the sealed evaluator response.  It is looked up
    only for selected HF queries; it is never copied into candidate rows or
    used by the proxy-only method.  The returned dictionary contains a result
    row and a query ledger for every method, budget, and candidate batch.
    """

    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")
    clean_budgets = tuple(sorted({int(value) for value in budgets}))
    if not clean_budgets or any(value < 0 for value in clean_budgets):
        raise ValueError("budgets must contain non-negative integers")
    results: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for group_index, group in enumerate(_ordered_groups(candidate_rows)):
        for budget in clean_budgets:
            for method_index, method in enumerate(METHODS):
                # Derive independent deterministic streams so adding a method
                # does not alter another method's random query schedule.
                rng = random.Random(seed + group_index * 100_003 + budget * 1_009 + method_index)
                result, ledger = _replay_group(
                    group,
                    true_deltas,
                    budget=budget,
                    method=method,
                    rng=rng,
                )
                results.append(result)
                queries.extend(ledger)
    return {"promotion_results": results, "hf_queries": queries}


def write_promotion_artifacts(
    result: Mapping[str, Sequence[Mapping[str, Any]]],
    output: Path,
    *,
    phase: str = "DEV",
    candidate_archive: Path | None = None,
    note: str = "paired promotion replay",
) -> dict[str, Any]:
    """Persist canonical promotion and HF query tables with a manifest."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    promotion_rows = [dict(row) for row in result.get("promotion_results", [])]
    query_rows = [dict(row) for row in result.get("hf_queries", [])]
    write_jsonl(promotion_rows, output / "promotion_results.jsonl")
    write_jsonl(query_rows, output / "hf_queries.jsonl")
    write_table(
        promotion_rows,
        output / "promotion_results",
        columns=(
            "phase",
            "method",
            "run_id",
            "round",
            "hf_budget",
            "selected_candidate",
            "true_best_candidate",
            "ISR",
            "hf_cost",
            "candidate_count",
            "candidate_batch_hash",
        ),
    )
    write_table(
        query_rows,
        output / "hf_queries",
        columns=(
            "phase",
            "method",
            "run_id",
            "round",
            "candidate_id",
            "candidate_hash",
            "query_index",
            "task_id",
            "paired_delta",
            "cost",
            "posterior_before",
            "posterior_after",
            "EVSI",
            "observed_information_gain",
            "candidate_batch_hash",
            "logical_hf_query",
            "physical_pair_evaluation",
            "cache_hit",
        ),
    )
    manifest = {
        "schema_version": "pivot-v15-promotion-replay-1",
        "phase": phase,
        "confirmatory": phase.upper() == "CONFIRMATORY",
        "candidate_archive_sha256": file_hash(candidate_archive) if candidate_archive and candidate_archive.is_file() else None,
        "promotion_result_count": len(promotion_rows),
        "hf_query_count": len(query_rows),
        "methods": sorted({str(row.get("method")) for row in promotion_rows}),
        "budgets": sorted({int(row.get("hf_budget", 0)) for row in promotion_rows}),
        "outcome_chasing": False,
        "note": note,
    }
    (output / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = ["METHODS", "expected_evsi", "replay_methods", "write_promotion_artifacts"]
