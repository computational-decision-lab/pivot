from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.finance_backtest.acquisition import load_public_finance_manifest
from pivot.environments.finance_backtest.public_data import (
    align_klines_to_depth_coverage,
    calibrate_depth_proxy,
    finance_config_from_public_data,
    load_binance_book_depth_archive,
    load_binance_kline_archive,
)
from pivot.environments.finance_backtest.world import (
    ExecutionReplayWorld,
    HistoricalBacktestWorld,
    PublicDepthExecutionWorld,
)
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.evaluation.paired import PairedEvaluator
from pivot.evaluation.uncertainty import bootstrap_mean_ci


def run_public_finance_calibration(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Run paired F0/F1/F2 evaluation over checksum-pinned public sessions."""

    config_path = Path(config_path).resolve()
    raw_config = config_path.read_bytes()
    payload = _mapping(yaml.safe_load(raw_config), "public finance config")
    manifest_path = _resolve(config_path, _required_str(payload, "dataset_manifest"))
    data_root = _resolve(config_path, _required_str(payload, "data_root"))
    manifest = load_public_finance_manifest(manifest_path)
    cache = data_root / manifest.dataset_id
    execution = _mapping(payload.get("execution", {}), "execution")
    incumbent = Policy.from_mapping(_numeric_mapping(payload.get("incumbent"), "incumbent"))
    candidate = Policy.from_mapping(_numeric_mapping(payload.get("candidate"), "candidate"))
    edit_type = _required_str(payload, "edit_type")
    transition = PolicyTransition(
        incumbent=incumbent,
        candidate=candidate,
        round_id=0,
        candidate_index=0,
        improvement_operator="typed_public_finance_edit",
        edit_type=edit_type,
        proxy_world_id="F0-public-observer",
        high_fidelity_world_id="F2-public-depth-proxy",
        config_id=manifest.dataset_id,
    )
    participation_rates = _positive_axis(
        payload.get("participation_rates"), "participation_rates", allow_zero=True
    )
    impact_multipliers = _positive_axis(
        payload.get("impact_multipliers", [1.0]), "impact_multipliers"
    )
    target_participation = float(payload.get("target_participation", 0.01))
    primary_multiplier = float(payload.get("primary_impact_multiplier", 1.0))
    recovery_floor = float(payload.get("recovery_floor", 1e-3))
    if not 0 < recovery_floor <= 1:
        raise ValueError("recovery_floor must be in (0, 1]")

    rows: list[dict[str, object]] = []
    calibrations: list[dict[str, object]] = []
    archive_records: list[dict[str, str]] = []
    for session_index, session in enumerate(manifest.sessions):
        source_klines = load_binance_kline_archive(
            cache / session.klines.filename,
            expected_sha256=session.klines.sha256,
            dataset_id=session.session_id,
            symbol=manifest.symbol,
            source_url=session.klines.url,
        )
        depth = load_binance_book_depth_archive(
            cache / session.book_depth.filename,
            expected_sha256=session.book_depth.sha256,
            dataset_id=session.session_id,
            symbol=manifest.symbol,
            source_url=session.book_depth.url,
        )
        _validate_session_dates(
            session.date,
            source_klines.bars[0].open_time_us,
            source_klines.bars[-1].open_time_us,
        )
        klines = align_klines_to_depth_coverage(source_klines, depth)
        calibration = calibrate_depth_proxy(klines, depth)
        applied_recovery = max(recovery_floor, calibration.descriptive_recovery_proxy)
        calibration_record = asdict(calibration)
        calibration_record.update(
            {
                "session_date": session.date,
                "source_n_kline_bars": len(source_klines.bars),
                "aligned_n_kline_bars": len(klines.bars),
                "dropped_kline_bars": len(source_klines.bars) - len(klines.bars),
                "applied_liquidity_recovery": applied_recovery,
                "kline_sha256": klines.archive_sha256,
                "book_depth_sha256": depth.archive_sha256,
            }
        )
        calibrations.append(calibration_record)
        archive_records.extend(
            [
                {
                    "session_id": session.session_id,
                    "kind": "klines",
                    "sha256": klines.archive_sha256,
                    "source_url": klines.source_url,
                },
                {
                    "session_id": session.session_id,
                    "kind": "book_depth",
                    "sha256": depth.archive_sha256,
                    "source_url": depth.source_url,
                },
            ]
        )
        finance = finance_config_from_public_data(
            klines,
            spread_bps=float(execution.get("spread_bps", 2.0)),
            fee_bps=float(execution.get("fee_bps", 1.0)),
            slippage_bps=float(execution.get("slippage_bps", 3.0)),
            partial_fill_ratio=float(execution.get("partial_fill_ratio", 0.75)),
            queue_depth=float(execution.get("queue_depth", 0.5)),
            strategy_frequency=str(execution.get("strategy_frequency", "bar")),
            simulation_frequency=str(execution.get("simulation_frequency", "event")),
        )
        context = [
            RolloutContext(
                seed=_session_seed(session.session_id, session_index),
                scenario_id=session.session_id,
                metadata={"session_date": session.date, "dataset_id": manifest.dataset_id},
            )
        ]
        delta_f0 = PairedEvaluator(HistoricalBacktestWorld(finance)).evaluate(
            transition, context
        ).delta
        delta_f1 = PairedEvaluator(ExecutionReplayWorld(finance)).evaluate(
            transition, context
        ).delta
        for multiplier in impact_multipliers:
            applied_impact = calibration.impact_coefficient * multiplier
            for participation in participation_rates:
                interactive = InteractiveMarketWorld(
                    InteractiveMarketConfig(
                        finance=finance,
                        participation_rate=participation,
                        impact_coefficient=applied_impact,
                        liquidity_recovery=applied_recovery,
                    )
                )
                depth_world = PublicDepthExecutionWorld(
                    klines,
                    depth,
                    finance,
                    participation_rate=participation,
                    liquidity_recovery=applied_recovery,
                )
                delta_f2 = PairedEvaluator(interactive, mode="actor").evaluate(
                    transition, context
                ).delta
                delta_f2_depth = PairedEvaluator(depth_world, mode="actor").evaluate(
                    transition, context
                ).delta
                rows.append(
                    {
                        "dataset_id": manifest.dataset_id,
                        "session_id": session.session_id,
                        "session_date": session.date,
                        "transition_id": transition.transition_id,
                        "transition_edit_type": edit_type,
                        "participation_rate": participation,
                        "impact_multiplier": multiplier,
                        "depth_proxy_impact_coefficient": calibration.impact_coefficient,
                        "applied_impact_coefficient": applied_impact,
                        "descriptive_recovery_proxy": calibration.descriptive_recovery_proxy,
                        "applied_liquidity_recovery": applied_recovery,
                        "delta_f0": delta_f0,
                        "delta_f1": delta_f1,
                        "delta_f2": delta_f2,
                        "delta_f2_depth": delta_f2_depth,
                        "execution_effect": delta_f1 - delta_f0,
                        "mechanical_effect": delta_f2 - delta_f1,
                        "depth_mechanical_effect": delta_f2_depth - delta_f1,
                        "improvement_reversal_f0_to_f2": delta_f0 > 0 and delta_f2 < 0,
                        "improvement_reversal_f1_to_f2": delta_f1 > 0 and delta_f2 < 0,
                        "depth_reversal_f1_to_f2": (
                            delta_f1 > 0 and delta_f2_depth < 0
                        ),
                        "paired": True,
                        "virtual_fills": True,
                        "causal_impact_identified": False,
                    }
                )

    summary = _summarize(
        rows,
        calibrations,
        target_participation=target_participation,
        primary_multiplier=primary_multiplier,
        n_sessions=len(manifest.sessions),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "public_finance_rows.json", rows)
    _write_json(output_dir / "depth_proxy_calibration.json", calibrations)
    _write_json(output_dir / "summary.json", summary)
    provenance = {
        "dataset_id": manifest.dataset_id,
        "provider": manifest.provider,
        "license": manifest.license,
        "market": manifest.market,
        "symbol": manifest.symbol,
        "interval": manifest.interval,
        "manifest_path": str(manifest.manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "transition_id": transition.transition_id,
        "archives": archive_records,
        "paired": True,
        "virtual_fills": True,
        "live_orders": False,
        "ground_truth_for_endogenous_response": False,
        "causal_impact_identified": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "provenance.json", provenance)
    return summary


def _summarize(
    rows: list[dict[str, object]],
    calibrations: list[dict[str, object]],
    *,
    target_participation: float,
    primary_multiplier: float,
    n_sessions: int,
) -> dict[str, object]:
    zero_effects = [
        _row_float(row, "delta_f2") - _row_float(row, "delta_f1")
        for row in rows
        if math.isclose(_row_float(row, "participation_rate"), 0.0, abs_tol=1e-15)
    ]
    zero_depth_effects = [
        _row_float(row, "delta_f2_depth") - _row_float(row, "delta_f1")
        for row in rows
        if math.isclose(_row_float(row, "participation_rate"), 0.0, abs_tol=1e-15)
    ]
    target_rows = [
        row
        for row in rows
        if math.isclose(_row_float(row, "participation_rate"), target_participation, abs_tol=1e-15)
        and math.isclose(_row_float(row, "impact_multiplier"), primary_multiplier, abs_tol=1e-15)
    ]
    if not zero_effects or not target_rows:
        raise ValueError("public calibration requires zero and target participation rows")
    target_effects = [
        _row_float(row, "delta_f2") - _row_float(row, "delta_f1") for row in target_rows
    ]
    target_depth_effects = [
        _row_float(row, "delta_f2_depth") - _row_float(row, "delta_f1")
        for row in target_rows
    ]
    ci_low, ci_high = bootstrap_mean_ci(target_effects, seed=20260819)
    depth_ci_low, depth_ci_high = bootstrap_mean_ci(target_depth_effects, seed=20260820)
    f1_positive = [row for row in target_rows if _row_float(row, "delta_f1") > 0]
    f0_positive = [row for row in target_rows if _row_float(row, "delta_f0") > 0]
    f0_to_f1_reversals = [
        row for row in f0_positive if _row_float(row, "delta_f1") < 0
    ]
    endogenous_reversals = [
        row for row in f1_positive if bool(row["improvement_reversal_f1_to_f2"])
    ]
    depth_endogenous_reversals = [
        row for row in f1_positive if bool(row["depth_reversal_f1_to_f2"])
    ]
    impacts = [_row_float(record, "impact_coefficient") for record in calibrations]
    recoveries = [_row_float(record, "descriptive_recovery_proxy") for record in calibrations]
    participation_diagnostics = _participation_diagnostics(rows, primary_multiplier)
    depth_reversal_participations = [
        _row_float(record, "participation_rate")
        for record in participation_diagnostics
        if _row_float(record, "n_depth_reversal_sessions") > 0
    ]
    return {
        "n_sessions": n_sessions,
        "n_rows": len(rows),
        "target_participation": target_participation,
        "primary_impact_multiplier": primary_multiplier,
        "zero_participation_max_abs_f2_minus_f1": max(abs(value) for value in zero_effects),
        "zero_participation_max_abs_f2_depth_minus_f1": max(
            abs(value) for value in zero_depth_effects
        ),
        "target_f2_minus_f1_mean": mean(target_effects),
        "target_f2_minus_f1_ci_low": ci_low,
        "target_f2_minus_f1_ci_high": ci_high,
        "target_f2_depth_minus_f1_mean": mean(target_depth_effects),
        "target_f2_depth_minus_f1_ci_low": depth_ci_low,
        "target_f2_depth_minus_f1_ci_high": depth_ci_high,
        "f1_to_f2_reversal_rate_given_f1_positive": _conditional_rate(
            f1_positive, "improvement_reversal_f1_to_f2"
        ),
        "f0_to_f2_reversal_rate_given_f0_positive": _conditional_rate(
            f0_positive, "improvement_reversal_f0_to_f2"
        ),
        "f0_to_f1_reversal_rate_given_f0_positive": (
            len(f0_to_f1_reversals) / len(f0_positive) if f0_positive else None
        ),
        "n_f0_positive_sessions": len(f0_positive),
        "n_f1_positive_sessions": len(f1_positive),
        "n_endogenous_reversal_sessions": len(endogenous_reversals),
        "endogenous_reversal_observed": bool(endogenous_reversals),
        "n_depth_endogenous_reversal_sessions": len(depth_endogenous_reversals),
        "depth_endogenous_reversal_observed": bool(depth_endogenous_reversals),
        "depth_reversal_observed_any_participation": bool(depth_reversal_participations),
        "first_depth_reversal_participation": (
            min(depth_reversal_participations) if depth_reversal_participations else None
        ),
        "participation_diagnostics": participation_diagnostics,
        "impact_coefficient_median": median(impacts),
        "impact_coefficient_min": min(impacts),
        "impact_coefficient_max": max(impacts),
        "descriptive_recovery_proxy_median": median(recoveries),
        "calibration_scope": "observed paths plus linearized one-percent depth proxy",
        "causal_impact_identified": False,
        "gate_e_promoted": False,
        "status": "partial_observational_calibration",
    }


def _conditional_rate(rows: list[dict[str, object]], key: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[key]) for row in rows) / len(rows)


def _participation_diagnostics(
    rows: list[dict[str, object]], primary_multiplier: float
) -> list[dict[str, object]]:
    primary = [
        row
        for row in rows
        if math.isclose(_row_float(row, "impact_multiplier"), primary_multiplier, abs_tol=1e-15)
    ]
    participations = sorted({_row_float(row, "participation_rate") for row in primary})
    diagnostics: list[dict[str, object]] = []
    for participation in participations:
        group = [
            row
            for row in primary
            if math.isclose(
                _row_float(row, "participation_rate"), participation, abs_tol=1e-15
            )
        ]
        f1_positive = [row for row in group if _row_float(row, "delta_f1") > 0]
        linear_reversals = [
            row for row in f1_positive if bool(row["improvement_reversal_f1_to_f2"])
        ]
        depth_reversals = [
            row for row in f1_positive if bool(row["depth_reversal_f1_to_f2"])
        ]
        diagnostics.append(
            {
                "participation_rate": participation,
                "n_sessions": len(group),
                "n_f1_positive_sessions": len(f1_positive),
                "linear_mechanical_effect_mean": mean(
                    _row_float(row, "mechanical_effect") for row in group
                ),
                "depth_mechanical_effect_mean": mean(
                    _row_float(row, "depth_mechanical_effect") for row in group
                ),
                "n_linear_reversal_sessions": len(linear_reversals),
                "n_depth_reversal_sessions": len(depth_reversals),
                "linear_reversal_rate_given_f1_positive": (
                    len(linear_reversals) / len(f1_positive) if f1_positive else None
                ),
                "depth_reversal_rate_given_f1_positive": (
                    len(depth_reversals) / len(f1_positive) if f1_positive else None
                ),
            }
        )
    return diagnostics


def _row_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _validate_session_dates(expected: str, first_us: int, last_us: int) -> None:
    first = datetime.fromtimestamp(first_us / 1_000_000, tz=timezone.utc).date().isoformat()
    last = datetime.fromtimestamp(last_us / 1_000_000, tz=timezone.utc).date().isoformat()
    if first != expected or last != expected:
        raise ValueError(
            f"kline session dates {first}..{last} do not match manifest date {expected}"
        )


def _session_seed(session_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{session_id}:{index}".encode()).hexdigest()
    return int(digest[:8], 16)


def _positive_axis(value: object, name: str, *, allow_zero: bool = False) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = [float(item) for item in value]
    minimum = 0.0 if allow_zero else 1e-300
    if any(item < minimum or not math.isfinite(item) for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be finite and {qualifier}")
    return values


def _numeric_mapping(value: object, name: str) -> dict[str, float]:
    payload = _mapping(value, name)
    if not payload:
        raise ValueError(f"{name} must not be empty")
    return {str(key): float(item) for key, item in payload.items()}


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
