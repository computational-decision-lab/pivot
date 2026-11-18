from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from pivot.evaluation.uncertainty import bootstrap_mean_ci

from .public_finance import run_public_finance_calibration


def aggregate_public_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_sessions: Sequence[tuple[str, str]],
    primary_participation: float,
    primary_impact_multiplier: float,
    holdout_dates: Sequence[str],
) -> dict[str, object]:
    """Aggregate predeclared public sessions without outcome-based selection."""

    expected = set(expected_sessions)
    primary: dict[tuple[str, str], dict[str, object]] = {}
    for source_row in rows:
        row = dict(source_row)
        if row.get("causal_impact_identified") is True:
            raise ValueError("public expansion rows cannot claim causal impact")
        if row.get("live_orders") is True:
            raise ValueError("public expansion rows cannot contain live orders")
        asset = _text(row, "asset")
        date = _text(row, "session_date")
        if math.isclose(
            _number(row, "participation_rate"), primary_participation, abs_tol=1e-15
        ) and math.isclose(
            _number(row, "impact_multiplier"), primary_impact_multiplier, abs_tol=1e-15
        ):
            key = (asset, date)
            if key in primary:
                raise ValueError(f"duplicate primary session: {asset}/{date}")
            primary[key] = row

    observed = set(primary)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    ordered = [primary[key] for key in sorted(observed & expected)]
    holdout_set = set(holdout_dates)
    summary = _group_metrics(ordered, seed=20260819)
    summary["n_expected_sessions"] = len(expected)
    summary["n_primary_sessions"] = len(ordered)
    summary["complete_grid"] = not missing and not unexpected and len(ordered) == len(expected)
    summary["missing_sessions"] = [[asset, date] for asset, date in missing]
    summary["unexpected_sessions"] = [[asset, date] for asset, date in unexpected]
    summary["holdout"] = _group_metrics(
        [row for row in ordered if _text(row, "session_date") in holdout_set], seed=20260820
    )

    by_asset: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ordered:
        by_asset[_text(row, "asset")].append(row)
        by_date[_text(row, "session_date")].append(row)
    summary["per_asset"] = {
        asset: _group_metrics(group, seed=20260830 + index)
        for index, (asset, group) in enumerate(sorted(by_asset.items()))
    }
    summary["per_calendar_date"] = {
        date: _group_metrics(group, seed=20260840 + index)
        for index, (date, group) in enumerate(sorted(by_date.items()))
    }
    summary.update(
        {
            "primary_participation": primary_participation,
            "primary_impact_multiplier": primary_impact_multiplier,
            "causal_impact_identified": False,
            "ground_truth_for_endogenous_response": False,
            "gate_e_promoted": False,
            "live_orders": False,
            "selection_rule": "include_every_predeclared_asset_date_pair",
        }
    )
    return summary


def run_public_finance_expansion(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Run the frozen multi-asset public audit and persist an immutable ledger."""

    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite public expansion evidence: {output_dir}")
    raw_config = config_path.read_bytes()
    config = _mapping(yaml.safe_load(raw_config), "expansion config")
    grid_path = _resolve(config_path, _required_str(config, "grid_config"))
    grid_raw = grid_path.read_bytes()
    grid = _mapping(yaml.safe_load(grid_raw), "expansion grid")
    symbols = _string_list(grid.get("symbols"), "grid.symbols")
    dates = _string_list(grid.get("dates"), "grid.dates")
    roles = _normalized_roles(_mapping(grid.get("audit_roles"), "grid.audit_roles"))
    holdout_dates = tuple(
        date for date in dates if str(roles.get(date, "")).startswith("frozen_holdout")
    )
    primary_participation = _number(config, "primary_participation")
    primary_multiplier = _number(config, "primary_impact_multiplier")
    evaluation = _mapping(grid.get("evaluation"), "grid.evaluation")
    if not math.isclose(
        primary_participation, _number(evaluation, "primary_participation")
    ):
        raise ValueError("expansion primary participation differs from frozen grid")
    if not math.isclose(
        primary_multiplier, _number(evaluation, "primary_impact_multiplier")
    ):
        raise ValueError("expansion primary impact multiplier differs from frozen grid")
    run_specs = config.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise ValueError("expansion config runs must be a non-empty list")
    specs_by_asset: dict[str, Path] = {}
    for index, raw_spec in enumerate(run_specs):
        spec = _mapping(raw_spec, f"runs[{index}]")
        asset = _required_str(spec, "asset")
        if asset in specs_by_asset:
            raise ValueError(f"duplicate expansion asset: {asset}")
        if asset not in symbols:
            raise ValueError(f"run asset not in frozen grid: {asset}")
        specs_by_asset[asset] = _resolve(config_path, _required_str(spec, "config"))
    if set(specs_by_asset) != set(symbols):
        raise ValueError("expansion configs must cover every frozen grid symbol exactly once")

    expected_sessions = tuple((asset, date) for asset in symbols for date in dates)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    subrun_summaries: dict[str, dict[str, object]] = {}
    manifest_hashes: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    for asset in symbols:
        subconfig = specs_by_asset[asset]
        try:
            _validate_subconfig(
                subconfig,
                expected_asset=asset,
                grid=grid,
                primary_participation=primary_participation,
                primary_multiplier=primary_multiplier,
            )
            suboutput = output_dir / asset.lower()
            subrun_summaries[asset] = dict(run_public_finance_calibration(subconfig, suboutput))
            provenance = _read_mapping(suboutput / "provenance.json")
            manifest_hashes[asset] = _required_str(provenance, "manifest_sha256")
            config_hashes[asset] = _required_str(provenance, "config_sha256")
            raw_rows = _read_rows(suboutput / "public_finance_rows.json")
            for row in raw_rows:
                row["asset"] = asset
                all_rows.append(row)
        except Exception as error:  # noqa: BLE001 - every asset failure must be logged
            failures.append({"asset": asset, "error": f"{type(error).__name__}: {error}"})

    summary = aggregate_public_rows(
        all_rows,
        expected_sessions=expected_sessions,
        primary_participation=primary_participation,
        primary_impact_multiplier=primary_multiplier,
        holdout_dates=holdout_dates,
    )
    summary.update(
        {
            "grid_id": _required_str(grid, "grid_id"),
            "n_assets": len(symbols),
            "n_dates": len(dates),
            "n_rows": len(all_rows),
            "n_failed_assets": len(failures),
            "complete_grid": bool(summary["complete_grid"]) and not failures,
            "status": "complete_observational_expansion" if not failures else "incomplete_observational_expansion",
        }
    )
    _write_json(output_dir / "public_expansion_rows.json", all_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "failure_ledger.json", failures)
    _write_json(
        output_dir / "provenance.json",
        {
            "grid_config": str(grid_path),
            "grid_sha256": hashlib.sha256(grid_raw).hexdigest(),
            "expansion_config": str(config_path),
            "expansion_config_sha256": hashlib.sha256(raw_config).hexdigest(),
            "asset_config_hashes": config_hashes,
            "manifest_hashes": manifest_hashes,
            "assets": symbols,
            "dates": dates,
            "holdout_dates": list(holdout_dates),
            "selection_rule": "include_every_predeclared_asset_date_pair",
            "causal_impact_identified": False,
            "live_orders": False,
            "subrun_summaries": subrun_summaries,
            "failures": failures,
        },
    )
    return summary


def _group_metrics(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
    effects = [_number(row, "depth_mechanical_effect") for row in rows]
    positive = [row for row in rows if _number(row, "delta_f1") > 0]
    reversals = [row for row in positive if bool(row.get("depth_reversal_f1_to_f2"))]
    return {
        "n_primary_sessions": len(rows),
        "n_f1_positive_sessions": len(positive),
        "n_depth_reversal_sessions": len(reversals),
        "depth_reversal_rate_given_f1_positive": (
            len(reversals) / len(positive) if positive else None
        ),
        "pooled_depth_mechanical_effect": _bootstrap_metric(effects, seed=seed),
    }


def _bootstrap_metric(values: Sequence[float], *, seed: int) -> dict[str, object] | None:
    if not values:
        return None
    low, high = bootstrap_mean_ci(values, seed=seed)
    return {"estimate": mean(values), "ci_low": low, "ci_high": high, "n": len(values)}


def _validate_subconfig(
    path: Path,
    *,
    expected_asset: str,
    grid: Mapping[str, Any],
    primary_participation: float,
    primary_multiplier: float,
) -> None:
    payload = _mapping(yaml.safe_load(path.read_bytes()), f"subconfig {path}")
    if _required_str(payload, "edit_type") != _required_str(
        _mapping(grid.get("update"), "grid.update"), "edit_type"
    ):
        raise ValueError(f"{path} edit_type differs from frozen grid")
    if _canonical(payload.get("incumbent")) != _canonical(
        _mapping(_mapping(grid.get("update"), "grid.update").get("incumbent"), "grid.incumbent")
    ):
        raise ValueError(f"{path} incumbent differs from frozen grid")
    if _canonical(payload.get("candidate")) != _canonical(
        _mapping(_mapping(grid.get("update"), "grid.update").get("candidate"), "grid.candidate")
    ):
        raise ValueError(f"{path} candidate differs from frozen grid")
    if not math.isclose(_number(payload, "target_participation"), primary_participation):
        raise ValueError(f"{path} target participation differs from frozen grid")
    if not math.isclose(_number(payload, "primary_impact_multiplier"), primary_multiplier):
        raise ValueError(f"{path} impact multiplier differs from frozen grid")
    evaluation = _mapping(grid.get("evaluation"), "grid.evaluation")
    for key in ("participation_rates", "impact_multipliers", "execution"):
        if _canonical(payload.get(key)) != _canonical(evaluation.get(key)):
            raise ValueError(f"{path} {key} differs from frozen grid")
    manifest_path = _resolve(path, _required_str(payload, "dataset_manifest"))
    manifest = _read_mapping(manifest_path)
    if _required_str(manifest, "symbol") != expected_asset:
        raise ValueError(f"{path} manifest symbol differs from run asset")
    expected_dates = set(_string_list(grid.get("dates"), "grid.dates"))
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or {
        _required_str(_mapping(session, "session"), "date") for session in sessions
    } != expected_dates:
        raise ValueError(f"{path} manifest dates differ from frozen grid")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a list of rows in {path}")
    return [dict(row) for row in payload]


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_bytes())
    return _mapping(payload, str(path))


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty string list")
    return [str(item) for item in value]


def _normalized_roles(payload: Mapping[Any, Any]) -> dict[str, str]:
    """Normalize YAML date keys, which PyYAML may load as date objects."""

    return {str(key): str(value) for key, value in payload.items()}


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
