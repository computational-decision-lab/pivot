from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pivot.v9.artifacts import build_manifest, read_jsonl_gz, write_json, write_provenance
from pivot.v9.evaluators import evaluate_ood
from pivot.v9.statistics import bootstrap_mean_ci

from .common import load_yaml, setup_output, write_decision


def run(config_path: Path, *, profile: dict[str, Any], output: Path, root: Path, resume: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    setup_output(output, resume=resume)
    source = root / str(config["source_e2c"]) / "transition_rows.jsonl.gz"
    if not source.is_file():
        decision = write_decision(
            output,
            experiment="E4C",
            status="IMPLEMENTATION_FAILURE",
            reason=f"E2C source artifact is missing: {source}",
            design_valid=False,
            powered=False,
            allowed_claim="None",
            forbidden_claim="Learned evaluator OOD generalization.",
            metrics={},
        )
        build_manifest(output, experiment_id="E4C", status=str(decision["status"]))
        return decision
    rows = read_jsonl_gz(source)
    split_rows = _build_splits(rows)
    reports: list[dict[str, Any]] = []
    for split, train_rows, test_rows in split_rows:
        for family in config["model_families"]:
            report = evaluate_ood(train_rows, test_rows, family=str(family))
            reports.append({"split": split, **report})
    gains = [float(report["transition_ISC"]) - float(report["global_ISC"]) for report in reports]
    low, high = bootstrap_mean_ci(gains, seed=202608264, draws=int(config["statistics"]["bootstrap_draws"]))
    test_seed_counts = {split: len({int(row["seed"]) for row in test_rows}) for split, _, test_rows in split_rows}
    summary = {
        "source_rows": len(rows),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reports": reports,
        "transition_minus_global_isc_mean": float(np.mean(gains)),
        "transition_minus_global_isc_median": float(np.median(gains)),
        "transition_minus_global_isc_ci_low": low,
        "transition_minus_global_isc_ci_high": high,
        "independent_seed_count": len({int(row["seed"]) for row in rows}),
        "test_seed_counts": test_seed_counts,
    }
    powered = all(int(value) >= int(config["minimum_independent_test_seeds"]) for value in test_seed_counts.values()) and int(profile["seeds"]) >= 30
    if not powered:
        status, reason = "UNDERPOWERED", "one or more registered OOD splits has fewer than the required independent test seeds"
    elif low > float(config["statistics"]["minimum_isc_gain"]):
        status, reason = "HYPOTHESIS_SUPPORTED", "differential evaluator ISC exceeds global evaluator ISC across registered OOD reports"
    else:
        status, reason = "HYPOTHESIS_NOT_SUPPORTED", "powered OOD comparison does not support a differential ISC gain"
    write_json(output / "ood_reports.json", reports)
    write_json(output / "ood_summary.json", summary)
    write_provenance(output / "provenance.json", experiment_id="E4C", config=config, root=root, seed_list=sorted({int(row["seed"]) for row in rows}))
    _write_csv(output / "ood_reports.csv", reports)
    decision = write_decision(
        output,
        experiment="E4C",
        status=status,
        reason=reason,
        design_valid=True,
        powered=powered,
        allowed_claim=("Registered OOD splits only." if status == "HYPOTHESIS_SUPPORTED" else reason),
        forbidden_claim="All learned evaluators preserve transition fidelity under arbitrary distribution shift.",
        metrics=summary,
    )
    build_manifest(output, experiment_id="E4C", status=str(decision["status"]))
    return decision


def _build_splits(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    if not rows:
        raise ValueError("E4C requires non-empty E2C rows")
    splits: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    seeds = sorted({int(row["seed"]) for row in rows})
    heldout_seeds = set(seeds[::3])
    splits.append(("trajectory", [row for row in rows if int(row["seed"]) not in heldout_seeds], [row for row in rows if int(row["seed"]) in heldout_seeds]))
    environments = sorted({str(row["environment_id"]) for row in rows})
    heldout_environment = environments[-1]
    splits.append(("environment", [row for row in rows if str(row["environment_id"]) != heldout_environment], [row for row in rows if str(row["environment_id"]) == heldout_environment]))
    operators = sorted({str(row["operator_family"]) for row in rows})
    heldout_operator = operators[-1]
    splits.append(("operator", [row for row in rows if str(row["operator_family"]) != heldout_operator], [row for row in rows if str(row["operator_family"]) == heldout_operator]))
    responses = sorted({float(row["response_strength"]) for row in rows})
    heldout_response = responses[-1]
    splits.append(("response_regime", [row for row in rows if float(row["response_strength"]) != heldout_response], [row for row in rows if float(row["response_strength"]) == heldout_response]))
    for name, train, test in splits:
        if not train or not test or {_row_identity(row) for row in train} & {_row_identity(row) for row in test}:
            raise ValueError(f"invalid E4C split: {name}")
    return splits


def _row_identity(row: dict[str, Any]) -> str:
    """Include regime context so repeated policy edits do not cause leakage."""

    return "|".join(
        str(row.get(key))
        for key in (
            "transition_id",
            "environment_id",
            "response_strength",
            "operator_family",
            "operator_shift",
            "seed",
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
