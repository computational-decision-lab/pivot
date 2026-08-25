#!/usr/bin/env python3
"""Evaluate the three public IMPROVE-X benchmark tasks from a frozen release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from improve_x.benchmark.dataset import ImprovementBenchDataset, ImprovementBenchRow
from improve_x.benchmark.tasks import (
    evaluate_explanation_task,
    evaluate_ranking_task,
    evaluate_sign_task,
)
from improve_x.metrics.fidelity import compute_layer_fidelity


def _source_transition_id(row: ImprovementBenchRow) -> str:
    source = row.metadata.get("source_transition_id")
    if source is not None:
        return str(source)
    suffix = f"-{row.world_level}"
    return row.transition_id.removesuffix(suffix)


def _combined_layer_rows(rows: Iterable[ImprovementBenchRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[ImprovementBenchRow]] = defaultdict(list)
    for row in rows:
        grouped[_source_transition_id(row)].append(row)
    combined: list[dict[str, object]] = []
    for source_id, members in sorted(grouped.items()):
        combined.append(
            {
                "transition_id": source_id,
                "delta_proxy": _first_number(member.proxy_delta for member in members),
                "delta_actor": _first_number(member.actor_delta for member in members),
                "delta_strategic": _first_number(member.strategic_delta for member in members),
            }
        )
    return combined


def _first_number(values: Iterable[float | None]) -> float | None:
    return next((float(value) for value in values if value is not None), None)


def evaluate(
    dataset: ImprovementBenchDataset,
    manifest_sha256: str,
    *,
    requested_split: str | None = None,
) -> dict[str, object]:
    rows = dataset.rows_for_split(requested_split) if requested_split else tuple(dataset.rows)
    predictions = {
        row.transition_id: row.proxy_delta
        for row in rows
        if row.proxy_delta is not None
    }
    sign_by_world = {
        world_level: evaluate_sign_task(
            [row for row in rows if row.world_level == world_level], predictions
        )
        for world_level in sorted({row.world_level for row in rows})
    }
    ranking = evaluate_ranking_task(rows, predictions)
    oracle_labels = {row.transition_id: row.failure_type for row in rows}
    layer_rows = tuple(_combined_layer_rows(rows))
    return {
        "schema_version": "improvementbench-evaluation-v1",
        "input_manifest_sha256": manifest_sha256,
        "dataset_valid": True,
        "requested_split": requested_split,
        "row_count": len(rows),
        "world_level_counts": dict(sorted(Counter(row.world_level for row in rows).items())),
        "failure_counts": dict(sorted(Counter(row.failure_type for row in rows).items())),
        "sign_by_world": sign_by_world,
        "proxy_ranking": ranking,
        "oracle_explanation_sanity": evaluate_explanation_task(rows, oracle_labels),
        "layer_fidelity": compute_layer_fidelity(layer_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ImprovementBench release")
    parser.add_argument("--input", type=Path, required=True, help="ImprovementBench directory")
    parser.add_argument("--output", type=Path, required=True, help="metrics output directory")
    parser.add_argument("--split", dest="requested_split", help="optional frozen split name")
    args = parser.parse_args()
    dataset = ImprovementBenchDataset.read(args.input)
    validation = dataset.validate()
    if not bool(validation["valid"]):
        raise ValueError(f"invalid ImprovementBench release: {validation['errors']}")
    manifest_sha256 = hashlib.sha256((args.input / "manifest.json").read_bytes()).hexdigest()
    result = evaluate(dataset, manifest_sha256, requested_split=args.requested_split)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": result["row_count"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
