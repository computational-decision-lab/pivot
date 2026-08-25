#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone
from pivot.transfer.reversal import compare_global_vs_local
from pivot.transfer.sampling import stratified_transition_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E4 global versus local fidelity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e4-global-vs-local"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "transitions"
    manifest = run_first_milestone(
        data_dir,
        PerformativeConfig(**payload.get("world", {})),
        payload.get("seeds", []),
        payload.get("candidate_scales", []),
        response_strengths=payload.get("response_strengths"),
        optimization_strengths=payload.get("optimization_strengths"),
    )
    rows = [json.loads(line) for line in (data_dir / "transitions.jsonl").read_text().splitlines()]
    budget = int(payload.get("hf_budget", len(rows) // 2))
    budget = max(1, min(len(rows) - 1, budget))
    train_rows = stratified_transition_sample(rows, budget)
    train_ids = {str(row["transition_id"]) for row in train_rows}
    test_rows = [row for row in rows if str(row["transition_id"]) not in train_ids]
    result = compare_global_vs_local(train_rows, test_rows, budget)
    summary = {key: value for key, value in result.items() if key not in {"local_rows", "train_transition_ids", "test_transition_ids"}}
    summary["transition_row_count"] = len(rows)
    summary["train_row_count"] = len(train_rows)
    summary["test_row_count"] = len(test_rows)
    summary["source_manifest"] = manifest.__dict__
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "budget_ledger.json").write_text(
        json.dumps(
            {
                "global_hf_transitions": result["global_hf_budget"],
                "local_hf_transitions": result["local_hf_budget"],
                "matched": result["global_hf_budget"] == result["local_hf_budget"] == budget,
                "sampling": "stratified_round_robin",
                "train_transition_ids": result["train_transition_ids"],
                "test_transition_ids": result["test_transition_ids"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(args.output / "local_predictions.csv", result["local_rows"])
    print(f"E4 rows={len(rows)} budget={budget} output={args.output}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
