#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.benchmark.improvementbench_v7 import (
    ImprovementBenchV7Dataset,
    ImprovementBenchV7Row,
    assign_leakage_safe_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe ImprovementBench V7")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--group-key", choices=["trajectory_id", "environment_id", "operator_id", "opponent_family", "response_regime"], default="trajectory_id")
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for path in args.input:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record: dict[str, Any] = json.loads(raw)
            record["source"] = path.stem
            source_variant = str(record.get("method", record.get("operator_id", "default")))
            record["transition_id"] = f"{path.stem}:{source_variant}:{record.get('transition_id', 'unknown')}"
            # Namespace every leakage group by source.  A trajectory label such
            # as ``trajectory-000`` is not globally unique across E3/E4/E7.
            for key in ("trajectory_id", "environment_id", "operator_id", "opponent_family", "response_regime"):
                value = record.get(key)
                if value is None:
                    value = {
                        "trajectory_id": record.get("transition_id", "unknown"),
                        "environment_id": "unknown",
                        "operator_id": record.get("method", "unknown"),
                        "opponent_family": "none",
                        "response_regime": "default",
                    }[key]
                record[key] = f"{path.stem}:{value}"
            records.append(record)
    splits = assign_leakage_safe_splits(records, seed=args.seed)
    rows = [ImprovementBenchV7Row.from_record(record, split=split) for record, split in zip(records, splits)]
    dataset = ImprovementBenchV7Dataset(rows, metadata={"split_group_key": "all_registered_groups", "split_seed": args.seed, "source_files": [str(path) for path in args.input]})
    manifest = dataset.write(args.output)
    print(json.dumps({"rows": len(rows), "manifest": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
