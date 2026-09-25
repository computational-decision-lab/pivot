"""Build V9 tables solely from frozen result JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import sha256, write_json


def build(root: Path) -> dict[str, Any]:
    output = root / "tables/v9"
    output.mkdir(parents=True, exist_ok=True)
    tables: list[dict[str, Any]] = []
    sources = (
        ("E2C", "operator_shift_summary.json", "e2c_cells", "json", "cells"),
        ("E3C", "closed_loop_summary.json", "e3c_methods", "json", "by_method_environment"),
        ("E4C", "ood_reports.json", "e4c_ood", "json", None),
        ("E5C", "efficiency_frontier.csv", "e5c_frontier", "csv", None),
        ("E5C", "calibration_robustness.json", "e5c_calibration", "calibration", None),
        ("E7C", "strategic_summary.json", "e7c_modes", "json", "by_mode"),
    )
    for experiment, source_name, output_name, source_kind, rows_key in sources:
        run = _preferred_run(root, experiment.lower())
        if not run:
            continue
        source = run / source_name
        records = _read_records(source, source_kind, rows_key)
        csv_path = output / f"{output_name}.csv"
        _write_csv(csv_path, records)
        parquet_path = output / f"{output_name}.parquet"
        _write_parquet(parquet_path, records)
        tables.append(
            {
                "experiment": experiment,
                "source": str((run / source_name).relative_to(root)),
                "csv": csv_path.name,
                "parquet": parquet_path.name,
                "source_sha256": sha256(run / source_name),
                "row_count": len(records),
            }
        )
    latex = output / "v9_main_metrics.tex"
    latex.write_text(_latex_table(root, tables) + "\n", encoding="utf-8")
    manifest = {"schema_version": "pivot-v9-table-v1", "tables": tables, "latex": latex.name}
    write_json(output / "table_manifest.json", manifest)
    return manifest


def _preferred_run(root: Path, experiment: str) -> Path | None:
    candidates = (
        root / f"results/v9/{experiment}-confirmatory",
        root / f"results/v9/{experiment}-development",
        root / f"results/v9/{experiment}-smoke",
    )
    return next((path for path in candidates if path.is_dir()), None)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_records(path: Path, source_kind: str, rows_key: str | None) -> list[dict[str, Any]]:
    """Normalize JSON, CSV, and nested calibration artifacts into table rows."""

    if source_kind == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if source_kind == "calibration":
        if not isinstance(payload, dict):
            raise TypeError(f"{path} does not contain calibration mappings")
        records: list[dict[str, Any]] = []
        for diagnostic, values in payload.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict):
                    records.append({"diagnostic": diagnostic, **value})
        return records
    rows = payload if rows_key is None else payload.get(rows_key, [])
    if not isinstance(rows, list):
        raise TypeError(f"{path} does not contain a row list")
    return [row for row in rows if isinstance(row, dict)]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows:
        table = pa.Table.from_pylist([{str(key): value for key, value in row.items()} for row in rows])
    else:
        table = pa.table({"empty": pa.array([], type=pa.string())})
    pq.write_table(table, path)


def _latex_table(root: Path, tables: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Experiment & Source rows & Artifact \\",
        r"\midrule",
    ]
    for table in tables:
        lines.append(f"{table['experiment']} & {table['row_count']} & \\texttt{{{table['csv']}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V9 tables")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = build(args.root.resolve())
    print(json.dumps({"tables": len(manifest["tables"])}, sort_keys=True))


if __name__ == "__main__":
    main()
