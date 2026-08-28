"""Candidate freezing and decision replay for the V15 development harness.

The replay consumes an immutable candidate table and a gate table.  It is
useful for validating schemas and paired-selection bookkeeping, but it is
explicitly labelled DEV until a locked external scaffold supplies deployment
evidence.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .protocol import content_hash, file_hash, write_jsonl, write_table


def freeze_candidate_archive(
    source: Path,
    output: Path,
    *,
    phase: str = "DEV",
    confirmatory: bool = False,
    source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Copy candidate rows into a content-addressed, read-only archive.

    The archive is the boundary between generation and validation.  Reusing an
    existing directory is allowed only when its content hash is identical;
    silently replacing a candidate set is never allowed.
    """

    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "promotion_candidates.jsonl"
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("candidate archive rows must be mappings")
            rows.append(payload)
    rows.sort(key=lambda row: (str(row.get("run_id")), int(row.get("round", 0)), int(row.get("candidate_index", 0))))
    normalized = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if target.is_file() and target.read_text(encoding="utf-8") != normalized:
        raise ValueError(f"candidate archive already exists with different content: {target}")
    if not target.is_file():
        write_jsonl(rows, target)
    digest = file_hash(target)
    manifest = {
        "schema_version": "pivot-v15-candidate-archive-1",
        "phase": str(phase),
        "immutable": True,
        "row_count": len(rows),
        "source_sha256": file_hash(source),
        "archive_sha256": digest,
        "candidate_batch_hash": content_hash([row.get("candidate_hash") for row in rows]),
        "confirmatory": bool(confirmatory),
        "source_manifest_sha256": source_manifest_sha256,
    }
    manifest["regeneration_allowed"] = False
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping):
            existing_unsigned = dict(existing)
            existing_unsigned.pop("manifest_sha256", None)
            expected_unsigned = dict(manifest)
            expected_unsigned.pop("manifest_sha256", None)
            if existing_unsigned != expected_unsigned:
                raise ValueError(f"candidate archive manifest mismatch: {manifest_path}")
    manifest["manifest_sha256"] = content_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('run_id')}::{row.get('round', 0)}"
        grouped.setdefault(key, []).append(row)
    return grouped


def replay_promotion(
    candidate_rows: Sequence[Mapping[str, Any]],
    gate_scores: Mapping[str, float],
    *,
    seed: int = 101,
) -> list[dict[str, Any]]:
    """Replay transparent selector baselines on the same candidate archive."""

    methods = ("Proxy Only", "Random HF", "Global-VOI", "PIVOT-VOI", "All-HF Oracle")
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for group in _group(candidate_rows).values():
        if not group:
            continue
        ordered = sorted(group, key=lambda row: int(row.get("candidate_index", 0)))
        def proxy(row: Mapping[str, Any]) -> float:
            return float(row.get("proxy_delta", row.get("delta_proxy", 0.0)))

        def gate(row: Mapping[str, Any]) -> float:
            return float(gate_scores.get(str(row.get("candidate_hash")), 0.0))

        max_gate = max(gate(row) for row in ordered)
        for method in methods:
            if method == "Proxy Only":
                selected = max(ordered, key=proxy)
            elif method == "Random HF":
                selected = ordered[rng.randrange(len(ordered))]
            elif method == "All-HF Oracle":
                selected = max(ordered, key=gate)
            else:
                # DEV selector: a deterministic deployment-informed correction
                # proxy, standing in for the future sealed posterior interface.
                selected = max(ordered, key=lambda row: proxy(row) + 0.25 * gate(row))
            selected_gate = gate(selected)
            results.append(
                {
                    "phase": "DEV",
                    "method": method,
                    "run_id": str(selected.get("run_id")),
                    "round": int(selected.get("round", 0)),
                    "selected_candidate": str(selected.get("candidate_hash")),
                    "true_best_candidate": str(max(ordered, key=gate).get("candidate_hash")),
                    "ISR": max_gate - selected_gate,
                    "hf_cost": 0.0 if method == "Proxy Only" else float(1 if method != "All-HF Oracle" else len(ordered)),
                    "candidate_batch_hash": content_hash([row.get("candidate_hash") for row in ordered]),
                }
            )
    return results


def write_promotion_replay(candidate_archive: Path, output: Path) -> dict[str, Any]:
    """Run a DEV replay using deterministic gate scores from sandbox traces."""

    rows = [json.loads(line) for line in candidate_archive.read_text(encoding="utf-8").splitlines() if line.strip()]
    gate_scores = {
        str(row["candidate_hash"]): float(bool(row.get("candidate_policy", {}).get("test_policy", {}).get("repair", False)))
        for row in rows
    }
    results = replay_promotion(rows, gate_scores)
    output.mkdir(parents=True, exist_ok=True)
    write_table(
        results,
        output / "promotion_results",
        columns=("phase", "method", "run_id", "round", "selected_candidate", "true_best_candidate", "ISR", "hf_cost", "candidate_batch_hash"),
    )
    write_jsonl(results, output / "promotion_results.jsonl")
    manifest = {
        "phase": "DEV",
        "confirmatory": False,
        "candidate_archive_sha256": file_hash(candidate_archive),
        "result_count": len(results),
        "methods": sorted({str(row["method"]) for row in results}),
        "outcome_chasing": False,
        "note": "Schema/selection replay only; gate scores are local reference diagnostics.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
