"""Registered paired, footprint, and VOI ablations on a frozen archive."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .confirmatory_guards import (
    registered_counts,
    reject_existing_confirmatory_output,
    require_registered_budgets,
    require_registered_count,
)
from .external_promotion import (
    _group_rows,
    _lock_protocol_inputs,
    build_query_plan,
    load_candidate_archive,
)
from .external_runtime import evaluate_with_inspect, resolve_runtime_settings
from .planes import load_task_planes
from .promotion import _candidate_id
from .protocol import AgentPolicy, canonical_json, file_hash, write_jsonl, write_table


def bounded_tasks(
    tasks: Sequence[Any], *, confirmatory: bool, task_limit: int | None
) -> list[Any]:
    """Apply a bounded task view only to DEV ablations."""

    if confirmatory and task_limit is not None:
        raise ValueError("confirmatory ablations cannot use a task limit")
    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive")
    return list(tasks if task_limit is None else tasks[:task_limit])


def _phase_output(root: Path, *, confirmatory: bool) -> Path:
    return Path(root).resolve() / "results/v15" / (
        "external-ablations" if confirmatory else "dev-external-ablations"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _resolve_frozen_archive(root: Path, *, confirmatory: bool, source: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve the immutable generation/validation boundary for ablations.

    Ablations must consume exactly the archive that promotion methods see.  A
    DEV run may create that archive once from the completed DEV transition
    output; confirmatory runs fail closed when the phase-1 archive is absent or
    its content/manifest hash does not agree.
    """

    archive_root = root / "results/v15" / (
        "external-candidate-archive" if confirmatory else "dev-external-candidate-archive"
    )
    archive_file = archive_root / "promotion_candidates.jsonl"
    archive_manifest_path = archive_root / "manifest.json"
    if not archive_file.is_file() or not archive_manifest_path.is_file():
        if confirmatory:
            raise ValueError("confirmatory ablations require an immutable candidate archive")
        from .evidence import freeze_candidate_archive

        freeze_candidate_archive(source, archive_root, phase="DEV", confirmatory=False)
    if not archive_file.is_file() or not archive_manifest_path.is_file():
        raise ValueError("candidate archive could not be materialized")
    payload = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("candidate archive manifest must be a mapping")
    expected_phase = "CONFIRMATORY" if confirmatory else "DEV"
    if (
        payload.get("phase") != expected_phase
        or payload.get("confirmatory") is not confirmatory
        or payload.get("immutable") is not True
        or payload.get("regeneration_allowed") is not False
        or payload.get("archive_sha256") != file_hash(archive_file)
    ):
        raise ValueError("candidate archive failed immutability/hash validation")
    if source.is_file() and payload.get("source_sha256") not in {None, file_hash(source)}:
        raise ValueError("candidate archive source has different content from the frozen archive")
    return archive_file, dict(payload)


def _unpaired_dev_rows(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    confirmatory: bool,
    verify_image: bool = False,
    task_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Estimate independent candidate/incumbent differences on gate tasks.

    The seeds differ by construction, so this is a genuine no-pairing
    diagnostic.  It is an ablation result only and is never passed to the
    promotion operator.
    """

    planes = load_task_planes(Path(root).resolve() / "configs/v15/task_manifest.json")
    tasks = bounded_tasks(planes.tasks("gate", role="baseline"), confirmatory=confirmatory, task_limit=task_limit)
    if not tasks:
        return []
    output = _phase_output(root, confirmatory=confirmatory)
    settings = resolve_runtime_settings(
        root,
        artifact_root=output / "artifacts",
        log_root=output / "inspect-logs",
        verify_image=verify_image,
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        incumbent = AgentPolicy.from_record(row["incumbent_policy"])
        candidate = AgentPolicy.from_record(row["candidate_policy"])
        left = evaluate_with_inspect(
            tasks, incumbent, settings, seed=int(row.get("seed", 0)) + 700001 + index,
            phase="ablation_unpaired/incumbent", role="baseline", run_id=f"unpaired-{_candidate_id(row)}-inc",
        )
        right = evaluate_with_inspect(
            tasks, candidate, settings, seed=int(row.get("seed", 0)) + 800001 + index,
            phase="ablation_unpaired/candidate", role="baseline", run_id=f"unpaired-{_candidate_id(row)}-cand",
        )
        incumbent_score = sum(record.success for record in left) / max(len(left), 1)
        candidate_score = sum(record.success for record in right) / max(len(right), 1)
        results.append(
            {
                "ablation": "no_pairing",
                "status": "COMPLETED",
                "candidate_id": _candidate_id(row),
                "run_id": str(row.get("run_id", "")),
                "round": int(row.get("round", 0)),
                "paired_delta": None,
                "unpaired_delta": float(candidate_score - incumbent_score),
                "incumbent_score": float(incumbent_score),
                "candidate_score": float(candidate_score),
                "independent_task_count": len(tasks),
                "note": "Independent seeds and fresh sandboxes; compare with paired truth audit after selection.",
            }
        )
    return results


def run_ablations(
    root: Path,
    *,
    confirmatory: bool = False,
    verify_image: bool = False,
    task_limit: int | None = None,
) -> dict[str, Any]:
    """Run all registered ablations available from the immutable candidate archive."""

    root = Path(root).resolve()
    if confirmatory and os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
        raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
    if confirmatory and not verify_image:
        raise ValueError("confirmatory ablations require sandbox image verification")
    if confirmatory:
        bounded_tasks((), confirmatory=True, task_limit=task_limit)
    lock = _lock_protocol_inputs(root) if confirmatory else None
    output = _phase_output(root, confirmatory=confirmatory)
    reject_existing_confirmatory_output(output, confirmatory)
    source = root / "results/v15" / (
        "external-transition-audit" if confirmatory else "dev-external-transition-audit"
    )
    if confirmatory:
        transition_manifest_path = source / "manifest.json"
        promotion_dir = root / "results/v15/external-promotion"
        promotion_manifest_path = promotion_dir / "manifest.json"
        if not transition_manifest_path.is_file() or not promotion_manifest_path.is_file():
            raise ValueError("confirmatory ablations require completed transition and promotion manifests")
        transition_manifest = json.loads(transition_manifest_path.read_text(encoding="utf-8"))
        promotion_manifest = json.loads(promotion_manifest_path.read_text(encoding="utf-8"))
        if any(
            not isinstance(manifest, Mapping)
            or manifest.get("phase") != "CONFIRMATORY"
            or manifest.get("status") != "COMPLETED"
            for manifest in (transition_manifest, promotion_manifest)
        ):
            raise ValueError("confirmatory ablations require completed CONFIRMATORY source phases")
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise TypeError("confirmatory config must be a mapping")
        counts = registered_counts(config, operator_count=int(transition_manifest.get("operator_count", 2)))
        expected = counts["trajectories"] * counts["rounds"] * counts["candidates"]
        require_registered_count(confirmatory, int(transition_manifest.get("candidate_count", 0)), expected, "transition candidate")
        require_registered_count(confirmatory, int(promotion_manifest.get("candidate_count", 0)), expected, "promotion candidate")
        if lock is None:
            raise RuntimeError("confirmatory lock was not resolved")
        require_registered_budgets(confirmatory, (1, 2, 4), lock.get("hf_budgets", (1, 2, 4)))
    if confirmatory:
        lock = _lock_protocol_inputs(root, phase="registered_ablations")
    archive, archive_manifest = _resolve_frozen_archive(root, confirmatory=confirmatory, source=source / "promotion_candidates.jsonl")
    rows = load_candidate_archive(archive)
    promotion_dir = root / "results/v15" / (
        "external-promotion" if confirmatory else "dev-external-promotion"
    )
    promotion_rows = _load_jsonl(promotion_dir / "promotion_results.jsonl")
    truth_rows = _load_jsonl(promotion_dir / "truth_audit.jsonl")
    records: list[dict[str, Any]] = []
    grouped = _group_rows(rows)
    for group_index, group in enumerate(grouped):
        for budget in (1, 2, 4):
            full = build_query_plan(group, method="PIVOT-VOI", budget=budget, seed=101 + group_index)
            no_footprint = [dict(row, footprint={}) for row in group]
            no_fp = build_query_plan(no_footprint, method="PIVOT-VOI", budget=budget, seed=101 + group_index)
            no_voi = build_query_plan(group, method="PIVOT-H", budget=budget, seed=101 + group_index)
            full_ids = [row["candidate_id"] for row in full]
            no_fp_ids = [row["candidate_id"] for row in no_fp]
            no_voi_ids = [row["candidate_id"] for row in no_voi]
            records.extend(
                [
                    {
                        "ablation": "no_footprint",
                        "status": "COMPLETED",
                        "run_id": str(group[0].get("run_id", "")),
                        "round": int(group[0].get("round", 0)),
                        "budget": budget,
                        "candidate_count": len(group),
                        "reference_method": "PIVOT-VOI",
                        "comparison_method": "PIVOT-VOI_without_footprint",
                        "reference_query_ids": full_ids,
                        "comparison_query_ids": no_fp_ids,
                        "query_overlap": len(set(full_ids) & set(no_fp_ids)) / max(len(set(full_ids) | set(no_fp_ids)), 1),
                        "metric": "query_set_jaccard",
                        "value": len(set(full_ids) & set(no_fp_ids)) / max(len(set(full_ids) | set(no_fp_ids)), 1),
                    },
                    {
                        "ablation": "no_voi",
                        "status": "COMPLETED",
                        "run_id": str(group[0].get("run_id", "")),
                        "round": int(group[0].get("round", 0)),
                        "budget": budget,
                        "candidate_count": len(group),
                        "reference_method": "PIVOT-VOI",
                        "comparison_method": "PIVOT-H",
                        "reference_query_ids": full_ids,
                        "comparison_query_ids": no_voi_ids,
                        "query_overlap": len(set(full_ids) & set(no_voi_ids)) / max(len(set(full_ids) | set(no_voi_ids)), 1),
                        "metric": "query_set_jaccard",
                        "value": len(set(full_ids) & set(no_voi_ids)) / max(len(set(full_ids) | set(no_voi_ids)), 1),
                    },
                ]
            )
    records.extend(
        _unpaired_dev_rows(
            root,
            rows,
            confirmatory=confirmatory,
            verify_image=verify_image,
            task_limit=task_limit,
        )
    )
    # Promotion rows are copied only as descriptive baseline diagnostics; their
    # post-decision truth is not fed back into any selector.
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in promotion_rows:
        by_method[str(row.get("method"))].append(float(row.get("ISR", 0.0)))
    for method, values in sorted(by_method.items()):
        records.append(
            {
                "ablation": "promotion_baseline_diagnostic",
                "status": "COMPLETED",
                "reference_method": method,
                "comparison_method": "All-HF Oracle",
                "metric": "mean_ISR",
                "value": sum(values) / max(len(values), 1),
                "independent_unit_count": len(values),
                "truth_audit_count": len(truth_rows),
            }
        )
    write_jsonl(records, output / "ablation_results.jsonl")
    write_table(
        records,
        output / "ablation_results",
        columns=(
            "ablation", "status", "run_id", "round", "budget", "candidate_id", "candidate_count",
            "reference_method", "comparison_method", "reference_query_ids", "comparison_query_ids",
            "query_overlap", "metric", "value", "paired_delta", "unpaired_delta", "incumbent_score",
            "candidate_score", "independent_task_count", "independent_unit_count", "truth_audit_count", "note",
        ),
    )
    return _manifest(
        output,
        {
            "schema_version": "pivot-v15-ablations-1",
            "phase": "CONFIRMATORY" if confirmatory else "DEV",
            "confirmatory": confirmatory,
            "status": "COMPLETED" if records else "UNDERPOWERED",
            "terminal_state": "UNDERPOWERED" if records and not confirmatory else None if records else "UNDERPOWERED",
            "execution_attempted": True,
            "design_status": "VALIDATED_DEV" if not confirmatory and records else "PENDING_ANALYSIS",
            "leakage_detected": False,
            "record_count": len(records),
            "candidate_count": len(rows),
            "candidate_batch_count": len(grouped),
            "promotion_result_count": len(promotion_rows),
            "truth_audit_count": len(truth_rows),
            "registered_ablation_families": ["no_pairing", "no_footprint", "no_voi", "promotion_baseline_diagnostic"],
            "assessment_accessed": False,
            "outcome_chasing": False,
            "candidate_archive_frozen": True,
            "candidate_archive_path": str(archive.relative_to(root)),
            "candidate_archive_sha256": file_hash(archive),
            "candidate_archive_manifest_sha256": file_hash(archive.parent / "manifest.json"),
            "candidate_archive_row_count": int(archive_manifest.get("row_count", len(rows))),
            "lock_hash": lock.get("lock_hash") if lock else None,
            "note": "All ablations use the frozen candidate archive; no result is used to alter the protocol or generate candidates.",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run registered V15 ablations")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_ablations(args.root.resolve(), confirmatory=args.confirmatory, task_limit=args.task_limit),
            sort_keys=True,
        )
    )
