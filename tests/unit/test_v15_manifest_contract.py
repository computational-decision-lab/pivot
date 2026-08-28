from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dev_manifest_backfill_assigns_underpowered_and_is_idempotent(tmp_path: Path) -> None:
    from experiments.v15.manifest_contract import backfill_dev_manifests

    directory = tmp_path / "results/v15/dev-external-transition-audit"
    directory.mkdir(parents=True)
    payload = {"phase": "DEV", "confirmatory": False, "status": "COMPLETED"}
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    first = backfill_dev_manifests(tmp_path)
    updated = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert "dev-external-transition-audit" in first["changed"]
    assert updated["terminal_state"] == "UNDERPOWERED"
    assert updated["execution_attempted"] is True
    assert updated["manifest_sha256"]

    second = backfill_dev_manifests(tmp_path)
    assert second["changed"] == []


def test_confirmatory_manifest_is_never_mutated(tmp_path: Path) -> None:
    from experiments.v15.manifest_contract import backfill_dev_manifests

    directory = tmp_path / "results/v15/dev-external-transition-audit"
    directory.mkdir(parents=True)
    payload = {"phase": "CONFIRMATORY", "confirmatory": True, "status": "COMPLETED"}
    path = directory / "manifest.json"
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")

    result = backfill_dev_manifests(tmp_path)
    assert result["skipped_confirmatory"] == ["dev-external-transition-audit"]
    assert path.read_text(encoding="utf-8") == original


def test_candidate_archive_manifest_migration_is_idempotent(tmp_path: Path) -> None:
    from experiments.v15.manifest_contract import backfill_dev_manifests
    from experiments.v15.protocol import file_hash

    directory = tmp_path / "results/v15/candidate-archive"
    directory.mkdir(parents=True)
    archive = directory / "promotion_candidates.jsonl"
    archive.write_text('{"candidate_id":"c0"}\n', encoding="utf-8")
    manifest = {
        "archive_sha256": file_hash(archive),
        "row_count": 1,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    first = backfill_dev_manifests(tmp_path)
    assert "candidate-archive" in first["changed"]
    manifest_after_first = (directory / "manifest.json").read_text(encoding="utf-8")
    second = backfill_dev_manifests(tmp_path)
    assert "candidate-archive" not in second["changed"]
    assert (directory / "manifest.json").read_text(encoding="utf-8") == manifest_after_first


def test_ablation_archive_is_materialized_and_validated_as_frozen(tmp_path: Path) -> None:
    from experiments.v15.run_ablations import _resolve_frozen_archive

    source = tmp_path / "results/v15/dev-external-transition-audit/promotion_candidates.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"run_id":"r","round":0,"candidate_index":0,"candidate_hash":"c"}\n', encoding="utf-8")

    archive, manifest = _resolve_frozen_archive(
        tmp_path,
        confirmatory=False,
        source=source,
    )

    assert archive == tmp_path / "results/v15/dev-external-candidate-archive/promotion_candidates.jsonl"
    assert manifest["immutable"] is True
    assert manifest["regeneration_allowed"] is False

    # A changed source cannot silently replace the already frozen archive.
    source.write_text('{"run_id":"r","round":0,"candidate_index":0,"candidate_hash":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        _resolve_frozen_archive(tmp_path, confirmatory=False, source=source)
