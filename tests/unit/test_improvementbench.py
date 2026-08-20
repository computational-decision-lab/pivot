from __future__ import annotations

import json
from pathlib import Path

import pytest

from improve_x.benchmark.dataset import ImprovementBenchDataset, ImprovementBenchRow
from improve_x.benchmark.tasks import (
    evaluate_explanation_task,
    evaluate_ranking_task,
    evaluate_sign_task,
)
from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


def make_rows() -> list[ImprovementBenchRow]:
    incumbent = Policy.from_mapping({"intensity": 0.1})
    rows: list[ImprovementBenchRow] = []
    for index, (proxy, actor) in enumerate(((1.0, 0.5), (0.5, -0.2))):
        transition = PolicyTransition(
            incumbent=incumbent,
            candidate=Policy.from_mapping({"intensity": 0.2 + index * 0.1}),
            round_id=0,
            candidate_index=index,
            improvement_operator="test",
            seed=3,
        )
        record = transition.to_record()
        record.update({"delta_proxy": proxy, "delta_actor": actor, "delta_true": actor})
        rows.append(ImprovementBenchRow.from_transition(record, world_level="actor"))
    return rows


def test_benchmark_row_preserves_null_worlds_and_classifies_failure() -> None:
    row = make_rows()[0]
    assert row.delta_strategic is None
    assert row.failure_type == "none"
    assert row.to_record()["world_level"] == "actor"


def test_dataset_round_trip_and_manifest_tamper_detection(tmp_path: Path) -> None:
    dataset = ImprovementBenchDataset(make_rows(), metadata={"seed": 3})
    dataset.write(tmp_path)
    loaded = ImprovementBenchDataset.read(tmp_path)
    assert loaded.rows == dataset.rows
    assert loaded.validate()["valid"] is True
    transitions = tmp_path / "transitions.jsonl"
    transitions.write_text(transitions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert loaded.validate()["valid"] is False


def test_benchmark_tasks_cover_sign_ranking_and_explanation() -> None:
    rows = make_rows()
    predictions = {row.transition_id: row.deployment_delta for row in rows}
    sign = evaluate_sign_task(rows, predictions)
    assert sign["accuracy"] == 1.0
    ranking = evaluate_ranking_task(rows, predictions)
    assert ranking["accuracy"] == 1.0
    explanation = evaluate_explanation_task(
        rows,
        {row.transition_id: row.failure_type for row in rows},
    )
    assert explanation["macro_accuracy"] == 1.0


def test_dataset_rejects_invalid_jsonl(tmp_path: Path) -> None:
    (tmp_path / "transitions.jsonl").write_text("not-json\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        ImprovementBenchDataset.read(tmp_path)


def test_dataset_validation_checks_schema_and_row_count(tmp_path: Path) -> None:
    dataset = ImprovementBenchDataset(make_rows())
    dataset.write(tmp_path, created_at="2026-08-20T00:00:00+00:00")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["row_count"] = 999
    (tmp_path / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    assert dataset.validate(tmp_path)["valid"] is False
    assert any("metadata row_count" in error for error in dataset.validate(tmp_path)["errors"])


def test_row_rejects_unknown_world_level() -> None:
    record = make_rows()[0].to_record()
    record["world_level"] = "unknown-world"

    with pytest.raises(ValueError, match="world_level"):
        ImprovementBenchRow.from_record(record)
