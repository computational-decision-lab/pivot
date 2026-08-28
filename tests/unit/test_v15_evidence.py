from __future__ import annotations

import json
from pathlib import Path


def test_candidate_archive_is_sorted_and_content_addressed(tmp_path: Path) -> None:
    from experiments.v15.evidence import freeze_candidate_archive

    source = tmp_path / "candidates.jsonl"
    source.write_text(
        json.dumps({"run_id": "r", "round": 0, "candidate_index": 1, "candidate_hash": "b"})
        + "\n"
        + json.dumps({"run_id": "r", "round": 0, "candidate_index": 0, "candidate_hash": "a"})
        + "\n",
        encoding="utf-8",
    )
    manifest = freeze_candidate_archive(source, tmp_path / "archive")
    rows = (tmp_path / "archive" / "promotion_candidates.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["candidate_hash"] == "a"
    assert manifest["immutable"] is True
    assert len(manifest["archive_sha256"]) == 64


def test_promotion_replay_uses_same_candidate_batch_for_methods() -> None:
    from experiments.v15.evidence import replay_promotion

    rows = [
        {"run_id": "r", "round": 0, "candidate_index": 0, "candidate_hash": "a", "proxy_delta": 0.8},
        {"run_id": "r", "round": 0, "candidate_index": 1, "candidate_hash": "b", "proxy_delta": 0.5},
    ]
    result = replay_promotion(rows, {"a": 0.1, "b": 0.9})
    assert len(result) == 5
    assert len({row["candidate_batch_hash"] for row in result}) == 1
    assert next(row for row in result if row["method"] == "All-HF Oracle")["ISR"] == 0.0
