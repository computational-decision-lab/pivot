from __future__ import annotations

import json
from pathlib import Path


def test_dev_smoke_writes_explicit_nonconfirmatory_manifest(tmp_path: Path) -> None:
    from experiments.v15.dev import run_smoke

    manifest = run_smoke(tmp_path / "smoke", candidates_per_operator=1)
    assert manifest["phase"] == "DEV"
    assert manifest["confirmatory"] is False
    assert manifest["outcome_chasing"] is False
    assert manifest["transition_count"] == 4
    assert (tmp_path / "smoke" / "autonomous_transitions.parquet").is_file()
    assert (tmp_path / "smoke" / "promotion_candidates.jsonl").is_file()
    payload = json.loads((tmp_path / "smoke" / "manifest.json").read_text())
    assert payload["terminal_state"] == "UNDERPOWERED"
    assert payload["design_status"] == "VALIDATED_DEV"
