from pathlib import Path

from pivot.logging.transition_store import TransitionStore


def test_store_writes_manifest_and_detects_tampering(tmp_path: Path) -> None:
    store = TransitionStore(tmp_path / "run", required_columns=("transition_id", "delta_proxy"))
    store.append({"transition_id": "t0", "delta_proxy": 1.0})
    manifest = store.finalize()
    assert manifest.row_count == 1
    assert (tmp_path / "run" / "transitions.jsonl").exists()
    assert (tmp_path / "run" / "manifest.json").exists()
    (tmp_path / "run" / "transitions.jsonl").write_text("tampered\n", encoding="utf-8")
    assert not TransitionStore.validate_manifest(tmp_path / "run")
