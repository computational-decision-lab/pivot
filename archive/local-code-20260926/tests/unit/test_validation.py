from __future__ import annotations

import json
from pathlib import Path

from pivot.validation import validate_run_artifacts


def test_validate_run_artifacts_checks_manifest_hashes(tmp_path: Path) -> None:
    data = tmp_path / "data.txt"
    data.write_text("ok", encoding="utf-8")
    from pivot.validation import sha256

    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"data.txt": sha256(data)}}), encoding="utf-8"
    )
    assert validate_run_artifacts(tmp_path)["valid"] is True
    data.write_text("tampered", encoding="utf-8")
    assert validate_run_artifacts(tmp_path)["valid"] is False
