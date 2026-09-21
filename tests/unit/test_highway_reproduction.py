from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reproduction.highway.run import verify_source


def _source(root: Path) -> None:
    (root / "source").mkdir()
    data = b"frozen source\n"
    (root / "source/runner.py").write_bytes(data)
    (root / "source-manifest.json").write_text(json.dumps({"files": {
        "runner.py": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    }}))


def test_frozen_source_rejects_modified_code(tmp_path: Path) -> None:
    _source(tmp_path)
    verify_source(tmp_path)
    (tmp_path / "source/runner.py").write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_source(tmp_path)


def test_frozen_source_rejects_unlisted_code(tmp_path: Path) -> None:
    _source(tmp_path)
    (tmp_path / "source/extra.py").write_text("unexpected")
    with pytest.raises(ValueError, match="membership"):
        verify_source(tmp_path)


def test_frozen_source_rejects_symlink(tmp_path: Path) -> None:
    _source(tmp_path)
    (tmp_path / "source/extra.py").symlink_to(tmp_path / "source/runner.py")
    with pytest.raises(ValueError, match="symlink"):
        verify_source(tmp_path)
