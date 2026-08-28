from __future__ import annotations

import json
from pathlib import Path


def test_publication_build_uses_stable_provenance_anchor() -> None:
    root = Path.cwd()
    payload = json.loads((root / "configs/v15/build_provenance.json").read_text(encoding="utf-8"))
    expected = payload["artifact_source_commit"]

    from experiments.v10.figures import _commit
    from scripts.build_opentikz_architecture import _git_commit

    assert _commit(root) == expected
    assert _git_commit(root) == expected


def test_explicit_build_provenance_override_is_honoured(monkeypatch) -> None:
    root = Path.cwd()
    monkeypatch.setenv("PIVOT_BUILD_COMMIT", "test-build-anchor")

    from experiments.v10.figures import _commit
    from scripts.build_opentikz_architecture import _git_commit

    assert _commit(root) == "test-build-anchor"
    assert _git_commit(root) == "test-build-anchor"
