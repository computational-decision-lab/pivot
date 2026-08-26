from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_v9_baseline import freeze_baseline, verify_baseline


def test_baseline_manifest_is_hash_bound_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "paper/iclr2027").mkdir(parents=True)
    (root / "paper/iclr2027/pivot_iclr2027_submission.pdf").write_bytes(b"pdf")
    (root / "results/v7").mkdir(parents=True)
    (root / "results/v7/metrics.json").write_text("{}", encoding="utf-8")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)

    output = tmp_path / "snapshot"
    manifest = freeze_baseline(root, output)
    assert manifest["repository"]["commit"]
    assert any(item["path"].endswith("metrics.json") for item in manifest["artifacts"])
    assert verify_baseline(root, output / "manifest.json")["valid"] is True
    with pytest.raises(FileExistsError):
        freeze_baseline(root, output)


def test_baseline_verification_detects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "results").mkdir(parents=True)
    (root / "results/out.json").write_text('{"x": 1}', encoding="utf-8")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    output = tmp_path / "snapshot"
    freeze_baseline(root, output)
    (root / "results/out.json").write_text('{"x": 2}', encoding="utf-8")
    report = verify_baseline(root, output / "manifest.json")
    assert report["valid"] is False
    assert report["mismatches"][0]["path"] == "results/out.json"


def test_registry_uses_terminal_states_and_separate_v9_paths() -> None:
    registry = Path("research/experiment_registry_v9.yaml").read_text(encoding="utf-8")
    state = json.loads(Path("research/research_state_v9.json").read_text(encoding="utf-8"))
    assert "IMPLEMENTATION_FAILURE" in registry
    assert "results/v9" in json.dumps(state)
    assert Path("configs/v9/profiles.yaml").is_file()
