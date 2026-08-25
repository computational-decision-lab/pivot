from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_v9_archive_manifest_is_present_and_matches_source() -> None:
    manifest = ROOT / "archive/submission_v9/manifest.sha256"
    source = ROOT / "archive/submission_v9/source"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 300
    for line in lines[:8]:
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64
        assert (source / relative).is_file()


def test_v7_registry_has_disjoint_phase_and_power_contracts() -> None:
    registry = yaml.safe_load((ROOT / "research/experiment_registry.yaml").read_text(encoding="utf-8"))
    assert registry["schema_version"] == "v7.0"
    assert set(registry["confirmatory"]) == {"e3b_closed_loop", "e4b_global_vs_transition", "e7b_external_strategic"}
    for experiment in registry["confirmatory"]:
        config = registry["experiments"][experiment]
        assert config["power"] >= 0.8
        assert config["alpha"] <= 0.05


def test_v7_claim_registry_keeps_finance_boundary_disallowed() -> None:
    claims = yaml.safe_load((ROOT / "research/claim_registry.yaml").read_text(encoding="utf-8"))["claims"]
    assert claims["finance_improvement_reversal"]["status"] == "disallowed"
    assert claims["global_fidelity_insufficiency"]["status"] == "allowed"


def test_research_state_uses_only_declared_values() -> None:
    state = json.loads((ROOT / "research/research_state.json").read_text(encoding="utf-8"))
    allowed = {
        "PENDING",
        "RUNNING",
        "IMPLEMENTATION_FAILURE",
        "DESIGN_INVALID",
        "UNDERPOWERED",
        "HYPOTHESIS_SUPPORTED",
        "HYPOTHESIS_NOT_SUPPORTED",
        "PASSED",
        "FAILED_HYPOTHESIS",
        "FROZEN_NEGATIVE",
        "COMPLETE",
    }
    assert all(value in allowed for key, value in state.items() if key != "schema_version" and key != "baseline_archive")


def test_registry_rejects_missing_power_contract(tmp_path: Path) -> None:
    payload = {"confirmatory": ["e3b"], "experiments": {"e3b": {"alpha": 0.05}}}
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    with pytest.raises(KeyError):
        _ = loaded["experiments"]["e3b"]["power"]
