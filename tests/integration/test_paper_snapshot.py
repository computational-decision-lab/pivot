from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_paper_snapshot import (
    ARCHITECTURE_FILES,
    CONTRAST_FILES,
    FIGURE_FILES,
    SUMMARY_FILES,
    freeze_snapshot,
)


def _make_source(root: Path) -> None:
    (root / "figures").mkdir(parents=True)
    for relative in FIGURE_FILES:
        path = root / "figures" / relative
        path.write_text("x", encoding="utf-8")
    for relative in SUMMARY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "figures/figure_validation.json":
            path.write_text(json.dumps({"valid": True, "checked": list(range(7))}), encoding="utf-8")
        else:
            path.write_text("{}" if relative.endswith(".json") else "x", encoding="utf-8")


def _make_architecture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ARCHITECTURE_FILES:
        path = root / relative
        path.write_text("{}" if relative.endswith(".json") else "x", encoding="utf-8")


def test_freeze_snapshot_copies_hash_indexed_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    ablations = tmp_path / "ablations"
    public = tmp_path / "public"
    _make_source(source)
    (ablations).mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    public.mkdir()
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    manifest = freeze_snapshot(source, ablations, public, tmp_path / "snapshot")
    assert len(manifest["files"]) == len(FIGURE_FILES) + len(SUMMARY_FILES) + 3
    assert manifest["claim_boundary"]["public_causal_impact_identified"] is False
    assert (tmp_path / "snapshot/figures/fig1_when_better_gets_worse.png").exists()


def test_freeze_snapshot_refuses_overwrite_and_missing_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "snapshot"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_snapshot(source, ablations, public, output)
    (source / "figures/fig7_strategic_reversal.csv").unlink()
    with pytest.raises(FileNotFoundError):
        freeze_snapshot(source, ablations, public, tmp_path / "missing")


def test_freeze_snapshot_binds_optional_e4_contrast(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    contrast = tmp_path / "contrast"
    contrast.mkdir()
    for relative in CONTRAST_FILES:
        (contrast / relative).write_text("{}" if relative.endswith(".json") else "x", encoding="utf-8")
    manifest = freeze_snapshot(source, ablations, public, tmp_path / "snapshot", contrast_root=contrast)
    assert "summaries/e4-contrast-comparison.json" in manifest["files"]
    assert manifest["claim_boundary"]["e4_contrast_is_controlled_diagnostic"] is True


def test_freeze_snapshot_binds_pinned_opentikz_architecture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    architecture = tmp_path / "architecture"
    _make_architecture(architecture)
    manifest = freeze_snapshot(
        source,
        ablations,
        public,
        tmp_path / "snapshot",
        architecture_root=architecture,
    )
    assert manifest["claim_boundary"]["opentikz_architecture_pinned"] is True
    assert manifest["files"]["figures/fig3_pivot_architecture.tex"]["source_kind"] == "opentikz-pinned"


def test_freeze_snapshot_sanitizes_machine_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    (source / "p2-summary.json").write_text(
        '{"input": "/tmp/private-run", "project": "/opt/projects/research/pivot"}',
        encoding="utf-8",
    )
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    freeze_snapshot(source, ablations, public, tmp_path / "snapshot")
    copied = (tmp_path / "snapshot/summaries/p2-summary.json").read_text(encoding="utf-8")
    assert "/tmp/" not in copied
    assert "/opt/projects" not in copied
    assert "<local-root>" in copied


def test_snapshot_manifest_uses_portable_source_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "ablation-aggregate.json").write_text("{}", encoding="utf-8")
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text("{}", encoding="utf-8")
    (public / "provenance.json").write_text("{}", encoding="utf-8")
    manifest = freeze_snapshot(source, ablations, public, tmp_path / "snapshot")
    assert manifest["source_roots"] == {
        "controlled": "<clean-room>",
        "ablations": "<clean-room>",
        "public_expansion": "<public-data-audit>",
        "e4_contrast": None,
    }
