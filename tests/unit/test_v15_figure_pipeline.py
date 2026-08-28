from __future__ import annotations

import json
from pathlib import Path


def test_figure_pipeline_requires_full_bundle_and_records_state(tmp_path: Path) -> None:
    from experiments.v15.figure_pipeline import bundle_figures

    source = tmp_path / "figures/v10"
    source.mkdir(parents=True)
    manifest = {"figures": [{"figure_id": "demo", "scientific_question": "q", "alias_of": None}]}
    (source / "figure_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for suffix, payload in (("pdf", b"%PDF"), ("svg", b"<svg/>"), ("png", b"png"), ("csv", b"x\n1\n"), ("parquet", b"parquet"), ("meta.json", b"{}")):
        (source / f"demo.{suffix}").write_bytes(payload)
    # The audit can proceed without optional pdf/image command output in this
    # isolated fixture; the bundle contract itself is still checked.
    report = bundle_figures(tmp_path, inspect_stamp="fixed")
    assert report["all_final"]
    bundle = tmp_path / "figures/v15/demo"
    assert (bundle / "figure.audit.json").is_file()
    assert json.loads((bundle / "figure.visual_audit.json").read_text())["state"] == "PAPER_CONTEXT_PASS"
    assert json.loads((tmp_path / ".codex/scientific-figure-memory/demo.json").read_text())["states"][-1] == "FINAL"


def test_figure_pipeline_requires_hash_bound_visual_signoff(tmp_path: Path) -> None:
    from experiments.v15.figure_pipeline import bundle_figures, write_visual_review_manifest
    from figures.v15.audit import audit

    source = tmp_path / "figures/v10"
    source.mkdir(parents=True)
    (source / "figure_manifest.json").write_text(
        json.dumps({"figures": [{"figure_id": "demo", "scientific_question": "q", "alias_of": None}]}),
        encoding="utf-8",
    )
    for suffix, payload in (
        ("pdf", b"%PDF"),
        ("svg", b"<svg/>"),
        ("png", b"png"),
        ("csv", b"x\n1\n"),
        ("parquet", b"parquet"),
        ("meta.json", b"{}"),
    ):
        (source / f"demo.{suffix}").write_bytes(payload)

    blocked = bundle_figures(tmp_path)
    assert blocked["all_final"] is False
    assert blocked["records"][0]["state"] == "BLOCKED"

    write_visual_review_manifest(tmp_path, reviewer="test")
    approved = bundle_figures(tmp_path)
    assert approved["all_final"] is True
    assert approved["records"][0]["visual"]["review_manifest"] == "figures/v15/visual_review_manifest.json"
    assert audit(tmp_path)["valid"] is True

    (source / "demo.csv").write_bytes(b"x\n2\n")
    invalidated = bundle_figures(tmp_path)
    assert invalidated["all_final"] is False
    assert invalidated["records"][0]["state"] == "BLOCKED"


def test_figure_audit_rejects_unbound_visual_pass(tmp_path: Path) -> None:
    from experiments.v15.figure_pipeline import bundle_figures
    from figures.v15.audit import audit

    source = tmp_path / "figures/v10"
    source.mkdir(parents=True)
    (source / "figure_manifest.json").write_text(
        json.dumps({"figures": [{"figure_id": "demo", "scientific_question": "q", "alias_of": None}]}),
        encoding="utf-8",
    )
    for suffix, payload in (
        ("pdf", b"%PDF"),
        ("svg", b"<svg/>"),
        ("png", b"png"),
        ("csv", b"x\n1\n"),
        ("parquet", b"parquet"),
        ("meta.json", b"{}"),
    ):
        (source / f"demo.{suffix}").write_bytes(payload)
    bundle_figures(tmp_path, inspect_stamp="fixture")
    assert audit(tmp_path)["status"] == "BLOCKED"
