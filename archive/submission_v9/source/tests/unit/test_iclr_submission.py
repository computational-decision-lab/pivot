from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_iclr_supplement import ALLOWLIST
from scripts.verify_iclr_submission import (
    _aux_path,
    _portable_path,
    audit_archive_members,
    audit_source_text,
    audit_style_hashes,
    build_decision,
)


def test_aux_path_falls_back_to_build_sidecar_for_copied_submission(tmp_path: Path) -> None:
    submission = tmp_path / "paper" / "pivot_submission.pdf"
    submission.parent.mkdir(parents=True)
    build_aux = submission.parent / "build" / "main.aux"
    build_aux.parent.mkdir()
    build_aux.write_text(r"\\newlabel{refs:start}{{}{10}}", encoding="utf-8")

    assert _aux_path(submission) == build_aux


def test_portable_path_avoids_machine_absolute_prefix() -> None:
    rendered = _portable_path(Path.cwd() / "paper" / "iclr2027" / "main.tex")
    assert rendered == "paper/iclr2027/main.tex"
    assert "/opt/projects/" not in rendered


def test_supplement_allowlist_contains_improvementbench_v2() -> None:
    assert "benchmarks/improvementbench/v2" in ALLOWLIST


def test_supplement_allowlist_contains_registered_theory_artifact() -> None:
    assert "results/theory" in ALLOWLIST


def test_audit_source_text_requires_anonymous_iclr_contract() -> None:
    source = r"""
    \documentclass{article}
    \usepackage{iclr2027_conference}
    \section*{AI Use Statement}
    \section*{Reproducibility Statement}
    Improvement Fidelity PIVOT
    """
    checks = audit_source_text(source)
    assert checks["official_style"]
    assert checks["ai_use_statement"]
    assert checks["reproducibility_statement"]
    assert checks["no_final_copy"]
    assert checks["no_identity_leak"]


def test_audit_source_text_rejects_identity_and_final_copy() -> None:
    source = r"""
    \usepackage{iclr2027_conference}
    \iclrfinalcopy
    \author{Ada Lovelace <ada@example.org>}
    """
    checks = audit_source_text(source)
    assert not checks["no_final_copy"]
    assert not checks["no_identity_leak"]


def test_audit_archive_members_rejects_private_paths_and_raw_archives() -> None:
    checks = audit_archive_members(
        ["README.md", "snapshot/manifest.json", "/opt/projects/private.txt", "vendor.zip"]
    )
    assert not checks["no_private_paths"]
    assert not checks["no_raw_archives"]


def test_build_decision_is_conditional_until_manual_gates_close(tmp_path: Path) -> None:
    report = {
        "machine_checks": {"pdf": True, "supplement": True},
        "manual_gates": {
            "openreview_profile": "pending",
            "author_quota": "pending",
        },
        "scientific_gates": {
            "external_interactive_response": "open",
        },
    }
    decision = build_decision(report)
    assert decision["decision"] == "CONDITIONAL GO"
    assert decision["submission_ready"] is False
    assert decision["blocking_gates"]


def test_build_decision_goes_only_when_all_gates_pass() -> None:
    report = {
        "machine_checks": {"pdf": True, "supplement": True},
        "manual_gates": {"openreview_profile": "pass", "author_quota": "pass"},
        "scientific_gates": {"external_interactive_response": "pass"},
    }
    decision = build_decision(report)
    assert decision["decision"] == "GO"
    assert decision["submission_ready"] is True
    assert decision["blocking_gates"] == []


def test_audit_style_hashes_detects_modified_official_file(tmp_path: Path) -> None:
    style = tmp_path / "style"
    style.mkdir()
    source = style / "iclr2027_conference.sty"
    source.write_text("official", encoding="utf-8")
    manifest = tmp_path / "style_manifest.json"
    manifest.write_text(
        json.dumps({"files": {source.name: hashlib.sha256(b"official").hexdigest()}}),
        encoding="utf-8",
    )
    assert audit_style_hashes(style, manifest)
    source.write_text("modified", encoding="utf-8")
    assert not audit_style_hashes(style, manifest)


def test_spotlight_upgrade_source_contains_transition_first_narrative() -> None:
    source = Path("paper/iclr2027/main.tex").read_text(encoding="utf-8")
    required = (
        "replacement operation",
        "rank policies correctly while ranking improvements incorrectly",
        "Contribution 1",
        "Contribution 2",
        "Contribution 3",
        "Contribution 4",
        "Decision Preservation Under Differential Error",
        "Why Transition Validation Differs from Active Learning",
        "Stress Tests Beyond Controlled Environments",
        "Value Fidelity versus Improvement Fidelity",
        "Q_{\\mathcal A}",
        "operator-relative Improvement Fidelity",
        "\\operatorname{IF}(V,\\mathcal A;L)",
        "zero-to-positive reversal boundary",
        "false improvement (improvement reversal)",
        "CTI",
        "0/7",
        "0/5",
    )
    missing = [token for token in required if token not in source]
    assert not missing, f"missing spotlight narrative tokens: {missing}"
