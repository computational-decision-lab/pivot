from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_iclr_supplement import (
    ALLOWLIST,
    LATEST_EVIDENCE_SOURCES,
    _copy_hash_bound_tree,
    _copy_sanitized,
)
from scripts.build_revision_evidence import public_audit
from scripts.verify_iclr_submission import (
    _aux_path,
    _portable_path,
    audit_archive_contents,
    audit_archive_members,
    audit_revision_evidence_archive,
    audit_source_text,
    audit_style_hashes,
    build_decision,
)


def test_hash_bound_tree_preserves_json_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    content = b'{ "b": 1, "a": 2 }\n'
    (source / "rows.json").write_bytes(content)
    _copy_hash_bound_tree(source, tmp_path / "export")
    assert (tmp_path / "export/rows.json").read_bytes() == content


def test_latest_supplement_inventory_contains_three_sealed_cohorts() -> None:
    assert set(LATEST_EVIDENCE_SOURCES) == {"melting_v4", "leduc_v2b", "kuhn_v2b"}
    for relative in LATEST_EVIDENCE_SOURCES.values():
        assert relative.is_dir()
        for name in (
            "summary.json",
            "seed_results.csv",
            "decisions_sealed.json",
            "scored_decisions.json",
            "posterior_v2_spec.json",
            "selection_seal.json",
        ):
            assert (relative / name).is_file()


def test_public_revision_audit_redacts_private_omission_names() -> None:
    source = {
        "archive_sha256": "archive-hash",
        "review_sha256": "review-hash",
        "manifest": {
            "entries": 687,
            "verified": 685,
            "missing": ["connection.json", "private-account.md"],
            "expected_missing": ["connection.json", "private-account.md"],
        },
        "seals": {},
        "recomputed_from_scored_rows": {},
        "recomputation_checks": {"headline": True},
        "summaries": {},
        "status_boundary": {},
    }

    rendered = json.dumps(public_audit(source), sort_keys=True)

    assert "connection.json" not in rendered
    assert "private-account.md" not in rendered
    assert public_audit(source)["manifest"] == {
        "entries": 687,
        "verified": 685,
        "private_omissions": 2,
    }


def test_revision_archive_audit_accepts_three_consistent_sealed_cohorts(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "supplement.zip"
    checks = {
        "melting_v4_primary": True,
        "melting_v4_interaction": True,
        "melting_v4_one_query": True,
        "melting_v4_short": True,
        "leduc_primary": True,
        "kuhn_primary": True,
    }
    public = {
        "archive_sha256": "a" * 64,
        "review_sha256": "b" * 64,
        "manifest": {"entries": 687, "verified": 685, "private_omissions": 2},
        "seals": {},
        "recomputation_checks": checks,
    }
    with zipfile.ZipFile(archive, "w") as output:
        for cohort in ("melting_v4", "leduc_v2b", "kuhn_v2b"):
            decisions = [{"root": 7, "method": "pivot_kg"}]
            scored = [{"root": 7, "method": "pivot_kg", "gain": 1.0}]
            decision_bytes = json.dumps(decisions).encode()
            seal = {
                "decisions_sha256": hashlib.sha256(decision_bytes).hexdigest(),
                "n_decisions": 1,
                "test_roots": [7],
            }
            public["seals"][cohort] = {
                "decisions_sha256": seal["decisions_sha256"],
                "n_decisions": 1,
                "roots": [7],
            }
            prefix = f"evidence/latest/{cohort}"
            output.writestr(f"{prefix}/decisions_sealed.json", decision_bytes)
            output.writestr(f"{prefix}/scored_decisions.json", json.dumps(scored))
            output.writestr(f"{prefix}/selection_seal.json", json.dumps(seal))
        output.writestr(
            "evidence/latest/revision_evidence_audit.json",
            json.dumps(public),
        )

    result = audit_revision_evidence_archive(archive)

    assert result["valid"] is True
    assert result["cohorts_verified"] == 3


def test_finalizer_normalizes_intermediate_supplement_digests() -> None:
    from experiments.v10.finalize import _stable_build_steps

    steps = [
        {
            "command": ["python", "scripts/build_iclr_supplement.py"],
            "stdout_tail": [
                (
                    '{"archive": "pivot_iclr2027_supplementary.zip", '
                    '"sha256": "changing", "snapshot_manifest_sha256": "stable"}'
                )
            ],
        }
    ]

    normalized = _stable_build_steps(steps)
    line = normalized[0]["stdout_tail"][0]
    payload = json.loads(line)
    assert payload["archive"] == "pivot_iclr2027_supplementary.zip"
    assert payload["snapshot_manifest_sha256"] == "stable"
    assert "sha256" not in payload


def test_aux_path_falls_back_to_build_sidecar_for_copied_submission(tmp_path: Path) -> None:
    submission = tmp_path / "paper" / "pivot_submission.pdf"
    submission.parent.mkdir(parents=True)
    build_aux = submission.parent / "build" / "main.aux"
    build_aux.parent.mkdir()
    build_aux.write_text(r"\\newlabel{refs:start}{{}{10}}", encoding="utf-8")

    assert _aux_path(submission) == build_aux


def test_portable_path_avoids_machine_absolute_prefix() -> None:
    rendered = _portable_path(Path.cwd() / "paper" / "main.tex")
    assert rendered == "paper/main.tex"
    assert "/opt/projects/" not in rendered


def test_supplement_allowlist_contains_improvementbench_v2() -> None:
    assert "benchmarks/improvementbench/v2" in ALLOWLIST


def test_supplement_allowlist_contains_v7_artifacts() -> None:
    assert "benchmarks/improvementbench/v7" in ALLOWLIST
    assert "results/v7" in ALLOWLIST
    assert "research" in ALLOWLIST


def test_supplement_allowlist_contains_registered_theory_artifact() -> None:
    assert "results/theory" in ALLOWLIST


def test_supplement_sanitizer_redacts_runtime_endpoint(tmp_path: Path) -> None:
    source = tmp_path / "runtime.json"
    target = tmp_path / "out" / "runtime.json"
    source.write_text('{"api_base": "https://private.example.invalid"}\n', encoding="utf-8")

    _copy_sanitized(source, target)

    rendered = target.read_text(encoding="utf-8")
    assert "private.example.invalid" not in rendered
    assert '"api_base": "<external-endpoint>"' in rendered


def test_supplement_sanitizer_redacts_jsonl_paths(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    target = tmp_path / "out" / "trace.jsonl"
    source.write_text('{"path": "/opt/projects/private/trace.json"}\n', encoding="utf-8")

    _copy_sanitized(source, target)

    assert "/opt/projects" not in target.read_text(encoding="utf-8")


def test_supplement_sanitizer_redacts_sealed_task_contents(tmp_path: Path) -> None:
    source = tmp_path / "task_manifest.json"
    target = tmp_path / "out" / "task_manifest.json"
    source.write_text(
        json.dumps(
            {
                "sealed": True,
                "planes": {
                    "proxy": [
                        {
                            "task_id": "p",
                            "family": "bug_fixing",
                            "files": {"example.py": "example task content"},
                            "metadata": {"instruction": "redacted fixture"},
                        }
                    ],
                    "gate": [],
                    "assessment": [],
                },
            }
        ),
        encoding="utf-8",
    )

    _copy_sanitized(source, target, Path("configs/v15/task_manifest.json"))

    payload = json.loads(target.read_text(encoding="utf-8"))
    rendered = target.read_text(encoding="utf-8")
    assert payload["public_redaction"] is True
    assert "example.py" not in rendered
    assert "example task content" not in rendered
    assert payload["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_supplement_sanitizer_redacts_sealed_lock_contents(tmp_path: Path) -> None:
    source = tmp_path / "confirmatory_lock.json"
    target = tmp_path / "out" / "confirmatory_lock.json"
    source.write_text(
        json.dumps(
            {
                "confirmatory_execution": "NOT_RUN",
                "sealed_planes": {
                    "planes": {
                        "proxy": [
                            {
                                "task_id": "p",
                                "family": "f",
                                "files": {"hidden.txt": "hidden"},
                            }
                        ],
                        "gate": [],
                        "assessment": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    _copy_sanitized(source, target, Path("experiments/v15/confirmatory_lock.json"))

    rendered = target.read_text(encoding="utf-8")
    assert "hidden.txt" not in rendered
    assert "hidden" not in rendered
    assert "public_redaction" in rendered


def test_supplement_allowlist_excludes_phase_level_v15_parquet() -> None:
    from scripts.build_iclr_supplement import (
        _is_public_parquet,
        _is_public_v15_path,
        _is_sealed_public_input,
    )

    assert _is_public_parquet(Path("results/v15/canonical/autonomous_transitions.parquet"))
    assert _is_public_parquet(Path("figures/v15/fig1/figure.parquet"))
    assert not _is_public_parquet(Path("results/v15/dev-external-transition-audit/autonomous_transitions.parquet"))
    assert _is_public_v15_path(Path("results/v15/dev-external-transition-audit/manifest.json"))
    assert not _is_public_v15_path(
        Path("results/v15/dev-external-transition-audit/artifacts/task.traj.json")
    )
    assert not _is_public_v15_path(
        Path("results/v15/dev-external-transition-audit/artifacts/task.execution.json")
    )
    assert _is_sealed_public_input(Path("configs/v15/task_manifest.json"))
    assert _is_sealed_public_input(Path("experiments/v15/confirmatory_lock_history.jsonl"))
    assert not _is_sealed_public_input(Path("configs/v15/task_manifest.public.json"))


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


def test_audit_archive_members_rejects_sealed_task_inputs() -> None:
    checks = audit_archive_members(
        ["README.md", "snapshot/manifest.json", "configs/v15/task_manifest.json"]
    )
    assert not checks["no_sealed_inputs"]
    assert checks["sealed_members"] == ["configs/v15/task_manifest.json"]


def test_audit_archive_contents_rejects_private_path_in_jsonl(tmp_path: Path) -> None:
    archive = tmp_path / "supplement.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("trace.jsonl", '{"path": "/opt/projects/private/trace"}\n')

    checks = audit_archive_contents(archive)

    assert not checks["valid"]
    assert checks["private_content_hits"] == ["trace.jsonl"]


def test_audit_archive_contents_accepts_clean_archive(tmp_path: Path) -> None:
    archive = tmp_path / "supplement.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("README.md", "public artifact\n")

    checks = audit_archive_contents(archive)

    assert checks["valid"]


def test_audit_archive_contents_rejects_forbidden_assistant_token(tmp_path: Path) -> None:
    archive = tmp_path / "supplement.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("notes.txt", "codex\n")

    checks = audit_archive_contents(archive)

    assert not checks["valid"]
    assert checks["assistant_content_hits"] == ["notes.txt"]


def test_audit_archive_members_allows_generated_v15_parquet_sources() -> None:
    checks = audit_archive_members(["results/v15/canonical/table.parquet", "README.md"])

    assert checks["no_raw_archives"]
    assert checks["generated_parquet_members"] == ["results/v15/canonical/table.parquet"]


def test_archive_accepts_paper_figure_tables_but_rejects_raw_parquet() -> None:
    checks = audit_archive_members(["paper/figures/release/figure.parquet", "README.md"])
    assert checks["no_raw_archives"]
    assert not audit_archive_members(["data/raw/vendor.parquet"])["no_raw_archives"]


def test_generated_parquet_inventory_is_not_a_submission_gate() -> None:
    report = {
        "machine_checks": {
            "pdf": True,
            "supplement_archive": True,
            "archive_no_raw_archives": True,
            "archive_generated_parquet_members": ["results/v15/canonical/table.parquet"],
        },
        "manual_gates": {},
        "scientific_gates": {},
    }

    decision = build_decision(report)

    assert decision["blocking_gates"] == []
    assert decision["submission_ready"] is True


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
    source = Path("paper/main.tex").read_text(encoding="utf-8")
    required = (
        "replacement operation",
        "rank policies correctly while ranking improvements incorrectly",
        "Contribution 1",
        "Contribution 2",
        "Contribution 3",
        "Operator Shift Bound",
        "Paired variance",
        "Decision preservation",
        "\\section{PIVOT-KG}",
        "Stress Tests Beyond Controlled Environments",
        "Uniform value fidelity is sufficient",
        "Q_{\\mathcal A}",
        "operator-relative Improvement Fidelity",
        "\\operatorname{IF}(V,\\mathcal A;L)",
        "raw sampled reversal-rate cells",
        "improvement reversal",
        "CTI",
        "0/7",
        "0/5",
    )
    missing = [token for token in required if token not in source]
    assert not missing, f"missing spotlight narrative tokens: {missing}"
