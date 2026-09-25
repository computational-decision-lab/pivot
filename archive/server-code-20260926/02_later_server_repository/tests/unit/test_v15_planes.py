from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_sealed_planes_enforce_role_boundaries() -> None:
    from experiments.v15.planes import AccessDenied, SealedDataPlanes, TaskSpec

    planes = SealedDataPlanes(
        proxy=(TaskSpec("p1", "bug_fixing", {"x": "1"}),),
        gate=(TaskSpec("g1", "bug_fixing", {"x": "2"}),),
        assessment=(TaskSpec("a1", "tool_intensive", {"x": "3"}),),
    )
    assert [task.task_id for task in planes.tasks("proxy", role="operator")] == ["p1"]
    with pytest.raises(AccessDenied):
        planes.tasks("gate", role="operator")
    with pytest.raises(AccessDenied):
        planes.tasks("assessment", role="pivot")
    assert [task.task_id for task in planes.tasks("gate", role="promotion")] == ["g1"]
    assert [task.task_id for task in planes.tasks("assessment", role="terminal_assessor")] == ["a1"]
    assert planes.access_log[-1]["role"] == "terminal_assessor"


def test_task_ids_must_be_disjoint() -> None:
    from experiments.v15.planes import SealedDataPlanes, TaskSpec

    with pytest.raises(ValueError, match="disjoint"):
        SealedDataPlanes(
            proxy=(TaskSpec("same", "a", {"a.txt": "a"}),),
            gate=(TaskSpec("same", "b", {"b.txt": "b"}),),
            assessment=(),
        )


def test_task_files_cannot_escape_the_sandbox() -> None:
    from experiments.v15.planes import TaskSpec

    with pytest.raises(ValueError, match="relative"):
        TaskSpec("bad", "bug_fixing", {"../outside.py": "x"})

    with pytest.raises(ValueError, match="relative"):
        TaskSpec("bad-absolute", "bug_fixing", {"/tmp/outside.py": "x"})


def test_sealed_plane_membership_cannot_be_reassigned() -> None:
    from dataclasses import FrozenInstanceError

    from experiments.v15.planes import SealedDataPlanes, TaskSpec

    planes = SealedDataPlanes(proxy=(TaskSpec("p", "f", {"a": "b"}),), gate=(), assessment=())
    with pytest.raises(FrozenInstanceError):
        planes.proxy = ()  # type: ignore[misc]


def test_task_manifest_loader_preserves_hashes_and_plane_boundaries(tmp_path) -> None:
    import json

    from experiments.v15.planes import load_task_planes

    manifest = {
        "sealed": True,
        "planes": {
            "proxy": [{"task_id": "p", "family": "f", "files": {"a.txt": "x"}}],
            "gate": [{"task_id": "g", "family": "f", "files": {"b.txt": "y"}}],
            "assessment": [{"task_id": "a", "family": "f", "files": {"c.txt": "z"}}],
        },
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    planes = load_task_planes(path)

    assert planes.manifest()["proxy"][0]["task_id"] == "p"
    assert planes.manifest()["assessment"][0]["task_hash"]


def test_task_manifest_loader_rejects_invalid_entry_and_hash(tmp_path) -> None:
    from experiments.v15.planes import load_task_planes

    invalid = {"sealed": True, "planes": {"proxy": ["not-a-task"], "gate": [], "assessment": []}}
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_task_planes(invalid_path)

    mismatch = {
        "sealed": True,
        "planes": {
            "proxy": [{"task_id": "p", "family": "f", "files": {"a": "b"}, "task_hash": "wrong"}],
            "gate": [],
            "assessment": [],
        },
    }
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_task_planes(mismatch_path)


def test_plane_manifest_and_access_snapshot_are_content_addressed() -> None:
    from experiments.v15.planes import SealedDataPlanes, TaskSpec

    planes = SealedDataPlanes(proxy=(TaskSpec("p", "f", {"a": "b"}),), gate=(), assessment=())
    planes.tasks("proxy", role="operator")
    snapshot = planes.access_log_snapshot()
    assert snapshot[0]["outcome"] == "granted"
    assert planes.manifest_sha256()
    snapshot[0]["role"] = "mutated"
    assert planes.access_log[0]["role"] == "operator"


def test_task_manifest_redaction_preserves_membership_without_contents() -> None:
    from experiments.v15.planes import redact_task_manifest

    payload = {
        "sealed": True,
        "planes": {
            "proxy": [
                {
                    "task_id": "p",
                    "family": "bug_fixing",
                    "files": {"example.py": "example content"},
                    "metadata": {"instruction": "redacted content"},
                }
            ],
            "gate": [{"task_id": "g", "family": "bug_fixing", "files": {"x": "y"}}],
            "assessment": [{"task_id": "a", "family": "tools", "files": {"z": "q"}}],
        },
    }

    redacted = redact_task_manifest(payload, source_sha256="source-digest")

    rendered = __import__("json").dumps(redacted, sort_keys=True)
    assert redacted["public_redaction"] is True
    assert redacted["source_sha256"] == "source-digest"
    assert redacted["plane_counts"] == {"proxy": 1, "gate": 1, "assessment": 1}
    assert "example.py" not in rendered
    assert "example content" not in rendered
    assert redacted["planes"]["proxy"][0]["task_hash"]


def test_task_manifest_redaction_is_idempotent() -> None:
    from experiments.v15.planes import redact_task_manifest

    summary = {
        "sealed": True,
        "public_redaction": True,
        "planes": {
            "proxy": [{"task_id": "p", "family": "f", "task_hash": "h"}],
            "gate": [],
            "assessment": [],
        },
    }

    assert redact_task_manifest(summary) == {
        "schema_version": "pivot-v15-task-manifest-public-1",
        "sealed": True,
        "public_redaction": True,
        "plane_counts": {"proxy": 1, "gate": 0, "assessment": 0},
        "planes": {
            "proxy": [{"task_id": "p", "task_hash": "h", "family": "f"}],
            "gate": [],
            "assessment": [],
        },
        "outcome_visibility": {
            "operator": ["proxy"],
            "pivot": ["proxy", "gate"],
            "promotion": ["proxy", "gate"],
            "terminal_assessor": ["assessment"],
        },
    }


def test_checked_in_public_manifest_contains_no_task_contents() -> None:
    public = json.loads(Path("configs/v15/task_manifest.public.json").read_text(encoding="utf-8"))
    rendered = json.dumps(public, sort_keys=True)

    assert public["public_redaction"] is True
    assert public["sealed"] is True
    assert public["plane_counts"] == {"proxy": 4, "gate": 4, "assessment": 4}
    assert "files" not in rendered
    assert "metadata" not in rendered
    assert "instruction" not in rendered
