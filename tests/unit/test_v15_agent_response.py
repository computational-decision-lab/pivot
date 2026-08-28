from __future__ import annotations

import json


def test_reviewer_prompt_is_identity_blind_and_uses_registered_inputs() -> None:
    from experiments.v15.agent_response import build_reviewer_prompt

    prompt = build_reviewer_prompt(
        patch="diff --git a/src/stats.py b/src/stats.py\n+ return sum(values) / len(values)",
        changed_interfaces=("src/stats.py",),
        execution_trace=("python -m unittest discover -s tests -v",),
        token_budget=256,
    )

    lowered = prompt.casefold()
    assert "src/stats.py" in prompt
    assert "python -m unittest" in prompt
    assert "candidate_id" not in lowered
    assert "gate" not in lowered
    assert "assessment" not in lowered
    assert "hypothesis" not in lowered
    assert "strategic" not in lowered


def test_parse_reviewer_response_rejects_unstructured_or_extra_fields() -> None:
    from experiments.v15.agent_response import parse_reviewer_response

    parsed = parse_reviewer_response(
        json.dumps(
            {
                "findings": [
                    {"path": "src/stats.py", "risk": "division edge case", "test": "empty input"}
                ]
            }
        )
    )
    assert parsed == ({"path": "src/stats.py", "risk": "division edge case", "test": "empty input"},)

    import pytest

    with pytest.raises(ValueError, match="JSON"):
        parse_reviewer_response("not-json")
    with pytest.raises(ValueError, match="unsupported"):
        parse_reviewer_response(json.dumps({"findings": [], "score": 1}))


def test_reviewer_request_digest_changes_with_patch() -> None:
    from experiments.v15.agent_response import reviewer_request_digest

    first = reviewer_request_digest(("src/a.py",), ("trace",), "patch-a")
    second = reviewer_request_digest(("src/a.py",), ("trace",), "patch-b")

    assert first != second
    assert len(first) == 64
