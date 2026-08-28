from __future__ import annotations

import json
from pathlib import Path

import pytest


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_cluster_bootstrap_uses_cluster_means_and_is_deterministic() -> None:
    from experiments.v15.scientific_analysis import cluster_bootstrap

    first = cluster_bootstrap([0.0, 0.0, 10.0], ["a", "a", "b"], seed=7, draws=300)
    second = cluster_bootstrap([0.0, 0.0, 10.0], ["a", "a", "b"], seed=7, draws=300)

    assert first == second
    # The independent units are the two cluster means (0 and 10), not rows.
    assert first["estimate"] == 5.0
    assert first["n_clusters"] == 2
    assert first["n_rows"] == 3


def test_transition_analysis_reports_sign_metrics_and_dev_terminal_state(tmp_path: Path) -> None:
    from experiments.v15.scientific_analysis import analyze_transition_artifact

    phase = tmp_path / "results/v15/dev-external-transition-audit"
    _jsonl(
        phase / "autonomous_transitions.jsonl",
        [
            {"run_id": "r1", "transition_id": "t1", "operator": "a", "task_family": "bug", "delta_proxy": 1.0, "delta_actor": -0.5, "footprint": {"prompt_token_delta": 2}},
            {"run_id": "r2", "transition_id": "t2", "operator": "b", "task_family": "tools", "delta_proxy": 1.0, "delta_actor": 0.5, "footprint": {"prompt_token_delta": 4}},
        ],
    )
    (phase / "manifest.json").write_text(
        json.dumps({"phase": "DEV", "status": "COMPLETED", "confirmatory": False, "execution_failure_count": 0, "outcome_chasing": False}),
        encoding="utf-8",
    )

    result = analyze_transition_artifact(tmp_path, min_clusters=30, draws=200)

    assert result["status"] == "DEV_ONLY"
    assert result["terminal_state"] == "UNDERPOWERED"
    assert result["metrics"]["IDE"]["estimate"] == 1.0
    assert result["metrics"]["IRR"]["estimate"] == 0.5
    assert result["metrics"]["ISC"]["estimate"] == 0.5
    assert result["outcome_chasing"] is False
    assert (tmp_path / "artifacts/v15/transition_analysis.json").is_file()
    assert (phase / "scientific_decision.json").is_file()


def test_transition_analysis_does_not_reconstruct_policy_rank_from_deltas(tmp_path: Path) -> None:
    from experiments.v15.scientific_analysis import analyze_transition_artifact

    phase = tmp_path / "results/v15/dev-external-transition-audit"
    rows = [
        {"run_id": "r1", "transition_id": "t1", "operator": "a", "task_family": "bug", "delta_proxy": 0.2, "delta_actor": 0.1},
        {"run_id": "r1", "transition_id": "t2", "operator": "a", "task_family": "bug", "delta_proxy": 0.3, "delta_actor": 0.2},
    ]
    _jsonl(phase / "autonomous_transitions.jsonl", rows)
    (phase / "manifest.json").write_text(
        json.dumps({"phase": "DEV", "status": "COMPLETED", "confirmatory": False, "outcome_chasing": False}),
        encoding="utf-8",
    )

    result = analyze_transition_artifact(tmp_path, min_clusters=30, draws=100)

    assert result["policy_level"]["available"] is False
    assert result["global_policy_value_fidelity_available"] is False


def test_transition_terminal_classifier_freezes_supported_and_null_results() -> None:
    from experiments.v15.scientific_analysis import classify_terminal_state

    supported = classify_terminal_state(
        attempted=True,
        confirmatory=True,
        design_valid=True,
        implementation_failures=0,
        n_clusters=30,
        estimate=0.4,
        ci_low=0.1,
        ci_high=0.7,
        alternative="positive",
    )
    null = classify_terminal_state(
        attempted=True,
        confirmatory=True,
        design_valid=True,
        implementation_failures=0,
        n_clusters=30,
        estimate=0.01,
        ci_low=-0.2,
        ci_high=0.2,
        alternative="positive",
    )

    assert supported == "HYPOTHESIS_SUPPORTED"
    assert null == "HYPOTHESIS_NOT_SUPPORTED"


def test_promotion_analysis_keeps_proxy_minus_pivot_orientation(tmp_path: Path) -> None:
    from experiments.v15.scientific_analysis import analyze_promotion_artifact

    phase = tmp_path / "results/v15/dev-external-promotion"
    _jsonl(
        phase / "promotion_results.jsonl",
        [
            {"method": "Proxy Only", "run_id": "r1", "round": 0, "hf_budget": 4, "ISR": 0.8, "candidate_batch_hash": "b1"},
            {"method": "PIVOT-VOI", "run_id": "r1", "round": 0, "hf_budget": 4, "ISR": 0.2, "candidate_batch_hash": "b1"},
            {"method": "Proxy Only", "run_id": "r2", "round": 0, "hf_budget": 4, "ISR": 0.6, "candidate_batch_hash": "b2"},
            {"method": "PIVOT-VOI", "run_id": "r2", "round": 0, "hf_budget": 4, "ISR": 0.1, "candidate_batch_hash": "b2"},
        ],
    )
    _jsonl(
        phase / "hf_queries.jsonl",
        [{"method": "PIVOT-VOI", "run_id": "r1", "round": 0, "logical_hf_query": True, "physical_pair_evaluation": True, "cache_hit": False}],
    )
    (phase / "manifest.json").write_text(
        json.dumps({"phase": "DEV", "status": "COMPLETED", "confirmatory": False, "execution_failure_count": 0, "outcome_chasing": False}),
        encoding="utf-8",
    )

    result = analyze_promotion_artifact(tmp_path, min_clusters=30, draws=200)

    assert result["terminal_state"] == "UNDERPOWERED"
    assert result["paired_effect"]["estimate"] == 0.55
    assert result["paired_effect"]["direction"] == "proxy_minus_pivot; positive_favors_pivot"
    assert result["query_accounting"]["logical_hf_queries"] == 1


def test_closed_loop_analysis_is_sealed_and_underpowered(tmp_path: Path) -> None:
    from experiments.v15.scientific_analysis import analyze_closed_loop_artifact

    phase = tmp_path / "results/v15/dev-external-closed-loop"
    _jsonl(
        phase / "closed_loop_results.jsonl",
        [
            {"method": "Proxy Only", "run_id": "r1", "round": 0, "CISR": 0.8},
            {"method": "PIVOT-VOI", "run_id": "r1", "round": 0, "CISR": 0.2},
        ],
    )
    _jsonl(
        phase / "assessment_results.jsonl",
        [
            {"method": "Proxy Only", "run_id": "r1", "assessment_score": 0.3, "queried_once": True, "role": "terminal_assessor"},
            {"method": "PIVOT-VOI", "run_id": "r1", "assessment_score": 0.7, "queried_once": True, "role": "terminal_assessor"},
        ],
    )
    (phase / "manifest.json").write_text(
        json.dumps({"phase": "DEV", "status": "COMPLETED", "confirmatory": False, "execution_failure_count": 0, "assessment_sealed_until_terminal": True, "terminal_assessment_exactly_once": True, "outcome_chasing": False}),
        encoding="utf-8",
    )

    result = analyze_closed_loop_artifact(tmp_path, min_clusters=30, draws=200)

    assert result["terminal_state"] == "UNDERPOWERED"
    assert result["assessment_audit"]["sealed"] is True
    assert result["endpoint_effect"]["estimate"] == pytest.approx(0.4)
    assert result["outcome_chasing"] is False
