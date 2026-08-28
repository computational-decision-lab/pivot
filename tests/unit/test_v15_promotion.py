from __future__ import annotations


def test_promotion_regret_orientation_and_common_archive() -> None:
    from experiments.v15.promotion import replay_methods

    rows = [
        {"run_id": "r", "round": 0, "candidate_index": 0, "candidate_id": "a", "candidate_hash": "a", "proxy_delta": 0.9, "footprint": {"prompt_semantic_distance": 0.1}},
        {"run_id": "r", "round": 0, "candidate_index": 1, "candidate_id": "b", "candidate_hash": "b", "proxy_delta": 0.4, "footprint": {"prompt_semantic_distance": 0.9}},
        {"run_id": "r", "round": 0, "candidate_index": 2, "candidate_id": "c", "candidate_hash": "c", "proxy_delta": 0.5, "footprint": {"prompt_semantic_distance": 0.8}},
    ]
    truth = {"a": 0.1, "b": 0.8, "c": 0.2}

    result = replay_methods(rows, truth, budgets=(1, 2), seed=7)

    assert {row["candidate_batch_hash"] for row in result["promotion_results"]}.__len__() == 1
    all_hf = next(row for row in result["promotion_results"] if row["method"] == "All-HF Oracle" and row["hf_budget"] == 2)
    assert all_hf["ISR"] == 0.0
    assert all_hf["selected_candidate"] == "b"
    assert all_hf["hf_cost"] == 3.0
    pivot = next(row for row in result["promotion_results"] if row["method"] == "PIVOT-VOI" and row["hf_budget"] == 2)
    assert pivot["ISR"] >= 0.0


def test_pivot_queries_only_registered_candidates() -> None:
    from experiments.v15.promotion import replay_methods

    rows = [
        {"run_id": "r", "round": 0, "candidate_index": 0, "candidate_id": "a", "candidate_hash": "a", "proxy_delta": 0.2, "footprint": {"prompt_semantic_distance": 0.1}},
        {"run_id": "r", "round": 0, "candidate_index": 1, "candidate_id": "b", "candidate_hash": "b", "proxy_delta": 0.1, "footprint": {"prompt_semantic_distance": 0.9}},
    ]

    result = replay_methods(rows, {"a": 0.2, "b": -0.1}, budgets=(1,), seed=1)
    queried = [row for row in result["hf_queries"] if row["method"] == "PIVOT-VOI"]
    assert len(queried) == 1
    assert queried[0]["candidate_id"] in {"a", "b"}


def test_expected_evsi_is_prequery_deterministic_regret_reduction() -> None:
    from experiments.v15.promotion import _initial_posterior, expected_evsi

    rows = [
        {
            "run_id": "r",
            "round": 0,
            "candidate_index": 0,
            "candidate_id": "a",
            "proxy_delta": 0.2,
            "footprint": {"prompt_semantic_distance": 0.1},
        },
        {
            "run_id": "r",
            "round": 0,
            "candidate_index": 1,
            "candidate_id": "b",
            "proxy_delta": 0.19,
            "footprint": {"prompt_semantic_distance": 0.9},
        },
    ]
    posterior = _initial_posterior(rows)
    first = expected_evsi(rows, posterior, rows[1], seed=7, fantasies=64, posterior_samples=128)
    second = expected_evsi(rows, posterior, rows[1], seed=7, fantasies=64, posterior_samples=128)

    assert first == second
    assert first >= 0.0
