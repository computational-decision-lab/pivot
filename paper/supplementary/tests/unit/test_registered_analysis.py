from __future__ import annotations

import json
from pathlib import Path

import pytest

from pivot.analysis.registered import (
    evaluate_gate_e,
    evaluate_gate_f,
    paired_bootstrap_mean_ci,
    summarize_e4_runs,
    summarize_e5_runs,
    summarize_e6_runs,
    summarize_e7_runs,
    summarize_e8_runs,
    summarize_e9_runs,
    summarize_p2_runs,
)


def test_paired_bootstrap_ci_uses_within_pair_differences() -> None:
    result = paired_bootstrap_mean_ci([10.0, 20.0, 30.0], [9.0, 18.0, 27.0], seed=7)
    assert result["estimate"] == pytest.approx(2.0)
    assert result["n"] == 3
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_p2_summary_reports_independent_run_and_response_contrast(tmp_path: Path) -> None:
    for index, seed in enumerate((11, 22, 33), start=1):
        run = tmp_path / f"p2-r{index}"
        run.mkdir()
        rows = [
            {"delta_proxy": 1.0, "delta_true": 0.5, "response_strength": 0.0, "update_footprint": 0.2},
            {"delta_proxy": 1.0, "delta_true": -0.5, "response_strength": 0.7, "update_footprint": 0.2},
        ]
        (run / "transitions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run / "provenance.json").write_text(
            json.dumps({"run_id": run.name, "seeds": [seed], "config_sha256": f"hash-{index}"}),
            encoding="utf-8",
        )
    summary = summarize_p2_runs(sorted(tmp_path.iterdir()), min_response_strength=0.3)
    assert summary["run_count"] == 3
    assert summary["high_response_irr"]["estimate"] == pytest.approx(1.0)
    assert summary["response_contrast"]["estimate"] == pytest.approx(1.0)
    assert summary["config_hashes"] == ["hash-1", "hash-2", "hash-3"]


def test_p2_summary_retains_failed_run_without_crashing(tmp_path: Path) -> None:
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "run_manifest.json").write_text(
        json.dumps({"run_id": "failed", "status": "failed", "seeds": [99]}),
        encoding="utf-8",
    )
    summary = summarize_p2_runs([failed])
    assert summary["run_count"] == 1
    assert summary["valid_run_count"] == 0
    assert summary["run_ids"] == ["failed"]


def test_e4_and_e5_summaries_keep_paired_method_differences(tmp_path: Path) -> None:
    e4_runs: list[Path] = []
    e5_runs: list[Path] = []
    for index in range(3):
        e4 = tmp_path / f"e4-{index}"
        e4.mkdir()
        (e4 / "comparison.json").write_text(
            json.dumps(
                {
                    "policy_value_mae": 1.0,
                    "global_improvement_sign_consistency": 0.5,
                    "improvement_sign_consistency": 0.8,
                    "global_improvement_differential_error": 2.0,
                    "improvement_differential_error": 1.0,
                    "global_update_selection_regret": 2.0,
                    "update_selection_regret": 1.0,
                }
            ),
            encoding="utf-8",
        )
        e4_runs.append(e4)
        e5 = tmp_path / f"e5-{index}"
        e5.mkdir()
        rows = []
        for group in ("g0", "g1"):
            rows.extend(
                [
                    {"method": "random_hf", "budget": 1, "group_id": group, "cti": 0.0, "isr": 2.0},
                    {"method": "top_proxy_hf", "budget": 1, "group_id": group, "cti": 0.5, "isr": 1.5},
                    {"method": "pivot", "budget": 1, "group_id": group, "cti": 1.0, "isr": 1.0},
                ]
            )
        (e5 / "group_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        e5_runs.append(e5)
    e4_summary = summarize_e4_runs(e4_runs)
    e5_summary = summarize_e5_runs(e5_runs, target_budget=1)
    assert e4_summary["local_minus_global_isc"]["estimate"] == pytest.approx(0.3)
    assert e4_summary["local_minus_global_ide"]["estimate"] == pytest.approx(-1.0)
    assert e4_summary["local_minus_global_isr"]["estimate"] == pytest.approx(-1.0)
    assert e5_summary["pivot_minus_random_cti"]["estimate"] == pytest.approx(1.0)
    assert e5_summary["random_minus_pivot_isr"]["estimate"] == pytest.approx(1.0)


def test_e6_summary_reports_zero_participation_equivalence_and_target_effect(tmp_path: Path) -> None:
    runs: list[Path] = []
    for index in range(3):
        run = tmp_path / f"e6-{index}"
        run.mkdir()
        rows = [
            {"seed": 1, "participation_rate": 0.0, "delta_f1": 1.0, "delta_f2": 1.0},
            {"seed": 1, "participation_rate": 0.05, "delta_f1": 1.0, "delta_f2": 0.5},
        ]
        (run / "finance_actor.json").write_text(json.dumps(rows), encoding="utf-8")
        (run / "run_manifest.json").write_text(
            json.dumps({"run_id": f"e6-{index}", "seeds": [index + 1]}), encoding="utf-8"
        )
        runs.append(run)
    summary = summarize_e6_runs(runs, target_participation=0.05)
    assert summary["zero_participation_f2_minus_f1"]["estimate"] == pytest.approx(0.0)
    assert summary["target_f2_minus_f1"]["estimate"] == pytest.approx(-0.5)
    assert evaluate_gate_e(summary)["E"] == "Pass"


def test_e7_e8_e9_summaries_report_strategic_and_closed_loop_effects(tmp_path: Path) -> None:
    e7_runs: list[Path] = []
    e8_runs: list[Path] = []
    e9_runs: list[Path] = []
    for index in range(3):
        e7 = tmp_path / f"e7-{index}"
        e7.mkdir()
        (e7 / "strategic_reversal.json").write_text(
            json.dumps(
                [
                    {
                        "delta_actor": 1.0,
                        "delta_strategic": -1.0,
                        "competition_effect": -2.0,
                        "strategic_improvement_reversal": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        e7_runs.append(e7)
        e8 = tmp_path / f"e8-{index}"
        e8.mkdir()
        (e8 / "competition.json").write_text(
            json.dumps(
                [
                    {
                        "mode": "adaptive",
                        "strategic_sensitivity": 0.01,
                        "delta_actor": 1.0,
                        "delta_strategic": 0.0,
                        "competition_effect": -1.0,
                        "strategic_improvement_reversal": False,
                    },
                    {
                        "mode": "adaptive",
                        "strategic_sensitivity": 0.1,
                        "delta_actor": 1.0,
                        "delta_strategic": -1.0,
                        "competition_effect": -2.0,
                        "strategic_improvement_reversal": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        e8_runs.append(e8)
        e9 = tmp_path / f"e9-{index}"
        e9.mkdir()
        (e9 / "closed_loop.json").write_text(
            json.dumps([{"round_id": 0, "hf_budget": 1, "selected_delta_true": 0.2}]),
            encoding="utf-8",
        )
        e9_runs.append(e9)
    e7 = summarize_e7_runs(e7_runs)
    e8 = summarize_e8_runs(e8_runs)
    e9 = summarize_e9_runs(e9_runs)
    assert e7["sirr"]["estimate"] == pytest.approx(1.0)
    assert e8["competition_effect"]["estimate"] == pytest.approx(-1.5)
    assert e8["sensitivity_contrast"]["estimate"] == pytest.approx(-1.0)
    assert evaluate_gate_f(e7, e8)["F"] == "Pass"
    assert e9["mean_rounds"] ["estimate"] == pytest.approx(1.0)
