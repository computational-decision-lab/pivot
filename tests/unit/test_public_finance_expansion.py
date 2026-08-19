from __future__ import annotations

import pytest

from pivot.analysis.public_finance_expansion import (
    _normalized_roles,
    _validate_subconfig,
    aggregate_public_rows,
    run_public_finance_expansion,
)


def _row(
    asset: str,
    date: str,
    effect: float,
    delta_f1: float,
    reversal: bool,
    *,
    participation: float = 0.01,
    multiplier: float = 1.0,
) -> dict[str, object]:
    return {
        "asset": asset,
        "session_date": date,
        "participation_rate": participation,
        "impact_multiplier": multiplier,
        "depth_mechanical_effect": effect,
        "delta_f1": delta_f1,
        "depth_reversal_f1_to_f2": reversal,
        "delta_f2_depth": delta_f1 + effect,
        "causal_impact_identified": False,
    }


def test_aggregate_public_rows_reports_pooled_asset_and_holdout_metrics() -> None:
    rows = [
        _row("BTCUSDT", "2023-01-01", -0.2, 1.0, True),
        _row("BTCUSDT", "2023-07-01", -0.4, 2.0, False),
        _row("ETHUSDT", "2023-01-01", -0.1, 1.0, False),
        _row("ETHUSDT", "2023-07-01", -0.3, 0.0, False),
        _row("BTCUSDT", "2023-01-01", -9.0, 1.0, True, participation=0.05),
    ]

    summary = aggregate_public_rows(
        rows,
        expected_sessions=(
            ("BTCUSDT", "2023-01-01"),
            ("BTCUSDT", "2023-07-01"),
            ("ETHUSDT", "2023-01-01"),
            ("ETHUSDT", "2023-07-01"),
        ),
        primary_participation=0.01,
        primary_impact_multiplier=1.0,
        holdout_dates=("2023-07-01",),
    )

    assert summary["complete_grid"] is True
    assert summary["n_primary_sessions"] == 4
    assert summary["n_f1_positive_sessions"] == 3
    assert summary["n_depth_reversal_sessions"] == 1
    assert summary["depth_reversal_rate_given_f1_positive"] == pytest.approx(1 / 3)
    assert summary["pooled_depth_mechanical_effect"]["estimate"] == pytest.approx(-0.25)
    assert summary["holdout"]["n_primary_sessions"] == 2
    assert summary["per_asset"]["BTCUSDT"]["n_primary_sessions"] == 2
    assert summary["causal_impact_identified"] is False
    assert summary["ground_truth_for_endogenous_response"] is False
    assert summary["gate_e_promoted"] is False


def test_aggregate_public_rows_rejects_duplicate_primary_session() -> None:
    rows = [
        _row("BTCUSDT", "2023-01-01", -0.2, 1.0, False),
        _row("BTCUSDT", "2023-01-01", -0.3, 1.0, False),
    ]

    with pytest.raises(ValueError, match="duplicate primary session"):
        aggregate_public_rows(
            rows,
            expected_sessions=(("BTCUSDT", "2023-01-01"),),
            primary_participation=0.01,
            primary_impact_multiplier=1.0,
            holdout_dates=(),
        )


def test_aggregate_public_rows_marks_missing_sessions_without_silent_completion() -> None:
    summary = aggregate_public_rows(
        [_row("BTCUSDT", "2023-01-01", -0.2, 1.0, False)],
        expected_sessions=(
            ("BTCUSDT", "2023-01-01"),
            ("ETHUSDT", "2023-01-01"),
        ),
        primary_participation=0.01,
        primary_impact_multiplier=1.0,
        holdout_dates=(),
    )

    assert summary["complete_grid"] is False
    assert summary["missing_sessions"] == [["ETHUSDT", "2023-01-01"]]


def test_normalized_roles_accept_yaml_date_keys() -> None:
    import datetime

    assert _normalized_roles({datetime.date(2023, 7, 1): "frozen_holdout_calendar_block"}) == {
        "2023-07-01": "frozen_holdout_calendar_block"
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("participation_rates", [0.0, 0.02], "participation_rates differs"),
        ("impact_multipliers", [1.0], "impact_multipliers differs"),
        ("execution", {"fee_bps": 9.0}, "execution differs"),
    ],
)
def test_validate_subconfig_rejects_frozen_grid_drift(
    tmp_path, field: str, replacement: object, message: str
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
symbol: BTCUSDT
sessions:
  - date: '2023-01-01'
""",
        encoding="utf-8",
    )
    config = {
        "dataset_manifest": str(manifest),
        "edit_type": "position_size",
        "incumbent": {"position_size": 0.2},
        "candidate": {"position_size": 0.6},
        "participation_rates": [0.0, 0.01],
        "impact_multipliers": [0.5, 1.0],
        "primary_impact_multiplier": 1.0,
        "target_participation": 0.01,
        "execution": {"fee_bps": 4.0},
    }
    config[field] = replacement
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    grid = {
        "dates": ["2023-01-01"],
        "update": {
            "edit_type": "position_size",
            "incumbent": {"position_size": 0.2},
            "candidate": {"position_size": 0.6},
        },
        "evaluation": {
            "participation_rates": [0.0, 0.01],
            "impact_multipliers": [0.5, 1.0],
            "primary_participation": 0.01,
            "primary_impact_multiplier": 1.0,
            "execution": {"fee_bps": 4.0},
        },
    }

    with pytest.raises(ValueError, match=message):
        _validate_subconfig(
            config_path,
            expected_asset="BTCUSDT",
            grid=grid,
            primary_participation=0.01,
            primary_multiplier=1.0,
        )


def test_run_public_expansion_refuses_nonempty_output(tmp_path) -> None:
    config = tmp_path / "invalid-but-must-not-overwrite.yaml"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_public_finance_expansion(config, output)
