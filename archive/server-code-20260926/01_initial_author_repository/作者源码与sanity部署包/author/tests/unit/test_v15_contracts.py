from __future__ import annotations

import json
from pathlib import Path


def test_claim_registry_has_evidence_contract_fields() -> None:
    import yaml

    payload = yaml.safe_load(Path("research/claims_v15.yaml").read_text(encoding="utf-8"))
    claims = payload["claims"]
    required = {
        "claim_id",
        "statement",
        "required_experiment",
        "required_terminal_state",
        "allowed_scope",
        "forbidden_scope",
        "paper_location",
    }
    assert all(required.issubset(item) for item in claims)


def test_canonical_transition_schema_has_flat_footprint_and_resource_columns() -> None:
    from experiments.v15.canonical import AUTONOMOUS_COLUMNS, FOOTPRINT_COLUMNS, RESOURCE_COLUMNS

    assert set(FOOTPRINT_COLUMNS).issubset({column.removeprefix("footprint_") for column in AUTONOMOUS_COLUMNS})
    assert set(RESOURCE_COLUMNS).issubset({column.removeprefix("resource_") for column in AUTONOMOUS_COLUMNS})


def test_write_table_flattens_nested_transition_metrics(tmp_path: Path) -> None:
    from experiments.v15.protocol import write_table

    outputs = write_table(
        [
            {
                "transition_id": "t1",
                "footprint": {"prompt_token_delta": 3.0},
                "resource_metrics": {"tokens": 17},
            }
        ],
        tmp_path / "transitions",
        columns=("transition_id", "footprint_prompt_token_delta", "resource_tokens"),
    )
    row = outputs["csv"].read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row == ["t1", "3.0", "17"]


def test_canonical_transition_table_drops_executor_paths() -> None:
    from experiments.v15.canonical import _flatten_transition

    row = {
        "transition_id": "t1",
        "footprint": {"prompt_token_delta": 3.0},
        "resource_metrics": {
            "proxy_candidate": {"tokens": 17},
            "candidate_final_tree_paths": ["/tmp/secret.final_tree"],
            "candidate_trajectories": ["/tmp/secret.traj.json"],
        },
    }

    flattened = _flatten_transition(row)

    assert flattened["resource_tokens"] == 17.0
    assert "secret" not in json.dumps(flattened["resource_metrics"])
    assert "candidate_final_tree_paths" not in flattened["resource_metrics"]


def test_external_adapter_contracts_are_dry_run_only() -> None:
    from experiments.v15.adapters import MiniSWEAdapter, PiAdapter

    for adapter in (MiniSWEAdapter(), PiAdapter()):
        status = adapter.status()
        assert status["execution_status"] == "NOT_RUN"
        assert status["model_calls_performed"] == 0
        assert adapter.command_preview("task-1", "policy-hash")["dry_run"] is True


def test_inspect_control_contract_never_executes_tasks() -> None:
    from experiments.v15.adapters.inspect_ai import InspectControlPlane

    control = InspectControlPlane()
    manifest = control.build_manifest(["proxy-1", "gate-1"], role="promotion")
    assert manifest["task_ids"] == ["proxy-1", "gate-1"]
    result = control.run(manifest)
    assert result["status"] == "NOT_RUN"
    assert result["model_calls_performed"] == 0


def test_figure_commands_accept_all_flag() -> None:
    import subprocess
    import sys

    for module in ("figures.v15.render", "figures.v15.audit", "figures.v15.iterate"):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--all", "--root", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        json.loads(completed.stdout)


def test_reviewer_attack_report_has_required_fields() -> None:
    from experiments.v15.reports import _reviewer_attack_table

    table = _reviewer_attack_table()
    header = table.splitlines()[0]
    assert "Severity" in header
    assert "Paper answer" in header
    assert "Remaining weakness" in header
    assert "Action required" in header
    assert sum(line.startswith("| A") for line in table.splitlines()[2:]) == 25


def test_master_loop_records_all_registered_phase_boundaries() -> None:
    from experiments.v15.master_loop import run_master_loop

    result = run_master_loop(Path("."))
    assert result["status"] == "BLOCKED"
    for phase in (
        "validate_mini_swe",
        "transitions",
        "promotion_replay",
        "closed_loop",
        "assessment",
        "pi_replication",
        "strategic",
        "ablations",
    ):
        assert phase in result["steps"]
    assert result["outcome_chasing"] is False


def test_confirmatory_config_declares_resource_and_lock_inputs() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    assert config["operator_prompts"]["forbidden_information"]
    assert config["sandbox"]["network"] == "disabled_by_default"
    assert config["resource_plan"]["estimated"]["inspect_evaluations"] > 0
    assert config["resource_plan"]["estimated"]["cost_usd"]


def test_validate_pi_module_entrypoint_is_available() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.v15.validate_pi", "--root", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["scaffold"] == "Pi"
    assert payload["status"] == "NOT_RUN"


def test_public_facades_expose_agent_agnostic_core_and_control_plane() -> None:
    import pivot_core
    import pivot_inspect

    assert pivot_core.PolicyTransition is not None
    assert pivot_core.RoundResult is not None
    assert pivot_core.run_pivot_voi_round is not None
    assert pivot_core.select_pivot_voi is not None
    assert pivot_inspect.InspectControlPlane().status()["execution_status"] == "NOT_RUN"
    assert pivot_inspect.MiniSWEAdapter().status()["execution_status"] == "NOT_RUN"
    assert pivot_inspect.PiAdapter().status()["execution_status"] == "NOT_RUN"


def test_master_loop_is_explicitly_available_from_command_surface() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.v15", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "master-loop" in completed.stdout


def test_pi_adapter_reports_built_cli_without_opening_a_model() -> None:
    from experiments.v15.adapters.pi import PiAdapter

    status = PiAdapter().status()
    assert status["execution_status"] == "NOT_RUN"
    assert status["model_calls_performed"] == 0
    assert "cli_path" in status


def test_control_plane_probe_recognizes_project_local_pi_cli() -> None:
    from experiments.v15.control_plane import probe_adapters

    statuses = {item.name: item for item in probe_adapters(Path("."))}
    pi = statuses["Pi"]
    assert pi.available is True
    assert pi.source == "project_local_probe"


def test_pi_command_preview_is_redacted_and_replayable() -> None:
    from experiments.v15.adapters.pi import PiAdapter

    preview = PiAdapter().command_preview("sealed-task", "policy-hash")
    assert preview["dry_run"] is True
    assert "<sealed-task>" in preview["command"]
    assert "policy-hash" in preview["command"]


def test_unflagged_external_phase_reports_intentional_dry_run(tmp_path: Path) -> None:
    from experiments.v15.commands import not_run

    result = not_run(tmp_path, "AUTONOMOUS_TRANSITION_AUDIT")

    assert result["execution_attempted"] is False
    assert "dry-run" in result["reason"]
    assert "unavailable" not in result["reason"]
    assert "--external --dev" in result["reason"]


def test_refresh_canonical_tables_materializes_dev_external_rows(tmp_path: Path) -> None:
    from experiments.v15.canonical import refresh_canonical_tables

    source = tmp_path / "results/v15/dev-external-transition-audit"
    source.mkdir(parents=True)
    transition = {
        "run_id": "run-1",
        "scaffold": "mini-SWE-agent",
        "operator": "operator-a",
        "task_family": "mixed",
        "round": 0,
        "transition_id": "transition-1",
        "incumbent_hash": "incumbent",
        "candidate_hash": "candidate",
        "delta_proxy": 0.1,
        "delta_actor": -0.2,
        "delta_strategic": None,
        "footprint": {"prompt_token_delta": 4.0},
        "resource_metrics": {"proxy_candidate": {"tokens": 12.0}},
    }
    (source / "autonomous_transitions.jsonl").write_text(json.dumps(transition) + "\n", encoding="utf-8")
    (source / "promotion_candidates.jsonl").write_text(json.dumps({"run_id": "run-1", "round": 0, "candidate_id": "candidate", "candidate_hash": "candidate", "proxy_delta": 0.1, "operator": "operator-a", "scaffold": "mini-SWE-agent"}) + "\n", encoding="utf-8")
    (source / "manifest.json").write_text(json.dumps({"phase": "DEV", "status": "COMPLETED"}) + "\n", encoding="utf-8")

    manifest = refresh_canonical_tables(tmp_path)

    assert manifest["phase"] == "DEV"
    assert manifest["rows"]["autonomous_transitions"] == 1
    table = (tmp_path / "results/v15/canonical/autonomous_transitions.csv").read_text(encoding="utf-8")
    assert "transition-1" in table
    assert "4.0" in table


def test_confirmatory_canonical_refresh_never_falls_back_to_dev(tmp_path: Path) -> None:
    from experiments.v15.canonical import refresh_canonical_tables

    source = tmp_path / "results/v15/dev-external-transition-audit"
    source.mkdir(parents=True)
    (source / "autonomous_transitions.jsonl").write_text(
        json.dumps({"transition_id": "dev-only"}) + "\n", encoding="utf-8"
    )
    (source / "manifest.json").write_text(
        json.dumps({"phase": "DEV", "status": "COMPLETED"}) + "\n", encoding="utf-8"
    )

    manifest = refresh_canonical_tables(tmp_path, confirmatory=True)

    assert manifest["confirmatory"] is False
    assert manifest["phase"] == "DEV"
    assert manifest["rows"]["autonomous_transitions"] == 0


def test_closed_loop_analysis_reports_canonical_phase(tmp_path: Path) -> None:
    from experiments.v15.commands import analyze_closed_loop

    canonical = tmp_path / "results/v15/canonical"
    canonical.mkdir(parents=True)
    (canonical / "closed_loop_results.csv").write_text("method\n\n", encoding="utf-8")
    (canonical / "manifest.json").write_text(
        json.dumps({"phase": "DEV"}) + "\n", encoding="utf-8"
    )

    result = analyze_closed_loop(tmp_path)

    assert result["status"] == "DEV_ONLY"
    assert result["phase"] == "DEV"
