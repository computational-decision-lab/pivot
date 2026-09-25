"""Public reproduction CLI behavior from a directory outside the checkout."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "reproduction" / "run.py"


def call(tmp_path, *args):
    return subprocess.run([sys.executable, str(LAUNCHER), *args], cwd=tmp_path,
                          text=True, capture_output=True)


def test_analyze_from_external_directory(tmp_path):
    output = tmp_path / "out"
    result = call(tmp_path, "analyze", "--experiment", "leduc_v4", "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "saved-analysis.json").read_text())
    assert report["results"]["leduc_v4"]["scored_rows"] == 3960
    assert report["results"]["scope"].startswith("Recomputed from saved labels")
    assert json.loads((output / "saved-evidence-verification.json").read_text())["input_files"] == 1051


def test_analyze_isolates_core_cohort(tmp_path):
    output = tmp_path / "out"
    result = call(tmp_path, "analyze", "--experiment", "kuhn", "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "saved-analysis.json").read_text())["results"]
    assert report["selected_core_cohort"]["roots"] == 30
    assert report["selected_core_cohort"]["horizon"] == 8
    assert report["core"]["roots"] == 90
    assert report["table1_comparison"]["replayed_decisions"] == 60


def test_full_requires_named_cohort_and_fresh_output(tmp_path):
    missing = call(tmp_path, "full", "--output", str(tmp_path / "new"))
    assert missing.returncode != 0
    assert "requires --experiment" in missing.stderr
    (tmp_path / "existing").mkdir()
    occupied = call(tmp_path, "full", "--experiment", "kuhn", "--output", str(tmp_path / "existing"))
    assert occupied.returncode != 0
    assert "new output directory" in occupied.stderr


def test_missing_full_inputs_fail_explicitly(tmp_path):
    result = call(tmp_path, "full", "--experiment", "meltingpot", "--output", str(tmp_path / "new"))
    assert result.returncode != 0
    assert "specialist model" in result.stderr
    assert not (tmp_path / "new").exists()


def test_smoke_reports_native_scope(tmp_path):
    output = tmp_path / "smoke"
    result = call(tmp_path, "smoke", "--experiment", "controlled", "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "smoke" / "controlled.json").read_text())
    assert report["status"] == "passed"
    assert report["native_environment_step"] is False
    assert report["environment_steps"] > 0


def test_reject_evidence_as_output(tmp_path):
    result = call(tmp_path, "verify", "--output", str(ROOT / "evidence" / "paper" / "bad"))
    assert result.returncode != 0
    assert "immutable evidence" in result.stderr


def test_saved_verifier_rejects_missing_and_corrupt_inputs(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.json").write_text(json.dumps({"files": [
        {"path": "sample.txt", "sha256": "0" * 64}]}))
    script = ROOT / "reproduction" / "verify_saved_evidence.py"
    for contents in (None, "changed"):
        if contents is not None:
            (evidence / "sample.txt").write_text(contents)
        result = subprocess.run([sys.executable, str(script), "--evidence", str(evidence),
                                 "--output", str(tmp_path / "result.json")],
                                cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode != 0
        assert not (tmp_path / "result.json").exists()
