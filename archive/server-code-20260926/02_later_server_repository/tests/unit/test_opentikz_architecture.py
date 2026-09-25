from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.bootstrap_opentikz import bootstrap_opentikz

PROJECT = Path(__file__).parents[2]
FIGURE = PROJECT / "paper/iclr2027/figures/fig3_pivot_architecture.tex"
META = PROJECT / "paper/iclr2027/figures/fig3_pivot_architecture.meta.json"
LOCK = PROJECT / "configs/tooling/opentikz.json"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def test_bootstrap_pins_a_clean_local_checkout_and_refuses_drift(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "--initial-branch=main")
    _git(remote, "config", "user.email", "test@example.invalid")
    _git(remote, "config", "user.name", "OpenTikZ test")
    for relative, content in {
        "catalog.json": "[]\n",
        "skills/using-opentikz/SKILL.md": "skill\n",
        "templates/system-block-diagram/template.tex": "figure\n",
        "templates/system-block-diagram/template.meta.json": "{}\n",
    }.items():
        path = remote / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(remote, "add", ".")
    _git(remote, "commit", "-m", "initial")
    commit = _git(remote, "rev-parse", "HEAD")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "repository": str(remote),
                "commit": commit,
                "template_id": "system-block-diagram",
                "files": {
                    relative: hashlib.sha256((remote / relative).read_bytes()).hexdigest()
                    for relative in (
                        "catalog.json",
                        "skills/using-opentikz/SKILL.md",
                        "templates/system-block-diagram/template.tex",
                        "templates/system-block-diagram/template.meta.json",
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "installed"
    result = bootstrap_opentikz(lock, destination)
    assert result["commit"] == commit
    assert bootstrap_opentikz(lock, destination)["commit"] == commit

    _git(remote, "checkout", "-b", "drift")
    (remote / "catalog.json").write_text("[1]\n", encoding="utf-8")
    _git(remote, "add", "catalog.json")
    _git(remote, "commit", "-m", "drift")
    _git(destination, "fetch", "origin", "drift")
    _git(destination, "checkout", "--detach", "FETCH_HEAD")
    with pytest.raises(RuntimeError, match="different commit"):
        bootstrap_opentikz(lock, destination)


def test_architecture_source_preserves_opentikz_contract() -> None:
    source = FIGURE.read_text(encoding="utf-8")
    metadata = json.loads(META.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert metadata["composed_of"] == ["system-block-diagram"]
    assert metadata["license"] == "CC0-1.0"
    assert lock["template_id"] == "system-block-diagram"
    assert "\\documentclass[border=6pt]{standalone}" in source
    assert "\\pdftrailerid{5049564f542d415243482d3030303031}" in source
    for color in ("pblue", "porange", "pteal", "ppurple", "pgray"):
        assert color in source
    assert "fill=white" not in source
    assert "draw=black" not in source
    for node in (
        "inc",
        "op",
        "batch",
        "proxy",
        "posterior",
        "regret",
        "foot",
        "evsi",
        "cost",
        "paired",
        "stop",
    ):
        assert f"({node})" in source
    assert "expected regret" in source
    assert "expected regret reduction" in source
    assert "cost-aware acquisition" in source
    assert "same state, noise," in source
    assert "opponent initialization" in source
    assert "\\draw[hfarrow] (paired) -- (stop)" in source
    assert "\\draw[loop] (stop.east)" in source
    assert "CANDIDATE GENERATION AND DIFFERENTIAL POSTERIOR" in source
    assert "DECISION-SENSITIVE PAIRED INTERVENTION" in source


def test_rendered_architecture_contains_all_paper_layers(tmp_path: Path) -> None:
    output = tmp_path / "architecture.pdf"
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tmp_path}",
            FIGURE.name,
        ],
        cwd=FIGURE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "fig3_pivot_architecture.pdf"
    extracted = subprocess.check_output(["pdftotext", "-layout", str(output), "-"], text=True)
    extracted = re.sub(r"\s+", " ", extracted)
    for token in (
        "PIVOT",
            "candidate",
            "batch",
            "cheap",
            "verifier",
            "differential",
            "posterior",
            "expected",
            "regret",
            "reduction",
            "cost-aware",
            "acquisition",
            "paired",
            "rollout",
            "stop",
    ):
        assert token in extracted


def test_architecture_is_standalone_compilable(tmp_path: Path) -> None:
    output = tmp_path / "build"
    output.mkdir()
    result = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output}",
            FIGURE.name,
        ],
        cwd=FIGURE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "fig3_pivot_architecture.pdf").is_file()


def test_paper_places_architecture_in_main_text_and_world_diagnostics_in_appendix() -> None:
    source = (PROJECT / "paper/iclr2027/main.tex").read_text(encoding="utf-8")
    assert source.index("fig3_pivot_voi.pdf") < source.index("fig2_operator_shift.png")
    assert source.index("fig2_operator_shift.png") < source.index("fig4_evidence_efficiency.png")
    appendix = source.index("\\appendix")
    observer = source.index("figA_response_footprint.png")
    assert observer > appendix
