from __future__ import annotations

from pathlib import Path

import pytest

from pivot.analysis.registry import load_registry, materialize_seed_config


def test_registry_requires_disjoint_nonempty_seed_sets(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
experiment: p2
base_config: base.yaml
seed_sets:
  - run_id: r1
    seeds: [1, 2]
  - run_id: r2
    seeds: [2, 3]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        load_registry(path)


def test_materialize_seed_config_does_not_mutate_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("world: {response_strength: 0.7}\nseeds: [1]\n", encoding="utf-8")
    output = tmp_path / "run.yaml"
    materialize_seed_config(base, [9, 10], output)
    assert "seeds: [1]" in base.read_text(encoding="utf-8")
    assert "seeds:" in output.read_text(encoding="utf-8")
    assert "9" in output.read_text(encoding="utf-8")


def test_registry_accepts_later_experiments(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(
        "experiment: e7\nbase_config: base.yaml\nseed_sets:\n  - run_id: r1\n    seeds: [1]\n",
        encoding="utf-8",
    )
    assert load_registry(path)["experiment"] == "e7"
