from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.build_highway_evidence import audit_cohort, paired_contrast


def _rows() -> list[dict]:
    return [
        {"seed": seed, "budget": 2, "method": method, "ISR": value}
        for seed in (11, 22, 33)
        for method, value in (("Uniform HF", 0.2), ("Calibrated PIVOT-KG", 0.1))
    ]


def test_paired_contrast_preserves_sign_and_seed_pairing() -> None:
    result, rows = paired_contrast(_rows()[::-1], [11, 22, 33], budget=2)
    assert result["mean"] == pytest.approx(0.1)
    assert result["ci95"] == pytest.approx([0.1, 0.1])
    assert result["n_seed_pairs"] == 3
    assert [row["seed"] for row in rows] == [11, 22, 33]


def test_missing_seed_pair_fails_instead_of_silently_dropping_it() -> None:
    with pytest.raises(ValueError, match="seed pairs"):
        paired_contrast(_rows()[:-1], [11, 22, 33], budget=2)


def test_duplicate_seed_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        paired_contrast(_rows() + _rows()[:1], [11, 22, 33], budget=2)


def test_nonfinite_regret_is_rejected() -> None:
    rows = _rows()
    rows[0]["ISR"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_contrast(rows, [11, 22, 33], budget=2)


def test_rehashed_promotion_still_has_to_match_candidate_truth(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    shutil.copytree(Path("evidence/highway/original"), root)
    path = root / "promotion-results.json"
    records = json.loads(path.read_text())
    # Even a self-consistent forged regret plus updated file checksum must not
    # bypass the independent post-decision candidate audit.
    records["rows"][0]["selected_actor_delta"] += 0.01
    records["rows"][0]["ISR"] -= 0.01
    path.write_text(json.dumps(records))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {
        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="candidate truth"):
        audit_cohort(root)
