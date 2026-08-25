from __future__ import annotations

import subprocess
import sys


def test_reproduce_paper_check_only_verifies_frozen_artifacts() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py", "--check-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"valid": true' in result.stdout
