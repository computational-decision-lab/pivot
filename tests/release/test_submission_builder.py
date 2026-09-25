"""Release exports must be deterministic, self-checking and confined."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_submission_release.py"


@pytest.fixture
def builder():
    if not SCRIPT.exists():
        pytest.skip("Release builder is intentionally excluded from anonymous supplementary ZIP")
    spec = importlib.util.spec_from_file_location("submission_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_is_deterministic_and_hashes_verify(builder, tmp_path):
    files = {"data/one.txt": (b"saved observations\n", False), "README.md": (b"instructions\n", False)}
    left, right = tmp_path / "a.zip", tmp_path / "b.zip"
    builder.write_archive(left, files, True)
    builder.write_archive(right, dict(reversed(list(files.items()))), True)
    assert left.read_bytes() == right.read_bytes()
    with zipfile.ZipFile(left) as archive:
        manifest = json.loads(archive.read("PACKAGE-MANIFEST.json"))
        assert manifest["anonymous"] is True
        for line in archive.read("SHA256SUMS").decode().splitlines():
            expected, path = line.split("  ", 1)
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected


def test_reject_external_symlink(builder, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (root / "code.py").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink leaves"):
        builder.collect(root, ["code.py"])


def test_reject_missing_inputs_and_unbundled_link_target(builder, tmp_path):
    with pytest.raises(FileNotFoundError):
        builder.collect(tmp_path, ["absent"])
    (tmp_path / "source.py").write_text("pass\n")
    (tmp_path / "alias.py").symlink_to("source.py")
    with pytest.raises(ValueError, match="target is not included"):
        builder.collect(tmp_path, ["alias.py"])


def test_reject_credential_without_printing_it(builder, tmp_path):
    # Synthetic token has the recognizable prefix/length, never a real credential.
    fake = "ghp_" + "X" * 32
    (tmp_path / "source.py").write_text(repr(fake))
    with pytest.raises(ValueError) as caught:
        builder.collect(tmp_path, ["source.py"])
    assert "source.py" in str(caught.value)
    assert fake not in str(caught.value)
