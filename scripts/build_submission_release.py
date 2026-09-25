#!/usr/bin/env python3
"""Build reproducibility ZIPs from explicit inputs, without changing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "iclr2027-repro-v1"
SKIP_PARTS = {".git", ".venv", "__pycache__", "__MACOSX", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".tex", ".bib", ".sty", ".bst", ".cls"}
SECRET = re.compile(rb"\b(?:olp_[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
IDENTITY = re.compile(r"computational-decision-lab|git\.overleaf\.com|overleaf\.com/project|/Users/[^/\s]+|\bcolin_pivot\b", re.I)
DOCS = ("docs/reviewer-guide.md", "docs/reproduction.md", "docs/figure-map.md", "docs/rights-and-dependencies.md", "docs/comparison-audit.md", "docs/protocol-chronology.md")
CORE_TESTS = ("test_metrics.py", "test_transition.py", "test_decomposition.py", "test_acquisition.py", "test_operator_shift.py", "test_pivot_voi.py", "test_validation.py", "test_sample_complexity.py", "test_transfer.py", "test_v9_environments.py", "test_v9_operators.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed(path: Path) -> bool:
    return (not any(x in SKIP_PARTS or x.endswith(".egg-info") for x in path.parts)
            and not path.name.startswith("._") and path.suffix not in {".pyc", ".pyo"}
            and path.name not in {".DS_Store", ".env", ".env.local"})


def collect(root: Path, names: list[str]) -> dict[str, tuple[bytes, bool]]:
    files = {}
    links = {}
    for name in names:
        base = root / name
        if not base.exists():
            raise FileNotFoundError(f"Required release input is missing: {name}")
        for path in sorted(base.rglob("*")) if base.is_dir() else [base]:
            if not (path.is_file() or path.is_symlink()) or not allowed(path.relative_to(root)):
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError(f"Symlink leaves release tree: {relative}")
                target = resolved.relative_to(root).as_posix()
                links[relative] = {"target": target,
                                   "original_link_sha256": digest(os.readlink(path).encode()),
                                   "export": "materialized regular file for portable ZIP extraction"}
                files[relative] = (resolved.read_bytes(), False)
            else:
                files[relative] = (path.read_bytes(), False)
    for relative, record in links.items():
        if record["target"] not in files:
            raise ValueError(f"Symlink target is not included: {relative}")
    if links:
        files["RELEASE-LINK-EXPORTS.json"] = ((json.dumps(links, indent=2) + "\n").encode(), False)
    for relative, (data, link) in files.items():
        if SECRET.search(data):
            raise ValueError(f"Potential credential in release input: {relative}")
    return files


def anonymous_readme() -> bytes:
    return b"""# Anonymous reproducibility supplement

When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds.

## Quick start (Python 3.10, Linux)

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproduction/environments/replay-requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python reproduction/run.py verify --output outputs/verify
.venv/bin/python reproduction/run.py analyze --output outputs/analysis
.venv/bin/python reproduction/run.py figures --output outputs/figures
```

See `docs/reproduction.md` for separate simulator environments, smoke checks,
full-cohort commands, and limitations. `docs/figure-map.md` maps the ten figures
to inputs, code, and supplied assets. `docs/comparison-audit.md` documents the
implemented PIVOT-KG versus Uniform comparison. `evidence/paper/manifest.json`
checks frozen inputs; `SHA256SUMS` checks every packaged file. Protocols and
source snapshots are included under `evidence/paper/`. Numeric results are not
changed by export. `export-adjustments.json` records minimal path adjustments.
For a short review route, start with `docs/reviewer-guide.md`.

Expected saved-data results include 51/90 disjoint optimal sets, Leduc's
primary contrast +0.0298887, and MetaDrive's primary contrast
+1.88922 [-1.29334, 5.49203] (unresolved). Simulator smoke validation is separate
from saved-data verification; full original studies were not rerun for release.
See `release/iclr2027-repro-v1/validation.json` for actual validation outcomes.

Only the current paper's reproduction materials are included. Historical
development implementations are not substitutes for frozen cohort protocols.
The manuscript snapshot is read-only. See `docs/rights-and-dependencies.md`
for dependency notices and the absence of a new project-wide license grant.
"""


def package_inputs(root: Path, anonymous: bool) -> dict[str, tuple[bytes, bool]]:
    common = ["pyproject.toml", "src", "evidence/paper", "reproduction",
              f"release/{VERSION}", *DOCS]
    if anonymous:
        names = common + ["experiments/__init__.py", "experiments/v9", "configs/v9",
                          "configs/controlled", "configs/theory", "tests/release"]
        names += ["tests/unit/" + name for name in CORE_TESTS]
    else:
        names = common + ["README.md", "Makefile", "experiments", "configs", "scripts", "tests",
                          "archive/server-code-20260926", "archive/local-code-20260926", "docs/code-provenance.md"]
    files = collect(root, names)
    if anonymous:
        files = {p: value for p, value in files.items()
                 if not p.startswith("src/pivot/cloud/")
                 and p != "tests/release/test_submission_builder.py"}
        # Keep source manifests used by frozen runner integrity checks. Identity
        # removal is limited to export-only provenance, never numeric evidence.
        files["README.md"] = (anonymous_readme(), False)
        for path, (data, link) in list(files.items()):
            if Path(path).suffix in TEXT_SUFFIXES:
                content = data.decode("utf8")
                if IDENTITY.search(content):
                    raise ValueError(f"Author identity must be removed before anonymous export: {path}")
    return files


def write_archive(destination: Path, files: dict[str, tuple[bytes, bool]], anonymous: bool) -> dict:
    manifest = {"schema": 1, "version": VERSION, "anonymous": anonymous,
                "files": [{"path": p, "sha256": digest(d), "bytes": len(d),
                           "type": "symlink" if link else "file"}
                          for p, (d, link) in sorted(files.items())]}
    files = dict(files)
    files["PACKAGE-MANIFEST.json"] = ((json.dumps(manifest, indent=2) + "\n").encode(), False)
    sums = "".join(f"{digest(data)}  {name}\n" for name, (data, _) in sorted(files.items()))
    files["SHA256SUMS"] = (sums.encode(), False)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, (data, link) in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = (stat.S_IFLNK | 0o777) if link else (stat.S_IFREG | (0o755 if name.endswith(".sh") else 0o644))
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data, compresslevel=9)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {destination.name}")
        for name, (data, _) in files.items():
            if z.read(name) != data:
                raise ValueError(f"ZIP content mismatch: {name}")
    return {"file": destination.name, "sha256": digest(destination.read_bytes()),
            "bytes": destination.stat().st_size, "files": len(files), "anonymous": anonymous}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        parser.error("--output must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for anonymous, label in [(False, "full"), (True, "anonymous")]:
        inputs = package_inputs(root, anonymous)
        destination = output / f"pivot-{VERSION}-{label}.zip"
        receipts.append(write_archive(destination, inputs, anonymous))
    (output / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in receipts))
    (output / "packages.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
