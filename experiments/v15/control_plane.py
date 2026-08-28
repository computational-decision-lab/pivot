"""Optional control-plane probes and adapter contracts.

The probe is intentionally non-invasive: it never installs packages, calls a
model, or evaluates a confirmatory task.  A missing external dependency is an
auditable availability boundary rather than an implicit fallback.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import MiniSWEAdapter, PiAdapter
from .adapters.inspect_ai import InspectControlPlane


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    import_name: str | None
    command: str | None
    available: bool
    version: str | None
    source: str


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    if not output:
        return None
    # Some CLIs print migration/configuration notices before the actual
    # version.  Keep the probe machine-readable and retain the first line that
    # looks like a version declaration.
    for line in output:
        if line.casefold().startswith(("this is ", "version")) or " version " in line.casefold():
            return line.strip()
    return output[-1].strip()


def probe_adapters(root: Path | None = None) -> list[AdapterStatus]:
    """Probe locally installed package/CLI metadata.

    ``Pi`` is built from the pinned source checkout and is not necessarily
    installed on ``PATH``.  When a project root is supplied, the probe checks
    that locked project-local CLI as well as the normal command path.
    """

    records: list[AdapterStatus] = []
    for name, import_name, command in (
        ("Inspect AI", "inspect_ai", "inspect"),
        ("mini-SWE-agent", "minisweagent", "mini-swe-agent"),
        ("Pi", None, "pi"),
    ):
        imported = import_name is not None and importlib.util.find_spec(import_name) is not None
        command_version = _command_version(command) if command else None
        local_status: dict[str, Any] = {}
        if name == "Pi" and root is not None:
            local_status = PiAdapter(root).status()
        available = imported or command_version is not None or bool(local_status.get("cli_exists"))
        version = command_version
        if imported and version is None:
            # Read installed distribution metadata without importing the
            # external runtime.  mini-SWE-agent prints a migration notice at
            # import time, which would corrupt JSON emitted by our CLI probe.
            distribution_names = {
                "inspect_ai": "inspect-ai",
                "minisweagent": "mini-swe-agent",
            }
            distribution = distribution_names.get(import_name or "")
            if distribution:
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:  # pragma: no cover - external package
                    version = None
        source = "project_local_probe" if local_status.get("cli_exists") and command_version is None else "local_probe"
        if name == "Pi" and local_status.get("source_commit"):
            version = str(local_status["source_commit"])
        records.append(
            AdapterStatus(
                name=name,
                import_name=import_name,
                command=command,
                available=available,
                version=None if version is None else str(version),
                source=source,
            )
        )
    return records


def control_plane_manifest(root: Path) -> dict[str, Any]:
    """Return a machine-readable availability report without running agents."""

    statuses = [asdict(item) for item in probe_adapters(root)]
    contracts = [InspectControlPlane().status(), MiniSWEAdapter().status(), PiAdapter(root).status()]
    return {
        "control_plane": "Inspect AI when available; adapters remain agent-agnostic",
        "status": statuses,
        "adapter_contracts": contracts,
        "confirmatory_execution": False,
        "reason": "No package installation or model invocation is performed by the probe.",
        "project_root": str(root),
    }


def write_manifest(root: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(control_plane_manifest(root), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
