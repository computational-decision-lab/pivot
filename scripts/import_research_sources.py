"""Archive the 2026-09-26 server ZIP and the current local research tree.

This is an archival import only. It never imports or runs archived code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import stat
import zipfile
from collections import Counter


GROUPS = {
    "01_作者仓库_初始上传版本": "01_initial_author_repository",
    "02_作者仓库_后续服务器版本": "02_later_server_repository",
    "03_MeltingPot_MPE2_早期实验": "03_early_meltingpot_mpe2",
    "04_PIVOTv2_MeltingPot_OpenSpiel_Highway": "04_pivot_v2_benchmarks",
    "05_OpenSpiel_早期冻结代码": "05_early_openspiel_frozen",
    "06_Leduc_v2b_修正版": "06_leduc_v2b_revised",
    "07_Leduc_v4": "07_leduc_v4",
    "08_MetaDrive_第一轮": "08_metadrive_first_round",
    "09_MetaDrive_精度复核": "09_metadrive_precision_check",
    "10_补充分析与作图代码": "10_supplementary_analysis_figures",
}
ZIP_ROOT = "服务器运行代码汇总_20260926"
LOCAL_DIRS = ("src", "experiments", "scripts", "tests", "configs")
LOCAL_ROOT_FILES = ("pyproject.toml", "uv.lock")
SERVER_EXCLUSIONS = {
    "sync_results.sh": "remote result synchronization with private endpoint",
    "deepseek_design_review.py": "external LLM review utility",
    "check_deepseek.py": "external LLM API connectivity and credential utility",
    "cloud_budget_watchdog.py": "cloud infrastructure budget watchdog",
    "bootstrap_cpu.sh": "cloud environment provisioning utility",
    "collect_melting_response_sprint.py": "remote SSH collection utility with private endpoint",
    "sync_v9_paper_assets.py": "paper asset synchronization utility",
}
LOCAL_EXCLUSIONS = {
    "scripts/tencent_resources.py": "cloud resource provisioning utility",
    "scripts/sync_v9_paper_assets.py": "paper asset synchronization utility",
    "tests/unit/test_tencent_lifecycle.py": "cloud resource provisioning test",
}
SECRET_PATTERNS = {
    "private-key-block": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential-assignment": re.compile(
        rb"(?im)^\s*(?:[A-Z_]*(?:API_KEY|ACCESS_KEY|SECRET|PASSWORD|TOKEN)[A-Z_]*)\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]"
    ),
    "private-ip": re.compile(rb"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"),
    "ssh-host": re.compile(rb"(?im)(?:ssh|scp|rsync|paramiko\.SSHClient).*?(?:root@|[a-z0-9.-]+\.internal)"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_path(raw: str) -> str:
    # macOS wrote UTF-8 filename bytes without setting the ZIP Unicode flag.
    return raw.encode("cp437").decode("utf-8")


def check_relative(parts: tuple[str, ...]) -> None:
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe path components: {parts!r}")
    if any("\\" in part or ":" in part for part in parts):
        raise ValueError(f"unsafe path components: {parts!r}")


def write_verified(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"existing destination differs: {path}")
    path.write_bytes(data)
    if sha256(path.read_bytes()) != sha256(data):
        raise ValueError(f"copy failed verification: {path}")


def check_source(data: bytes, path: str) -> None:
    alerts = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    if alerts:
        raise ValueError(f"sensitive source needs review: {path} ({', '.join(alerts)})")
    if path.endswith(".py"):
        # Compilation by AST parsing does not execute module-level code or create pyc files.
        ast.parse(data.decode("utf-8-sig"), filename=path)


def import_server(source: pathlib.Path, dest: pathlib.Path) -> dict:
    records = []
    seen_dest = set()
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed for {bad}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            original = zip_path(info.filename)
            data = archive.read(info)
            record = {"original_path": original, "sha256": sha256(data), "bytes": len(data)}
            parts = pathlib.PurePosixPath(original).parts
            check_relative(parts)
            if parts[0] == "__MACOSX" or any(p.startswith("._") for p in parts):
                record.update(status="excluded", reason="macOS resource fork metadata")
            elif len(parts) < 3 or parts[0] != ZIP_ROOT or parts[1] not in GROUPS:
                raise ValueError(f"unexpected ZIP layout: {original}")
            elif parts[-1] in SERVER_EXCLUSIONS:
                record.update(status="excluded", reason=SERVER_EXCLUSIONS[parts[-1]])
            else:
                relative = pathlib.PurePosixPath(GROUPS[parts[1]], *parts[2:])
                if str(relative) in seen_dest:
                    raise ValueError(f"destination collision: {relative}")
                seen_dest.add(str(relative))
                if stat.S_ISLNK(info.external_attr >> 16):
                    target = data.decode("utf-8")
                    link = dest / relative
                    resolved = (link.parent / target).resolve()
                    if not resolved.is_relative_to((dest / GROUPS[parts[1]]).resolve()):
                        raise ValueError(f"symlink escapes group: {original}")
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink() and link.readlink().as_posix() != target:
                        raise ValueError(f"existing symlink differs: {link}")
                    if not link.is_symlink():
                        link.symlink_to(target)
                    record.update(status="archived", destination=str(relative), type="symlink", link_target=target)
                else:
                    check_source(data, original)
                    write_verified(dest / relative, data)
                    record.update(status="archived", destination=str(relative))
            records.append(record)
    return {
        "source_zip": source.name,
        "source_zip_sha256": sha256(source.read_bytes()),
        "groups": GROUPS,
        "counts": dict(Counter(r["status"] for r in records)),
        "files": records,
    }


def import_local(source: pathlib.Path, dest: pathlib.Path) -> dict:
    records = []
    for directory in (*LOCAL_DIRS, *LOCAL_ROOT_FILES):
        candidate = source / directory
        paths = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
        for path in paths:
            if not path.is_file() or any(p in ("__pycache__", ".pytest_cache", ".ruff_cache") for p in path.parts):
                continue
            relative = path.relative_to(source).as_posix()
            data = path.read_bytes()
            record = {"original_path": relative, "sha256": sha256(data), "bytes": len(data)}
            if any(part.endswith(".egg-info") for part in path.relative_to(source).parts):
                record.update(status="excluded", reason="generated Python package metadata")
            elif relative.startswith("src/pivot/cloud/"):
                record.update(status="excluded", reason="cloud resource provisioning code")
            elif relative in LOCAL_EXCLUSIONS:
                record.update(status="excluded", reason=LOCAL_EXCLUSIONS[relative])
            elif path.suffix not in (".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".lock", ".txt", ".md", ".ts"):
                record.update(status="excluded", reason="non-source or generated file")
            else:
                check_source(data, relative)
                write_verified(dest / relative, data)
                record.update(status="archived", destination=relative)
            records.append(record)
    return {"source_root": str(source), "counts": dict(Counter(r["status"] for r in records)), "files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=pathlib.Path, required=True)
    parser.add_argument("--local-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    server = import_server(args.source_zip, root / "archive/server-code-20260926")
    local = import_local(args.local_root, root / "archive/local-code-20260926")
    later = {
        r["destination"]: r["sha256"]
        for r in server["files"]
        if r["status"] == "archived" and r["destination"].startswith("02_later_server_repository/")
    }
    for record in local["files"]:
        if record["status"] != "archived":
            continue
        counterpart = "02_later_server_repository/" + record["destination"]
        record["later_server_path"] = counterpart if counterpart in later else None
        record["later_server_comparison"] = (
            "absent" if counterpart not in later else
            "identical" if record["sha256"] == later[counterpart] else "different"
        )
    local["later_server_counts"] = dict(Counter(
        r["later_server_comparison"] for r in local["files"] if r["status"] == "archived"
    ))
    for name, value in (("server-code-20260926", server), ("local-code-20260926", local)):
        path = root / "archive" / name / "source-manifest.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("server:", server["counts"], "local:", local["counts"])


if __name__ == "__main__":
    main()
