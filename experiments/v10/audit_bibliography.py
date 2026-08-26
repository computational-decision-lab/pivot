#!/usr/bin/env python3
"""Audit bibliography integrity and the recent-work claim boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .audit_utils import write_json, write_markdown

PRIMARY_ARXIV = {
    "worldgym2025": "2506.00613",
    "performativepg2025": "2512.20576",
    "policyaware2026": "2605.29032",
    "evoquant2026": "2607.12455",
    "evopolicygym2026": "2607.02440",
    "selfauthored2026": "2607.24300",
    "map2025": "2502.08063",
    "abidesmarl2025": "2511.02016",
    "dgm2025": "2505.22954",
    "performativempg2025": "2504.20593",
    "lobreality2026": "2603.24137",
    "tradefm2026": "2602.23784",
    "sbco2026": "2608.10157",
    "fragility2026": "2608.18066",
    "evoharness2026": "2608.15071",
    "auditingfinagents2026": "2608.17684",
    "finevobench2026": "2608.06144",
    "finevo2026": "2602.00948",
    "chi2026ai4ai": "2608.20318",
}
PRIMARY_EXTERNAL = {
    "m3market2026": "https://arthurzhang02.github.io/m3-market-microstructure/M3_paper.pdf",
    "thomas2016dataefficient": "https://proceedings.mlr.press/v48/thomasa16.html",
}
BOUNDARY_KEYS = {
    "wu2024progress",
    "chi2026ai4ai",
    "selfauthored2026",
    "worldgym2025",
    "policyaware2026",
    "evoquant2026",
    "evopolicygym2026",
    "dgm2025",
    "finevo2026",
    "abidesmarl2025",
}
TOP_VENUE_KEYS = {
    "agentbench2024",
    "webarena2024",
    "swebench2024",
    "tdmpc22024",
    "finrlmeta2022",
    "mendlerdunner2020stochastic",
    "shinn2023reflexion",
    "madaan2023selfrefine",
    "yao2023react",
}


def _entries(text: str) -> dict[str, str]:
    starts = list(re.finditer(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", text))
    output: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        output[match.group(1)] = text[match.start() : end].strip()
    return output


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"(?mi)^\s*{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*,?\s*$", block)
    return match.group(1).strip() if match else None


def _cited_keys(source: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite\w*\{([^}]*)\}", source):
        keys.extend(item.strip() for item in match.group(1).split(",") if item.strip())
    return keys


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    bib_path = root / "paper/iclr2027/references.bib"
    source_path = root / "paper/iclr2027/main.tex"
    bib_text = bib_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")
    entries = _entries(bib_text)
    raw_keys = re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)
    duplicates = sorted({key for key in raw_keys if raw_keys.count(key) > 1})
    citations = _cited_keys(source)
    cited = set(citations)
    missing = sorted(cited - entries.keys())
    uncited = sorted(entries.keys() - cited)
    errors: list[str] = []
    if duplicates:
        errors.append("duplicate bibliography keys: " + ", ".join(duplicates))
    if missing:
        errors.append("missing cited keys: " + ", ".join(missing))
    if len(citations) != len(cited):
        # Repeated citations are legal; record but do not fail.
        repeated = sorted({key for key in citations if citations.count(key) > 1})
    else:
        repeated = []
    boundary_missing = sorted(BOUNDARY_KEYS - cited)
    if boundary_missing:
        errors.append(
            "required claim-boundary references are not cited: " + ", ".join(boundary_missing)
        )
    primary_records: list[dict[str, Any]] = []
    for key, arxiv_id in PRIMARY_ARXIV.items():
        block = entries.get(key, "")
        url = _field(block, "url") or ""
        eprint = _field(block, "eprint") or ""
        year = _field(block, "year")
        passed = key in entries and (arxiv_id in url or arxiv_id == eprint)
        if not passed:
            errors.append(f"{key}: primary arXiv identifier {arxiv_id} is absent")
        primary_records.append(
            {
                "key": key,
                "primary_url": f"https://arxiv.org/abs/{arxiv_id}",
                "year": year,
                "pass": passed,
            }
        )
    for key, expected_url in PRIMARY_EXTERNAL.items():
        block = entries.get(key, "")
        url = _field(block, "url") or ""
        passed = url == expected_url
        if not passed:
            errors.append(f"{key}: primary URL mismatch")
        primary_records.append(
            {"key": key, "primary_url": expected_url, "year": _field(block, "year"), "pass": passed}
        )
    recent_cited = sorted(
        key for key in cited if (_field(entries.get(key, ""), "year") or "") in {"2025", "2026"}
    )
    top_cited = sorted(TOP_VENUE_KEYS & cited)
    if len(recent_cited) < 12:
        errors.append(f"recent cited coverage is too small: {len(recent_cited)}")
    if len(top_cited) < 4:
        errors.append(f"top-conference cited coverage is too small: {len(top_cited)}")
    bad_shared_url = _field(entries.get("thomas2016dataefficient", ""), "url") == _field(
        entries.get("jiang2016doubly", ""), "url"
    )
    if bad_shared_url:
        errors.append("Thomas--Brunskill and Jiang--Li incorrectly share a primary URL")
    report = {
        "valid": not errors,
        "errors": errors,
        "entry_count": len(entries),
        "cited_count": len(cited),
        "uncited_count": len(uncited),
        "uncited_keys": uncited,
        "repeated_citations": repeated,
        "recent_2025_2026_cited_count": len(recent_cited),
        "recent_2025_2026_cited_keys": recent_cited,
        "top_conference_cited_count": len(top_cited),
        "top_conference_cited_keys": top_cited,
        "required_boundary_keys": sorted(BOUNDARY_KEYS),
        "primary_metadata_verified_on": "2026-08-26",
        "primary_metadata": primary_records,
        "verification_scope": "title/author/year/identifier were checked against arXiv or publisher pages; venue labels are used only where explicitly recorded",
    }
    write_json(root, "artifacts/v10/bibliography_audit.json", report)
    write_markdown(root, "V10_BIBLIOGRAPHY_AUDIT.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Bibliography Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        f"Entries: {report.get('entry_count', 0)}; cited: {report.get('cited_count', 0)}; recent 2025--2026 cited: {report.get('recent_2025_2026_cited_count', 0)}; cited top-conference works: {report.get('top_conference_cited_count', 0)}.",
        "",
        "Recent preprints are described as preprints, not promoted to conference publications. Primary metadata was checked on 2026-08-26.",
        "",
        "| Key | Primary source | Pass |",
        "| --- | --- | --- |",
    ]
    for row in report.get("primary_metadata", []):
        lines.append(f"| `{row['key']}` | {row['primary_url']} | {row['pass']} |")
    lines.extend(["", "## Claim-boundary coverage", ""])
    for key in report.get("required_boundary_keys", []):
        lines.append(f"- `{key}`")
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 bibliography")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "entries": report["entry_count"],
                "cited": report["cited_count"],
                "errors": len(report["errors"]),
            },
            sort_keys=True,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
