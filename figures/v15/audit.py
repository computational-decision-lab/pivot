from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.v15.figure_pipeline import _load_visual_review


def audit(root: Path, *, all_figures: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / "artifacts/v15/figure_status.json"
    if not path.is_file():
        return {"status": "BLOCKED", "reason": "figure status manifest missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records", []) if isinstance(payload, dict) else []
    records = raw_records if isinstance(raw_records, list) else []
    reviewed, review_error = _load_visual_review(root)
    record_ids = [str(item.get("figure_id", "")) for item in records if isinstance(item, dict)]
    records_well_formed = bool(records) and len(record_ids) == len(records) and all(record_ids)
    visual_records_valid = all(
        item.get("state") == "FINAL"
        and item.get("visual", {}).get("state") == "PAPER_CONTEXT_PASS"
        and str(item.get("visual", {}).get("review_manifest", ""))
        == "figures/v15/visual_review_manifest.json"
        and reviewed.get(str(item.get("figure_id", "")))
        == item.get("audit", {}).get("source_hashes", {})
        for item in records
        if isinstance(item, dict)
    )
    valid = (
        bool(payload.get("all_final"))
        and records_well_formed
        and visual_records_valid
        and review_error is None
        and set(record_ids) == set(reviewed)
    )
    return {
        "status": "PASS" if valid else "BLOCKED",
        "valid": valid,
        "figure_count": len(records),
        "all_final": bool(payload.get("all_final")),
        "visual_context_pass": valid,
        "records_well_formed": records_well_formed,
        "review_coverage_match": set(record_ids) == set(reviewed),
        "review_manifest_valid": review_error is None,
        "review_manifest_error": review_error,
        "selection": "all_manifest_figures" if all_figures else "manifest_default",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit publication figure passports")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--all", action="store_true", help="audit every manifest figure")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, all_figures=args.all), sort_keys=True))
