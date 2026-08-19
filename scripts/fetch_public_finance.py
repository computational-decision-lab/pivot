#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.finance_backtest.acquisition import (
    acquire_public_finance_data,
    load_public_finance_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch checksum-pinned public finance archives")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/public"))
    args = parser.parse_args()

    manifest = load_public_finance_manifest(args.manifest)
    result = acquire_public_finance_data(manifest, args.output_root)
    receipt = {
        "dataset_id": result.dataset_id,
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": result.manifest_sha256,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "live_orders": False,
        "archives": [
            {
                "session_id": record.session_id,
                "kind": record.kind,
                "source_url": record.source_url,
                "expected_sha256": record.expected_sha256,
                "actual_sha256": record.actual_sha256,
                "local_path": str(record.local_path),
                "status": record.status,
            }
            for record in result.archives
        ],
    }
    receipt_path = Path(args.output_root).resolve() / result.dataset_id / "acquisition_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_id": result.dataset_id,
                "archives": len(result.archives),
                "downloaded": sum(record.status == "downloaded" for record in result.archives),
                "reused": sum(record.status == "reused" for record in result.archives),
                "receipt": str(receipt_path),
                "live_orders": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
