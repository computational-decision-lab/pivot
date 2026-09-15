from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pivot.analysis.public_finance import run_public_finance_calibration


def _archive(path: Path, member: str, body: str) -> str:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, body)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_calibration_is_session_paired_and_does_not_promote_gate(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "public-test-v1"
    cache.mkdir(parents=True)
    kline = cache / "BTCUSDT-1m-2023-01-01.zip"
    kline_sha = _archive(
        kline,
        "klines.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1672531200000,100,101,99,100,10,1672531259999,1000,20,5,500,0\n"
        "1672531260000,100,102,99,101,20,1672531319999,2000,30,12,1200,0\n"
        "1672531320000,101,102,99,100,15,1672531379999,1500,25,10,1000,0\n"
        "1672531380000,100,103,99,102,25,1672531439999,2500,40,20,2000,0\n",
    )
    depth = cache / "BTCUSDT-bookDepth-2023-01-01.zip"
    depth_sha = _archive(
        depth,
        "depth.csv",
        "timestamp,percentage,depth,notional\n"
        "2023-01-01 00:00:00,-1,100,10000\n"
        "2023-01-01 00:00:00,1,100,10000\n"
        "2023-01-01 00:01:00,-1,120,12000\n"
        "2023-01-01 00:01:00,1,120,12000\n"
        "2023-01-01 00:02:00,-1,110,11000\n"
        "2023-01-01 00:02:00,1,110,11000\n",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
dataset_id: public-test-v1
provider: Binance Public Data
license: MIT
market: futures_um
symbol: BTCUSDT
interval: 1m
sessions:
  - session_id: btcusdt-2023-01-01
    date: '2023-01-01'
    klines:
      filename: {kline.name}
      url: https://data.binance.vision/{kline.name}
      sha256: {kline_sha}
    book_depth:
      filename: {depth.name}
      url: https://data.binance.vision/{depth.name}
      sha256: {depth_sha}
""",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
dataset_manifest: {manifest}
data_root: {tmp_path / 'cache'}
execution:
  spread_bps: 0.0
  fee_bps: 0.0
  slippage_bps: 0.0
  partial_fill_ratio: 1.0
  queue_depth: 1.0
incumbent:
  intensity: 0.0
  bias: 100.0
  position_size: 0.2
candidate:
  intensity: 0.0
  bias: 100.0
  position_size: 0.6
edit_type: position_size
participation_rates: [0.0, 0.01]
impact_multipliers: [1.0]
target_participation: 0.01
""",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    summary = run_public_finance_calibration(config, output)
    rows = json.loads((output / "public_finance_rows.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["delta_f2"] == rows[0]["delta_f1"]
    assert rows[0]["transition_edit_type"] == "position_size"
    assert rows[1]["applied_impact_coefficient"] > 0
    assert summary["n_sessions"] == 1
    assert summary["zero_participation_max_abs_f2_minus_f1"] == 0.0
    assert summary["causal_impact_identified"] is False
    assert summary["gate_e_promoted"] is False
    assert provenance["live_orders"] is False
    assert provenance["ground_truth_for_endogenous_response"] is False
