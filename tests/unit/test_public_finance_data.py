from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from pivot.environments.finance_backtest.public_data import (
    align_klines_to_depth_coverage,
    calibrate_depth_proxy,
    finance_config_from_public_data,
    load_binance_book_depth_archive,
    load_binance_kline_archive,
)


def _archive(path: Path, member: str, body: str) -> str:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, body)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_verified_binance_kline_archive_and_normalizes_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT-1m-2023-01-01.zip"
    digest = _archive(
        path,
        "BTCUSDT-1m-2023-01-01.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1672531200000,100,102,99,101,10,1672531259999,1005,20,6,603,0\n"
        "1672531260000,101,103,100,102,12,1672531319999,1218,24,7,711,0\n",
    )

    dataset = load_binance_kline_archive(
        path,
        expected_sha256=digest,
        dataset_id="binance-um-btcusdt-2023-01-01",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/example.zip",
    )

    assert dataset.archive_sha256 == digest
    assert dataset.timestamp_unit == "milliseconds"
    assert dataset.bars[0].open_time_us == 1_672_531_200_000_000
    assert dataset.prices == (101.0, 102.0)
    assert dataset.quote_volumes == (1005.0, 1218.0)
    assert dataset.source_kind == "public_observational_market_data"
    assert dataset.ground_truth_for_endogenous_response is False


def test_kline_loader_rejects_checksum_and_non_monotone_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    digest = _archive(
        path,
        "bad.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1672531260000,101,103,100,102,12,1672531319999,1218,24,7,711,0\n"
        "1672531200000,100,102,99,101,10,1672531259999,1005,20,6,603,0\n",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_binance_kline_archive(
            path,
            expected_sha256="0" * 64,
            dataset_id="bad",
            symbol="BTCUSDT",
            source_url="https://example.invalid/bad.zip",
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_binance_kline_archive(
            path,
            expected_sha256=digest,
            dataset_id="bad",
            symbol="BTCUSDT",
            source_url="https://example.invalid/bad.zip",
        )


def test_depth_proxy_and_public_finance_config_preserve_provenance(tmp_path: Path) -> None:
    kline_path = tmp_path / "klines.zip"
    kline_digest = _archive(
        kline_path,
        "klines.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1672531200000,100,102,99,101,10,1672531259999,1000,20,6,600,0\n"
        "1672531260000,101,103,100,102,20,1672531319999,2000,24,7,700,0\n"
        "1672531320000,102,104,101,103,30,1672531379999,3000,30,18,1800,0\n",
    )
    depth_path = tmp_path / "depth.zip"
    depth_digest = _archive(
        depth_path,
        "depth.csv",
        "timestamp,percentage,depth,notional\n"
        "2023-01-01 00:00:00,-1,100,10000\n"
        "2023-01-01 00:00:00,1,110,12000\n"
        "2023-01-01 00:01:00,-1,105,11000\n"
        "2023-01-01 00:01:00,1,115,13000\n",
    )
    klines = load_binance_kline_archive(
        kline_path,
        expected_sha256=kline_digest,
        dataset_id="public-session",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/klines.zip",
    )
    depth = load_binance_book_depth_archive(
        depth_path,
        expected_sha256=depth_digest,
        dataset_id="public-session",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/depth.zip",
    )

    calibration = calibrate_depth_proxy(klines, depth)
    config = finance_config_from_public_data(klines)

    assert calibration.method == "linearized_one_percent_depth_proxy"
    assert calibration.causal_impact_identified is False
    assert calibration.median_quote_volume == pytest.approx(2000.0)
    assert calibration.median_one_percent_depth_notional == pytest.approx(11_500.0)
    assert calibration.impact_coefficient == pytest.approx(0.01 * 2000.0 / 11_500.0)
    assert 0.0 <= calibration.descriptive_recovery_proxy <= 1.0
    assert config.prices == (101.0, 102.0, 103.0)
    assert config.fixture_version == "public-session"
    assert config.data_source == "binance-public-data"
    assert config.data_sha256 == kline_digest
    assert config.ground_truth_for_endogenous_response is False


def test_book_depth_rejects_duplicate_timestamp_level(tmp_path: Path) -> None:
    path = tmp_path / "depth.zip"
    digest = _archive(
        path,
        "depth.csv",
        "timestamp,percentage,depth,notional\n"
        "2023-01-01 00:00:00,-1,100,10000\n"
        "2023-01-01 00:00:00,-1,101,11000\n",
    )

    with pytest.raises(ValueError, match="duplicate timestamp/percentage"):
        load_binance_book_depth_archive(
            path,
            expected_sha256=digest,
            dataset_id="duplicate-depth",
            symbol="BTCUSDT",
            source_url="https://example.invalid/depth.zip",
        )


def test_kline_alignment_uses_only_observed_depth_coverage(tmp_path: Path) -> None:
    kline_path = tmp_path / "klines.zip"
    kline_digest = _archive(
        kline_path,
        "klines.csv",
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1672531200000,100,101,99,100,10,1672531259999,1000,20,5,500,0\n"
        "1672531260000,100,101,99,100,10,1672531319999,1000,20,5,500,0\n"
        "1672531320000,100,101,99,100,10,1672531379999,1000,20,5,500,0\n"
        "1672531380000,100,101,99,100,10,1672531439999,1000,20,5,500,0\n",
    )
    depth_path = tmp_path / "depth.zip"
    depth_digest = _archive(
        depth_path,
        "depth.csv",
        "timestamp,percentage,depth,notional\n"
        "2023-01-01 00:01:30,-1,100,10000\n"
        "2023-01-01 00:01:30,1,100,10000\n"
        "2023-01-01 00:03:00,-1,100,10000\n"
        "2023-01-01 00:03:00,1,100,10000\n",
    )
    klines = load_binance_kline_archive(
        kline_path,
        expected_sha256=kline_digest,
        dataset_id="alignment",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/klines.zip",
    )
    depth = load_binance_book_depth_archive(
        depth_path,
        expected_sha256=depth_digest,
        dataset_id="alignment",
        symbol="BTCUSDT",
        source_url="https://data.binance.vision/depth.zip",
    )

    aligned = align_klines_to_depth_coverage(klines, depth)

    assert [bar.open_time_us for bar in aligned.bars] == [
        1_672_531_320_000_000,
        1_672_531_380_000_000,
    ]
