from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pivot.environments.finance_backtest.acquisition import (
    acquire_public_finance_data,
    load_public_finance_manifest,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(tmp_path: Path, kline_sha: str, depth_sha: str) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        f"""
dataset_id: binance-um-btcusdt-public-v1
provider: Binance Public Data
license: MIT
market: futures_um
symbol: BTCUSDT
interval: 1m
sessions:
  - session_id: btcusdt-2023-01-01
    date: '2023-01-01'
    klines:
      filename: BTCUSDT-1m-2023-01-01.zip
      url: https://data.binance.vision/BTCUSDT-1m-2023-01-01.zip
      sha256: {kline_sha}
    book_depth:
      filename: BTCUSDT-bookDepth-2023-01-01.zip
      url: https://data.binance.vision/BTCUSDT-bookDepth-2023-01-01.zip
      sha256: {depth_sha}
""",
        encoding="utf-8",
    )
    return path


def test_acquisition_is_checksum_pinned_atomic_and_cache_reusable(tmp_path: Path) -> None:
    payloads = {
        "https://data.binance.vision/BTCUSDT-1m-2023-01-01.zip": b"kline archive",
        "https://data.binance.vision/BTCUSDT-bookDepth-2023-01-01.zip": b"depth archive",
    }
    manifest_path = _manifest(
        tmp_path,
        _digest(payloads["https://data.binance.vision/BTCUSDT-1m-2023-01-01.zip"]),
        _digest(payloads["https://data.binance.vision/BTCUSDT-bookDepth-2023-01-01.zip"]),
    )
    manifest = load_public_finance_manifest(manifest_path)
    calls: list[str] = []

    def downloader(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(payloads[url])

    first = acquire_public_finance_data(manifest, tmp_path / "cache", downloader=downloader)
    second = acquire_public_finance_data(manifest, tmp_path / "cache", downloader=downloader)

    assert len(first.archives) == 2
    assert all(record.status == "downloaded" for record in first.archives)
    assert all(record.status == "reused" for record in second.archives)
    assert calls == list(payloads)
    assert first.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert all(record.local_path.exists() for record in first.archives)
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_acquisition_refuses_corrupt_existing_cache(tmp_path: Path) -> None:
    manifest = load_public_finance_manifest(
        _manifest(tmp_path, _digest(b"expected kline"), _digest(b"expected depth"))
    )
    cache = tmp_path / "cache" / manifest.dataset_id
    cache.mkdir(parents=True)
    (cache / "BTCUSDT-1m-2023-01-01.zip").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="cached archive failed checksum"):
        acquire_public_finance_data(manifest, tmp_path / "cache")


def test_manifest_rejects_untrusted_domain_and_path_traversal(tmp_path: Path) -> None:
    path = _manifest(tmp_path, "a" * 64, "b" * 64)
    body = path.read_text(encoding="utf-8")
    path.write_text(
        body.replace(
            "https://data.binance.vision/BTCUSDT-1m-2023-01-01.zip",
            "https://example.invalid/BTCUSDT-1m-2023-01-01.zip",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.binance.vision"):
        load_public_finance_manifest(path)

    path.write_text(body.replace("BTCUSDT-1m-2023-01-01.zip", "../escape.zip"), encoding="utf-8")
    with pytest.raises(ValueError, match="base name"):
        load_public_finance_manifest(path)
