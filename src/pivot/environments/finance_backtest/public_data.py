from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.typing import NDArray

from .config import FinanceConfig

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
BOOK_DEPTH_COLUMNS = ("timestamp", "percentage", "depth", "notional")


@dataclass(frozen=True)
class KlineBar:
    open_time_us: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_us: int
    quote_volume: float
    trade_count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float


@dataclass(frozen=True)
class PublicKlineDataset:
    dataset_id: str
    symbol: str
    source_url: str
    archive_sha256: str
    archive_member: str
    timestamp_unit: str
    bars: tuple[KlineBar, ...]
    source_kind: str = "public_observational_market_data"
    ground_truth_for_endogenous_response: bool = False

    @property
    def prices(self) -> tuple[float, ...]:
        return tuple(bar.close for bar in self.bars)

    @property
    def quote_volumes(self) -> tuple[float, ...]:
        return tuple(bar.quote_volume for bar in self.bars)


@dataclass(frozen=True)
class BookDepthRow:
    timestamp_us: int
    percentage: float
    depth: float
    notional: float


@dataclass(frozen=True)
class PublicBookDepthDataset:
    dataset_id: str
    symbol: str
    source_url: str
    archive_sha256: str
    archive_member: str
    rows: tuple[BookDepthRow, ...]
    source_kind: str = "public_observational_market_data"
    ground_truth_for_endogenous_response: bool = False


@dataclass(frozen=True)
class DepthProxyCalibration:
    dataset_id: str
    symbol: str
    method: str
    median_quote_volume: float
    median_one_percent_depth_notional: float
    impact_coefficient: float
    descriptive_recovery_proxy: float
    n_kline_bars: int
    n_depth_snapshots: int
    causal_impact_identified: bool = False


def load_binance_kline_archive(
    path: Path,
    *,
    expected_sha256: str,
    dataset_id: str,
    symbol: str,
    source_url: str,
) -> PublicKlineDataset:
    """Load one checksum-pinned Binance kline CSV archive without extraction."""

    archive_sha256 = verify_sha256(path, expected_sha256)
    member, rows = _read_single_csv(path)
    if not rows:
        raise ValueError("kline archive contains no rows")
    first = tuple(value.strip().lower() for value in rows[0])
    data_rows = rows[1:] if first == KLINE_COLUMNS else rows
    if not data_rows:
        raise ValueError("kline archive contains a header but no data")

    bars: list[KlineBar] = []
    timestamp_unit: str | None = None
    previous_open_time = -1
    for row_number, row in enumerate(data_rows, start=2 if first == KLINE_COLUMNS else 1):
        if len(row) != len(KLINE_COLUMNS):
            raise ValueError(
                f"kline row {row_number} has {len(row)} columns; expected {len(KLINE_COLUMNS)}"
            )
        try:
            open_time_us, row_unit = _epoch_to_microseconds(row[0])
            close_time_us, close_unit = _epoch_to_microseconds(row[6])
            open_price, high, low, close = (float(row[index]) for index in range(1, 5))
            volume = float(row[5])
            quote_volume = float(row[7])
            trade_count = int(row[8])
            taker_buy_volume = float(row[9])
            taker_buy_quote_volume = float(row[10])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid kline row {row_number}: {error}") from error
        if row_unit != close_unit:
            raise ValueError(f"mixed timestamp units within kline row {row_number}")
        if timestamp_unit is None:
            timestamp_unit = row_unit
        elif row_unit != timestamp_unit:
            raise ValueError("mixed timestamp units across kline rows")
        if open_time_us <= previous_open_time:
            raise ValueError("kline open times must be strictly increasing")
        if close_time_us <= open_time_us:
            raise ValueError(f"kline close time must follow open time at row {row_number}")
        if not all(math.isfinite(value) for value in (open_price, high, low, close)):
            raise ValueError(f"non-finite OHLC value at kline row {row_number}")
        if low <= 0 or low > min(open_price, close) or high < max(open_price, close):
            raise ValueError(f"invalid OHLC ordering at kline row {row_number}")
        if min(volume, quote_volume, taker_buy_volume, taker_buy_quote_volume) < 0:
            raise ValueError(f"negative volume at kline row {row_number}")
        if taker_buy_volume > volume + 1e-9 or taker_buy_quote_volume > quote_volume + 1e-6:
            raise ValueError(f"taker-buy volume exceeds total volume at kline row {row_number}")
        if trade_count < 0:
            raise ValueError(f"negative trade count at kline row {row_number}")
        bars.append(
            KlineBar(
                open_time_us=open_time_us,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                close_time_us=close_time_us,
                quote_volume=quote_volume,
                trade_count=trade_count,
                taker_buy_volume=taker_buy_volume,
                taker_buy_quote_volume=taker_buy_quote_volume,
            )
        )
        previous_open_time = open_time_us
    if len(bars) < 2:
        raise ValueError("kline dataset must contain at least two bars")
    return PublicKlineDataset(
        dataset_id=_required(dataset_id, "dataset_id"),
        symbol=_required(symbol, "symbol"),
        source_url=_required(source_url, "source_url"),
        archive_sha256=archive_sha256,
        archive_member=member,
        timestamp_unit=timestamp_unit or "unknown",
        bars=tuple(bars),
    )


def load_binance_book_depth_archive(
    path: Path,
    *,
    expected_sha256: str,
    dataset_id: str,
    symbol: str,
    source_url: str,
) -> PublicBookDepthDataset:
    """Load one checksum-pinned Binance percentage-depth CSV archive."""

    archive_sha256 = verify_sha256(path, expected_sha256)
    member, rows = _read_single_csv(path)
    if not rows:
        raise ValueError("book-depth archive contains no rows")
    first = tuple(value.strip().lower() for value in rows[0])
    if first != BOOK_DEPTH_COLUMNS:
        raise ValueError(f"book-depth header must be {BOOK_DEPTH_COLUMNS!r}")
    data_rows = rows[1:]
    parsed: list[BookDepthRow] = []
    seen: set[tuple[int, float]] = set()
    previous_timestamp = -1
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != len(BOOK_DEPTH_COLUMNS):
            raise ValueError(
                f"book-depth row {row_number} has {len(row)} columns; "
                f"expected {len(BOOK_DEPTH_COLUMNS)}"
            )
        try:
            timestamp_us = _datetime_to_microseconds(row[0])
            percentage = float(row[1])
            depth = float(row[2])
            notional = float(row[3])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid book-depth row {row_number}: {error}") from error
        if timestamp_us < previous_timestamp:
            raise ValueError("book-depth timestamps must be non-decreasing")
        key = (timestamp_us, percentage)
        if key in seen:
            raise ValueError("book-depth archive contains duplicate timestamp/percentage rows")
        if percentage == 0 or not all(
            math.isfinite(value) for value in (percentage, depth, notional)
        ):
            raise ValueError(f"invalid book-depth values at row {row_number}")
        if depth <= 0 or notional <= 0:
            raise ValueError(f"book-depth liquidity must be positive at row {row_number}")
        parsed.append(BookDepthRow(timestamp_us, percentage, depth, notional))
        seen.add(key)
        previous_timestamp = timestamp_us
    if not parsed:
        raise ValueError("book-depth archive contains a header but no data")
    return PublicBookDepthDataset(
        dataset_id=_required(dataset_id, "dataset_id"),
        symbol=_required(symbol, "symbol"),
        source_url=_required(source_url, "source_url"),
        archive_sha256=archive_sha256,
        archive_member=member,
        rows=tuple(parsed),
    )


def calibrate_depth_proxy(
    klines: PublicKlineDataset, depth: PublicBookDepthDataset
) -> DepthProxyCalibration:
    """Derive a descriptive, non-causal linearized impact proxy.

    A participation fraction is converted to notional with median one-minute
    quote volume, then divided by median cumulative depth within one percent.
    The result is useful for sensitivity ranges, not as interventional truth.
    """

    if klines.dataset_id != depth.dataset_id or klines.symbol != depth.symbol:
        raise ValueError("kline and book-depth datasets must identify the same session")
    quote_volumes = [value for value in klines.quote_volumes if value > 0]
    one_percent_rows = [row for row in depth.rows if math.isclose(abs(row.percentage), 1.0)]
    if not quote_volumes:
        raise ValueError("positive quote volume is required for depth calibration")
    if not one_percent_rows:
        raise ValueError("book-depth data must include both or either one-percent levels")
    median_quote_volume = float(median(quote_volumes))
    median_depth = float(median(row.notional for row in one_percent_rows))
    impact_coefficient = 0.01 * median_quote_volume / median_depth

    by_timestamp: dict[int, list[float]] = {}
    for row in one_percent_rows:
        by_timestamp.setdefault(row.timestamp_us, []).append(row.notional)
    depth_series = np.asarray(
        [median(by_timestamp[timestamp]) for timestamp in sorted(by_timestamp)], dtype=float
    )
    recovery_proxy = _descriptive_recovery(depth_series)
    return DepthProxyCalibration(
        dataset_id=klines.dataset_id,
        symbol=klines.symbol,
        method="linearized_one_percent_depth_proxy",
        median_quote_volume=median_quote_volume,
        median_one_percent_depth_notional=median_depth,
        impact_coefficient=impact_coefficient,
        descriptive_recovery_proxy=recovery_proxy,
        n_kline_bars=len(klines.bars),
        n_depth_snapshots=len(by_timestamp),
    )


def align_klines_to_depth_coverage(
    klines: PublicKlineDataset, depth: PublicBookDepthDataset
) -> PublicKlineDataset:
    """Trim bars to timestamps with an already observed depth snapshot."""

    if klines.dataset_id != depth.dataset_id or klines.symbol != depth.symbol:
        raise ValueError("kline and book-depth datasets must identify the same session")
    first_depth = min(row.timestamp_us for row in depth.rows)
    last_depth = max(row.timestamp_us for row in depth.rows)
    bars = tuple(
        bar for bar in klines.bars if first_depth <= bar.open_time_us <= last_depth
    )
    if len(bars) < 2:
        raise ValueError("fewer than two kline bars overlap observed depth coverage")
    return replace(klines, bars=bars)


def finance_config_from_public_data(
    dataset: PublicKlineDataset,
    *,
    spread_bps: float = 2.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 3.0,
    partial_fill_ratio: float = 0.75,
    queue_depth: float = 0.5,
    strategy_frequency: str = "bar",
    simulation_frequency: str = "event",
) -> FinanceConfig:
    return FinanceConfig(
        prices=dataset.prices,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        partial_fill_ratio=partial_fill_ratio,
        queue_depth=queue_depth,
        strategy_frequency=strategy_frequency,
        simulation_frequency=simulation_frequency,
        fixture_version=dataset.dataset_id,
        data_source="binance-public-data",
        data_sha256=dataset.archive_sha256,
        source_url=dataset.source_url,
        ground_truth_for_endogenous_response=False,
        virtual_fills=True,
    )


def verify_sha256(path: Path, expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FileNotFoundError(f"cannot read public-data archive: {path}") from error
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def _read_single_csv(path: Path) -> tuple[str, list[list[str]]]:
    try:
        with ZipFile(path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            csv_members = [member for member in members if member.filename.lower().endswith(".csv")]
            if len(csv_members) != 1 or len(members) != 1:
                raise ValueError("public-data archive must contain exactly one CSV member")
            member = csv_members[0]
            if member.flag_bits & 0x1:
                raise ValueError("encrypted public-data archives are not supported")
            with archive.open(member, "r") as binary, io.TextIOWrapper(
                binary, encoding="utf-8-sig", newline=""
            ) as text:
                return member.filename, list(csv.reader(text))
    except BadZipFile as error:
        raise ValueError(f"invalid ZIP archive: {path}") from error


def _epoch_to_microseconds(raw: str) -> tuple[int, str]:
    value = int(raw)
    if 1_000_000_000_000 <= value < 100_000_000_000_000:
        return value * 1000, "milliseconds"
    if 100_000_000_000_000 <= value < 100_000_000_000_000_000:
        return value, "microseconds"
    raise ValueError(f"unsupported epoch timestamp magnitude: {value}")


def _datetime_to_microseconds(raw: str) -> int:
    parsed = datetime.fromisoformat(raw.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000_000)


def _descriptive_recovery(depth_series: NDArray[np.float64]) -> float:
    if depth_series.size < 3 or float(np.std(depth_series)) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(depth_series[:-1], depth_series[1:])[0, 1])
    if not math.isfinite(correlation):
        return 0.0
    return float(np.clip(1.0 - correlation, 0.0, 1.0))


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized
