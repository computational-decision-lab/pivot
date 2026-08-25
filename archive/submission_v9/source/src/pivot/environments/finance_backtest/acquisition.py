from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from .public_data import verify_sha256

Downloader = Callable[[str, Path], None]


@dataclass(frozen=True)
class PublicArchiveSpec:
    kind: str
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class PublicSessionSpec:
    session_id: str
    date: str
    klines: PublicArchiveSpec
    book_depth: PublicArchiveSpec


@dataclass(frozen=True)
class PublicFinanceManifest:
    dataset_id: str
    provider: str
    license: str
    market: str
    symbol: str
    interval: str
    sessions: tuple[PublicSessionSpec, ...]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class AcquiredArchive:
    session_id: str
    kind: str
    source_url: str
    expected_sha256: str
    actual_sha256: str
    local_path: Path
    status: str


@dataclass(frozen=True)
class AcquisitionResult:
    dataset_id: str
    manifest_path: Path
    manifest_sha256: str
    archives: tuple[AcquiredArchive, ...]


def load_public_finance_manifest(path: Path) -> PublicFinanceManifest:
    manifest_path = Path(path).resolve()
    raw = manifest_path.read_bytes()
    payload = yaml.safe_load(raw)
    root = _mapping(payload, "manifest")
    sessions_raw = root.get("sessions")
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ValueError("manifest sessions must be a non-empty list")
    sessions: list[PublicSessionSpec] = []
    seen: set[str] = set()
    for index, raw_session in enumerate(sessions_raw):
        session = _mapping(raw_session, f"sessions[{index}]")
        session_id = _path_component(_required_str(session, "session_id"), "session_id")
        if session_id in seen:
            raise ValueError("manifest session IDs must be unique")
        sessions.append(
            PublicSessionSpec(
                session_id=session_id,
                date=_required_str(session, "date"),
                klines=_archive_spec(_mapping(session.get("klines"), "klines"), "klines"),
                book_depth=_archive_spec(
                    _mapping(session.get("book_depth"), "book_depth"), "book_depth"
                ),
            )
        )
        seen.add(session_id)
    return PublicFinanceManifest(
        dataset_id=_path_component(_required_str(root, "dataset_id"), "dataset_id"),
        provider=_required_str(root, "provider"),
        license=_required_str(root, "license"),
        market=_required_str(root, "market"),
        symbol=_required_str(root, "symbol"),
        interval=_required_str(root, "interval"),
        sessions=tuple(sessions),
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def acquire_public_finance_data(
    manifest: PublicFinanceManifest,
    output_root: Path,
    *,
    downloader: Downloader | None = None,
) -> AcquisitionResult:
    """Download checksum-pinned archives without replacing existing cache files."""

    download = downloader or _download_https
    destination = Path(output_root).resolve() / manifest.dataset_id
    destination.mkdir(parents=True, exist_ok=True)
    acquired: list[AcquiredArchive] = []
    for session in manifest.sessions:
        for archive in (session.klines, session.book_depth):
            target = destination / archive.filename
            if target.exists():
                try:
                    digest = verify_sha256(target, archive.sha256)
                except ValueError as error:
                    raise ValueError(f"cached archive failed checksum: {target}") from error
                status = "reused"
            else:
                digest = _download_verified(download, archive, target)
                status = "downloaded"
            acquired.append(
                AcquiredArchive(
                    session_id=session.session_id,
                    kind=archive.kind,
                    source_url=archive.url,
                    expected_sha256=archive.sha256,
                    actual_sha256=digest,
                    local_path=target,
                    status=status,
                )
            )
    return AcquisitionResult(
        dataset_id=manifest.dataset_id,
        manifest_path=manifest.manifest_path,
        manifest_sha256=manifest.manifest_sha256,
        archives=tuple(acquired),
    )


def _download_verified(download: Downloader, archive: PublicArchiveSpec, target: Path) -> str:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{archive.filename}.", suffix=".part", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        download(archive.url, temporary)
        digest = verify_sha256(temporary, archive.sha256)
        if target.exists():
            raise FileExistsError(f"refusing to replace archive created concurrently: {target}")
        temporary.replace(target)
        temporary = None
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download_https(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "pivot-research-public-data/0.1"})
    with urlopen(request, timeout=60) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _archive_spec(payload: Mapping[str, Any], kind: str) -> PublicArchiveSpec:
    filename = _path_component(_required_str(payload, "filename"), f"{kind}.filename")
    url = _required_str(payload, "url")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "data.binance.vision":
        raise ValueError(f"{kind}.url must use https://data.binance.vision")
    if Path(parsed.path).name != filename:
        raise ValueError(f"{kind}.url path must end with its declared filename")
    digest = _required_str(payload, "sha256").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{kind}.sha256 must be a lowercase 64-character hexadecimal digest")
    return PublicArchiveSpec(kind=kind, filename=filename, url=url, sha256=digest)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _path_component(value: str, name: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be a base name without path traversal")
    return value
