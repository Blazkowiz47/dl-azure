"""Azure tar downloads with process-safe cache reservations."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceModifiedError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.storage.blob import BlobClient
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


class CacheCapacityError(RuntimeError):
    """The cache cannot fit a download without evicting reserved shards."""


class AzureShardCache:
    """Share validated downloads and cache reservations across local processes."""

    def __init__(
        self,
        cache_dir: str,
        *,
        cache_size_bytes: int,
        path_resolver: Callable[[str], str | Path | None] | None = None,
        state_dir: str | Path | None = None,
        download_retries: int = 5,
        retry_backoff_seconds: float = 1,
        retry_backoff_max_seconds: float = 30,
        retry_jitter: bool = True,
        connection_timeout_seconds: float = 20,
        read_timeout_seconds: float = 120,
        lock_timeout_seconds: float = 3600,
    ) -> None:
        if cache_size_bytes <= 0:
            raise ValueError("cache_size_bytes must be greater than zero")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.path_resolver = path_resolver
        state_root = Path(state_dir).expanduser().resolve() if state_dir else None
        self.lock_dir = (
            state_root / "locks"
            if state_root
            else self.cache_dir.parent / f".{self.cache_dir.name}.locks"
        )
        self.part_dir = (
            state_root / "parts"
            if state_root
            else self.cache_dir.parent / f".{self.cache_dir.name}.parts"
        )
        self.pin_dir = (
            state_root / "pins"
            if state_root
            else self.cache_dir.parent / f".{self.cache_dir.name}.pins"
        )
        self.record_dir = (
            state_root / "records"
            if state_root
            else self.cache_dir.parent / f".{self.cache_dir.name}.records"
        )
        for directory in (
            self.cache_dir,
            self.lock_dir,
            self.part_dir,
            self.pin_dir,
            self.record_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.cache_size_bytes = cache_size_bytes
        self.download_retries = download_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.retry_jitter = retry_jitter
        self.connection_timeout_seconds = connection_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        # Adopt files from the default cache written before destination records
        # existed. Custom destinations are registered only when explicitly used.
        with self._guard():
            for path in self.cache_dir.iterdir():
                if path.is_file() and re.fullmatch(
                    r"[0-9a-f]{20}-.+\.(tar|tar\.gz|tgz)", path.name
                ):
                    record = self.record_dir / f"{self._cache_key(path)}.json"
                    if not record.exists():
                        self._write_record(
                            record,
                            {
                                "path": str(path),
                                "identity": None,
                                "accessed": path.stat().st_mtime,
                            },
                        )

    def local_path(self, url: str) -> Path:
        """Resolve a destination; custom resolvers receive a URL without SAS tokens.

        Relative results are beneath cache_dir; absolute results may be anywhere.
        Returning None selects the default hashed filename.
        """
        parsed = urlsplit(url)
        if parsed.scheme in {"", "file"}:
            return Path(unquote(parsed.path if parsed.scheme else url))
        identity = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), unquote(parsed.path), "", "")
        )
        if self.path_resolver is not None:
            public_url = urlunsplit(
                (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
            )
            resolved = self.path_resolver(public_url)
            if resolved is not None:
                path = Path(resolved).expanduser()
                return (path if path.is_absolute() else self.cache_dir / path).resolve()
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        basename = Path(unquote(parsed.path)).name or "shard.tar"
        basename = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in basename
        )
        return self.cache_dir / f"{digest}-{basename}"

    @staticmethod
    def _cache_key(destination: Path) -> str:
        return hashlib.sha256(str(destination).encode("utf-8")).hexdigest()

    @staticmethod
    def _write_record(path: Path, data: dict[str, Any]) -> None:
        # All callers hold the cache guard, so one temporary name is sufficient.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        os.replace(temporary, path)

    def _guard(self) -> FileLock:
        return FileLock(
            str(self.lock_dir / "cache.lock"), timeout=self.lock_timeout_seconds
        )

    @contextlib.contextmanager
    def reserve(self, url: str) -> Iterator[Path]:
        """Prevent eviction until the context exits, including before download."""
        destination = self.local_path(url)
        if urlsplit(url).scheme in {"", "file"}:
            yield destination
            return
        key = self._cache_key(destination)
        parsed = urlsplit(url)
        identity = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), unquote(parsed.path), "", "")
        )
        pin = self.pin_dir / f"{key}.{uuid.uuid4().hex}.pin"
        lease = FileLock(str(pin), thread_local=False)
        with self._guard():
            record = self.record_dir / f"{key}.json"
            entry = (
                json.loads(record.read_text())
                if record.exists()
                else {"path": str(destination), "identity": None, "accessed": 0}
            )
            if entry["identity"] not in {None, identity}:
                raise ValueError(
                    "Cache destination is already assigned to a different shard"
                )
            entry["identity"] = identity
            self._write_record(record, entry)
            lease.acquire()
        try:
            yield destination
        finally:
            with self._guard():
                lease.release()
                pin.unlink(missing_ok=True)

    def _reserve_space(self, destination: Path, size: int, partial: Path) -> None:
        # Called with the shard's download lock held. The global guard makes
        # admission, eviction, pin creation, and file promotion atomic together.
        if size > self.cache_size_bytes:
            raise CacheCapacityError("Shard is larger than cache.cache_size_gb")
        with self._guard():
            reserved = 0
            disk_reserved = 0
            device = destination.parent.stat().st_dev
            for record in self.part_dir.glob("*.size"):
                name = record.name[:-5]
                download = json.loads(record.read_text())
                if isinstance(download, int):
                    # Previous releases kept partials beside the size record.
                    partials = list(self.part_dir.glob(f".{name}.*.part"))
                    download = {"size": download, "device": self.part_dir.stat().st_dev}
                else:
                    partials = [Path(download["partial"])]
                download_lock = FileLock(str(self.lock_dir / f"{name}.lock"))
                try:
                    with download_lock.acquire(timeout=0):
                        record.unlink(missing_ok=True)
                        for abandoned in partials:
                            abandoned.unlink(missing_ok=True)
                except Timeout:
                    reserved += download["size"]
                    if download["device"] == device:
                        written = sum(
                            path.stat().st_size for path in partials if path.exists()
                        )
                        disk_reserved += max(0, download["size"] - written)

            files = []
            for record in self.record_dir.glob("*.json"):
                entry = json.loads(record.read_text())
                path = Path(entry["path"])
                if path.is_file():
                    files.append(
                        (entry["accessed"], record.stem, path, path.stat().st_size)
                    )
            files.sort()
            used = sum(item[3] for item in files)
            for _, key, path, file_size in files:
                if used + reserved + size <= self.cache_size_bytes:
                    break
                protected = False
                pins = set(self.pin_dir.glob(f"{key}.*.pin")) | set(
                    self.pin_dir.glob(f"{path.name}.*.pin")
                )
                for pin in pins:
                    try:
                        with FileLock(str(pin)).acquire(timeout=0):
                            pass
                    except Timeout:
                        protected = True
                    else:
                        pin.unlink(missing_ok=True)
                if not protected:
                    used -= file_size
                    path.unlink()
            if used + reserved + size > self.cache_size_bytes:
                raise CacheCapacityError(
                    "Shard cache is full: release unused shard reservations or "
                    "increase cache.cache_size_gb"
                )
            if shutil.disk_usage(destination.parent).free < size + disk_reserved:
                raise CacheCapacityError(
                    "Insufficient disk space for queued shard downloads"
                )
            self._write_record(
                self.part_dir / f"{self._cache_key(destination)}.size",
                {"size": size, "partial": str(partial), "device": device},
            )

    def ensure(self, url: str) -> Path:
        """Download on a miss; callers retaining the path should also reserve it."""
        parsed = urlsplit(url)
        public_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        with self.reserve(url) as destination:
            if parsed.scheme in {"", "file"}:
                return destination
            key = self._cache_key(destination)
            size_record = self.part_dir / f"{key}.size"
            record = self.record_dir / f"{key}.json"
            with FileLock(
                str(self.lock_dir / f"{key}.lock"),
                timeout=self.lock_timeout_seconds,
            ):
                # Acquiring this lock proves any previous download of this
                # shard has exited. Clear its abandoned reservation first.
                with self._guard():
                    if size_record.exists():
                        Path(json.loads(size_record.read_text())["partial"]).unlink(
                            missing_ok=True
                        )
                        size_record.unlink()
                    # A process can exit between creating its temporary file
                    # and recording its size reservation.
                    for abandoned in destination.parent.glob(f".{key}.*.part"):
                        abandoned.unlink(missing_ok=True)
                if destination.is_file():
                    try:
                        with tarfile.open(destination, "r:*") as archive:
                            archive.next()
                    except (OSError, tarfile.TarError):
                        with self._guard():
                            destination.unlink(missing_ok=True)
                    else:
                        with self._guard():
                            entry = json.loads(record.read_text())
                            entry["accessed"] = time.time()
                            self._write_record(record, entry)
                        return destination

                for attempt in range(self.download_retries + 1):
                    descriptor = -1
                    temporary_path: Path | None = None
                    client: BlobClient | None = None
                    reserved = False
                    try:
                        client = BlobClient.from_blob_url(url)
                        request_options = {
                            "connection_timeout": self.connection_timeout_seconds,
                            "read_timeout": self.read_timeout_seconds,
                        }
                        properties = client.get_blob_properties(**request_options)
                        expected_size = int(properties.size)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        descriptor, temporary_name = tempfile.mkstemp(
                            dir=destination.parent,
                            prefix=f".{key}.",
                            suffix=".part",
                        )
                        temporary_path = Path(temporary_name)
                        self._reserve_space(destination, expected_size, temporary_path)
                        reserved = True
                        download_options = dict(request_options)
                        etag = getattr(properties, "etag", None)
                        if etag is not None:
                            download_options.update(
                                etag=etag, match_condition=MatchConditions.IfNotModified
                            )
                        download_options["validate_content"] = True
                        downloader = client.download_blob(**download_options)
                        with os.fdopen(descriptor, "wb") as handle:
                            descriptor = -1
                            downloaded_size = downloader.readinto(handle)
                        if downloaded_size != expected_size:
                            raise OSError(f"Azure shard size mismatch for {public_url}")
                        try:
                            with tarfile.open(temporary_path, "r:*") as archive:
                                archive.next()
                        except (OSError, tarfile.TarError) as exc:
                            raise ValueError(
                                f"Downloaded Azure shard is not a tar archive: {public_url}"
                            ) from exc
                        with self._guard():
                            os.replace(temporary_path, destination)
                            entry = json.loads(record.read_text())
                            entry["accessed"] = time.time()
                            self._write_record(record, entry)
                            size_record.unlink()
                            reserved = False
                        return destination
                    except CacheCapacityError:
                        raise
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        retryable = isinstance(
                            exc,
                            (
                                OSError,
                                TimeoutError,
                                ServiceRequestError,
                                ServiceResponseError,
                                ResourceModifiedError,
                                ValueError,
                            ),
                        ) or (
                            isinstance(exc, HttpResponseError)
                            and status_code is not None
                            and (status_code in {408, 429} or status_code >= 500)
                        )
                        if not retryable or attempt >= self.download_retries:
                            raise RuntimeError(
                                f"Azure shard download failed for {public_url}: "
                                f"{type(exc).__name__}"
                            ) from None
                        delay = min(
                            self.retry_backoff_seconds * (2**attempt),
                            self.retry_backoff_max_seconds,
                        )
                        if self.retry_jitter and delay > 0:
                            delay *= random.uniform(0.5, 1.5)
                        logger.warning(
                            "Azure shard download failed (%s/%s) for %s: %s; retrying in %.2fs",
                            attempt + 1,
                            self.download_retries + 1,
                            parsed.path,
                            type(exc).__name__,
                            delay,
                        )
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                        with self._guard():
                            if temporary_path is not None:
                                temporary_path.unlink(missing_ok=True)
                            if reserved:
                                size_record.unlink(missing_ok=True)
                        if client is not None:
                            with contextlib.suppress(Exception):
                                client.close()
                    time.sleep(delay)
        raise RuntimeError("Azure shard download did not complete")

    def __call__(
        self, urls: Iterable[str | dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Open WebDataset streams while holding their cache reservations."""
        for item in urls:
            sample = dict(item) if isinstance(item, dict) else {"url": item}
            url = str(sample["url"])
            parsed = urlsplit(url)
            with self.reserve(url), self.ensure(url).open("rb") as stream:
                sample.update(
                    url=urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
                    stream=stream,
                    local_path=str(self.local_path(url)),
                )
                yield sample
