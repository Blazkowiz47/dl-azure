"""Azure shard path providers for WebDataset-backed tar datasets."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import random
import tarfile
import tempfile
import time
from collections.abc import Iterable, Iterator
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
from dl_core.datasets import TarShardWrapper
from filelock import FileLock
from torch.utils.data import Dataset

from dl_azure.datasets.base import AzureBlobMixin, AzureComputeMixin

logger = logging.getLogger(__name__)


class _AzureRetryingShardCache:
    """Lazily download Azure shards into a process-safe local cache."""

    def __init__(
        self,
        cache_dir: str,
        *,
        cache_size_bytes: int,
        download_retries: int,
        retry_backoff_seconds: float,
        retry_backoff_max_seconds: float,
        retry_jitter: bool,
        connection_timeout_seconds: float,
        read_timeout_seconds: float,
        lock_timeout_seconds: float,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.cache_dir.parent / f".{self.cache_dir.name}.locks"
        self.part_dir = self.cache_dir.parent / f".{self.cache_dir.name}.parts"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.part_dir.mkdir(parents=True, exist_ok=True)
        self.download_retries = download_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.retry_jitter = retry_jitter
        self.connection_timeout_seconds = connection_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        from webdataset.cache import LRUCleanup

        self.cleaner = LRUCleanup(
            str(self.cache_dir),
            cache_size_bytes,
            interval=30,
        )

    def __call__(
        self, urls: Iterable[str | dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        for item in urls:
            sample = dict(item) if isinstance(item, dict) else {"url": item}
            url = str(sample["url"])
            parsed = urlsplit(url)
            if parsed.scheme in {"", "file"}:
                destination = Path(unquote(parsed.path if parsed.scheme else url))
            else:
                identity = urlunsplit(
                    (
                        parsed.scheme.lower(),
                        parsed.netloc.lower(),
                        unquote(parsed.path),
                        "",
                        "",
                    )
                )
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                basename = Path(unquote(parsed.path)).name or "shard.tar"
                basename = "".join(
                    character
                    if character.isalnum() or character in "._-"
                    else "_"
                    for character in basename
                )
                destination = self.cache_dir / f"{digest}-{basename}"
                with FileLock(
                    str(self.lock_dir / f"{destination.name}.lock"),
                    timeout=self.lock_timeout_seconds,
                ):
                    valid_cache = False
                    if destination.is_file():
                        try:
                            with tarfile.open(destination, "r:*") as archive:
                                archive.next()
                            valid_cache = True
                        except (OSError, tarfile.TarError):
                            destination.unlink(missing_ok=True)
                    if valid_cache:
                        os.utime(destination, None)
                    else:
                        self.cleaner.cleanup()

                        for attempt in range(self.download_retries + 1):
                            descriptor, temporary_name = tempfile.mkstemp(
                                dir=self.part_dir,
                                prefix=f".{destination.name}.",
                                suffix=".part",
                            )
                            temporary_path = Path(temporary_name)
                            client: BlobClient | None = None
                            try:
                                client = BlobClient.from_blob_url(url)
                                request_options = {
                                    "connection_timeout": (
                                        self.connection_timeout_seconds
                                    ),
                                    "read_timeout": self.read_timeout_seconds,
                                }
                                properties = client.get_blob_properties(
                                    **request_options
                                )
                                etag = getattr(properties, "etag", None)
                                expected_size = getattr(properties, "size", None)
                                download_options = dict(request_options)
                                if etag is not None:
                                    download_options.update(
                                        etag=etag,
                                        match_condition=(
                                            MatchConditions.IfNotModified
                                        ),
                                    )
                                download_options["validate_content"] = True
                                downloader = client.download_blob(
                                    **download_options
                                )
                                with os.fdopen(descriptor, "wb") as handle:
                                    descriptor = -1
                                    downloaded_size = downloader.readinto(handle)
                                if (
                                    expected_size is not None
                                    and downloaded_size != int(expected_size)
                                ):
                                    raise OSError(
                                        f"Azure shard size mismatch for {url}: "
                                        f"expected {expected_size}, downloaded "
                                        f"{downloaded_size}"
                                    )
                                try:
                                    with tarfile.open(
                                        temporary_path, "r:*"
                                    ) as archive:
                                        archive.next()
                                except (OSError, tarfile.TarError) as exc:
                                    raise ValueError(
                                        "Downloaded Azure shard is not a tar "
                                        f"archive: {url}"
                                    ) from exc
                                os.replace(temporary_path, destination)
                                break
                            except Exception as exc:
                                if descriptor >= 0:
                                    os.close(descriptor)
                                    descriptor = -1
                                temporary_path.unlink(missing_ok=True)
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
                                    and (
                                        status_code in {408, 429}
                                        or status_code >= 500
                                    )
                                )
                                if (
                                    not retryable
                                    or attempt >= self.download_retries
                                ):
                                    raise
                                delay = min(
                                    self.retry_backoff_seconds * (2**attempt),
                                    self.retry_backoff_max_seconds,
                                )
                                if self.retry_jitter and delay > 0:
                                    delay *= random.uniform(0.5, 1.5)
                                logger.warning(
                                    "Azure shard download failed (%s/%s) for "
                                    "%s: %s; retrying in %.2fs",
                                    attempt + 1,
                                    self.download_retries + 1,
                                    parsed.path,
                                    exc,
                                    delay,
                                )
                                time.sleep(delay)
                            finally:
                                if descriptor >= 0:
                                    os.close(descriptor)
                                if client is not None:
                                    with contextlib.suppress(Exception):
                                        client.close()
            sample.update(
                url=url,
                stream=destination.open("rb"),
                local_path=str(destination),
            )
            yield sample


class AzureComputeTarShardWrapper(AzureComputeMixin, TarShardWrapper):
    """Provide mounted Azure ML tar paths to WebDataset."""

    @property
    def shard_root(self) -> Path:
        """Resolve relative shard paths beneath the mounted dataset root."""

        return self.root_dir


class AzureStreamingTarShardWrapper(AzureBlobMixin, TarShardWrapper):
    """Provide authenticated Azure blob URLs to WebDataset."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        cache_config = self.config.get("cache", {})
        self._azure_shard_cache_options: dict[str, Any] | None = None
        if cache_config and cache_config.get("enabled", True):
            if "cache_size" in cache_config:
                raise ValueError(
                    "Azure tar cache size is configured in GB; replace cache_size "
                    "with cache_size_gb"
                )
            cache_size_gb = float(cache_config.get("cache_size_gb", 3000))
            if cache_size_gb <= 0:
                raise ValueError("cache.cache_size_gb must be greater than zero")
            download_retries = int(cache_config.get("download_retries", 5))
            if download_retries < 0:
                raise ValueError("cache.download_retries cannot be negative")
            retry_backoff_seconds = float(
                cache_config.get("retry_backoff_seconds", 1)
            )
            retry_backoff_max_seconds = float(
                cache_config.get("retry_backoff_max_seconds", 30)
            )
            if retry_backoff_seconds < 0 or retry_backoff_max_seconds < 0:
                raise ValueError("Azure tar cache retry backoff cannot be negative")
            webdataset_config = dict(self.webdataset_config)
            cache_dir = str(
                Path(
                    cache_config.get("cache_dir", "~/.cache/dl-azure/shards")
                ).expanduser()
            )
            cache_size_bytes = int(cache_size_gb * 1024**3)
            webdataset_config["cache_dir"] = cache_dir
            webdataset_config["cache_size"] = cache_size_bytes
            self.webdataset_config = webdataset_config
            self._azure_shard_cache_options = {
                "cache_dir": cache_dir,
                "cache_size_bytes": cache_size_bytes,
                "download_retries": download_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
                "retry_backoff_max_seconds": retry_backoff_max_seconds,
                "retry_jitter": bool(cache_config.get("retry_jitter", True)),
                "connection_timeout_seconds": float(
                    cache_config.get("connection_timeout_seconds", 20)
                ),
                "read_timeout_seconds": float(
                    cache_config.get("read_timeout_seconds", 120)
                ),
                "lock_timeout_seconds": float(
                    cache_config.get("lock_timeout_seconds", 3600)
                ),
            }

    def build_dataset(self, data: list[dict], split: str) -> Dataset:
        """Replace WebDataset's cache stage with the retrying Azure cache."""

        dataset = super().build_dataset(data, split)
        if self._azure_shard_cache_options is None:
            return dataset

        from webdataset.cache import FileCache

        pipelines = getattr(dataset, "datasets", [dataset])
        replaced = 0
        for pipeline in pipelines:
            for index, stage in enumerate(pipeline.pipeline):
                if isinstance(stage, FileCache):
                    pipeline.pipeline[index] = _AzureRetryingShardCache(
                        **self._azure_shard_cache_options
                    )
                    replaced += 1
                    break
        if replaced != len(pipelines):
            raise RuntimeError("Could not install the retrying Azure shard cache")
        return dataset

    def build_shard_sources(self, split: str) -> list[dict[str, Any]]:
        """Build weighted Azure blob sources; subclasses may override."""

        remote_shards = self.get_configured_shards(split)
        if not remote_shards:
            prefixes = self.config.get("shard_prefixes", {})
            if isinstance(prefixes, dict):
                prefixes = prefixes.get(split, [])
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            if not prefixes:
                fallback_prefix = self.config.get("shard_prefix")
                prefixes = [fallback_prefix] if fallback_prefix else []
            if not prefixes:
                raise ValueError(
                    "Azure tar datasets require shards, shard_prefix, or shard_prefixes"
                )
            remote_shards = [
                {"path": blob_path}
                for prefix in prefixes
                for blob_path in self.scan_paths(prefix)
                if blob_path.lower().endswith((".tar", ".tar.gz", ".tgz"))
            ]
        return [{"name": split, "weight": 1.0, "shards": remote_shards}]

    def get_shard_sources(self, split: str) -> list[dict[str, Any]]:
        """Convert logical Azure blob sources into authenticated URLs."""

        authenticated_sources = []
        for source in self.build_shard_sources(split):
            authenticated_shards = []
            for configured_shard in source.get("shards", []):
                shard = (
                    dict(configured_shard)
                    if isinstance(configured_shard, dict)
                    else {"path": str(configured_shard)}
                )
                blob_path = str(shard["path"])
                if not blob_path.lower().endswith((".tar", ".tar.gz", ".tgz")):
                    raise ValueError(
                        f"Azure WebDataset shards must be tar archives: {blob_path}"
                    )
                authenticated_shards.append(
                    {
                        **shard,
                        "path": self.azure_service.get_blob_sas_url(
                            self.container_name,
                            blob_path,
                            expiry_hours=int(self.config.get("sas_expiry_hours", 168)),
                        ),
                        "source_path": blob_path,
                    }
                )
            authenticated_sources.append({**source, "shards": authenticated_shards})
        return authenticated_sources


__all__ = [
    "AzureComputeTarShardWrapper",
    "AzureStreamingTarShardWrapper",
]
