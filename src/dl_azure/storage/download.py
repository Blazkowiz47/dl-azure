"""Configurable AzCopy and SDK downloads into caller-owned temporary files."""

from __future__ import annotations

import errno
import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError
from azure.core.pipeline.transport import RequestsTransport
from azure.storage.blob import BlobClient

logger = logging.getLogger(__name__)
_AZCOPY_PATHS: dict[tuple[int, str], str | None] = {}
_AZCOPY_LOCK = threading.Lock()


class AzureDownloader:
    """Share file transfer settings without retaining clients or credentials.

    The caller owns temporary-file cleanup, validation specific to its format,
    and promotion to the final destination. Missing AzCopy is remembered per
    executable per process; transfer failures fall back only for that file.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        validate_content: bool = False,
        connection_timeout_seconds: float = 20,
        read_timeout_seconds: float = 120,
    ) -> None:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("azure.download must be a mapping")
        unknown = set(config) - {"backend", "fallback_to_sdk", "azcopy", "sdk"}
        if unknown:
            raise ValueError(f"Unknown azure.download options: {sorted(unknown)}")
        self.backend = config.get("backend", "azcopy")
        if self.backend not in {"azcopy", "sdk"}:
            raise ValueError("azure.download.backend must be 'azcopy' or 'sdk'")
        self.fallback_to_sdk = config.get("fallback_to_sdk", True)
        if not isinstance(self.fallback_to_sdk, bool):
            raise ValueError("azure.download.fallback_to_sdk must be a boolean")

        self.sdk = {
            "max_concurrency": 32,
            "connection_pool_size": 32,
            "chunk_size_mib": 4,
            "initial_request_size_mib": 32,
            "read_buffer_kib": 64,
            "connection_timeout_seconds": connection_timeout_seconds,
            "read_timeout_seconds": read_timeout_seconds,
            "validate_content": validate_content,
        }
        self.azcopy = {
            "executable": "azcopy",
            "concurrency": None,
            "buffer_gb": None,
            "cap_mbps": 0,
            "timeout_seconds": None,
        }
        for name, options in (("sdk", self.sdk), ("azcopy", self.azcopy)):
            supplied = config.get(name, {})
            if not isinstance(supplied, dict):
                raise TypeError(f"azure.download.{name} must be a mapping")
            unknown = set(supplied) - set(options)
            if unknown:
                raise ValueError(
                    f"Unknown azure.download.{name} options: {sorted(unknown)}"
                )
            options.update(supplied)

        for name, value in self.sdk.items():
            if name == "validate_content":
                if not isinstance(value, bool):
                    raise ValueError(
                        "azure.download.sdk.validate_content must be a boolean"
                    )
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (not name.endswith("_seconds") and not isinstance(value, int))
            ):
                raise ValueError(f"azure.download.sdk.{name} must be positive")
        if self.sdk["validate_content"] and self.sdk["chunk_size_mib"] > 4:
            raise ValueError(
                "SDK transactional MD5 requires chunk_size_mib <= 4; "
                "set azure.download.sdk.validate_content to false for larger chunks"
            )
        executable = self.azcopy["executable"]
        if not isinstance(executable, str) or not executable:
            raise ValueError("azure.download.azcopy.executable must be a nonempty path")
        concurrency = self.azcopy["concurrency"]
        if concurrency is not None and concurrency != "AUTO" and (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency <= 0
        ):
            raise ValueError("azure.download.azcopy.concurrency must be positive or AUTO")
        for name in ("buffer_gb", "cap_mbps", "timeout_seconds"):
            value = self.azcopy[name]
            if value is None and name != "cap_mbps":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if name == "cap_mbps" else value <= 0)
            ):
                raise ValueError(f"azure.download.azcopy.{name} has an invalid value")

    @property
    def request_options(self) -> dict[str, Any]:
        """Options shared by file downloads and SDK reads into memory."""
        return {
            "max_concurrency": self.sdk["max_concurrency"],
            "validate_content": self.sdk["validate_content"],
            **self.property_options,
        }

    @property
    def property_options(self) -> dict[str, Any]:
        """Timeouts for the metadata requests surrounding a transfer."""
        return {
            "connection_timeout": self.sdk["connection_timeout_seconds"],
            "read_timeout": self.sdk["read_timeout_seconds"],
        }

    def client_options(self) -> dict[str, Any]:
        """Build a fresh transport with the configured pool and read buffer."""
        session = requests.Session()
        pool_size = self.sdk["connection_pool_size"]
        for scheme in ("http://", "https://"):
            session.mount(
                scheme,
                requests.adapters.HTTPAdapter(
                    pool_connections=pool_size, pool_maxsize=pool_size,
                ),
            )
        return {
            "transport": RequestsTransport(
                session=session,
                session_owner=True,
                connection_data_block_size=self.sdk["read_buffer_kib"] * 1024,
                **self.property_options,
            ),
            "max_chunk_get_size": self.sdk["chunk_size_mib"] * 1024**2,
            "max_single_get_size": self.sdk["initial_request_size_mib"] * 1024**2,
        }

    def download(
        self,
        client: BlobClient,
        temporary_path: Path,
        *,
        properties: Any = None,
        azcopy_url: str | Callable[[], str] | None = None,
    ) -> None:
        """Download and verify a blob; never promote an incomplete file."""
        if properties is None:
            properties = client.get_blob_properties(**self.property_options)
        expected_size = int(properties.size)
        etag = getattr(properties, "etag", None)
        backends = [self.backend]
        executable = None
        if self.backend == "azcopy":
            key = (os.getpid(), self.azcopy["executable"])
            with _AZCOPY_LOCK:
                if key not in _AZCOPY_PATHS:
                    _AZCOPY_PATHS[key] = shutil.which(self.azcopy["executable"])
                    if _AZCOPY_PATHS[key] is None:
                        logger.info("AzCopy is unavailable; subsequent downloads use the SDK")
                executable = _AZCOPY_PATHS[key]
            if executable is None:
                if not self.fallback_to_sdk:
                    raise RuntimeError("AzCopy is unavailable and SDK fallback is disabled")
                backends = ["sdk"]
            elif self.fallback_to_sdk:
                backends.append("sdk")

        for backend in backends:
            try:
                if backend == "azcopy":
                    source = azcopy_url() if callable(azcopy_url) else azcopy_url
                    source = source or client.url
                    # Isolate job plans and logs, which can contain signed URLs.
                    with tempfile.TemporaryDirectory(prefix="dl-azure-azcopy-") as state:
                        environment = os.environ.copy()
                        for name, variable in (
                            ("concurrency", "AZCOPY_CONCURRENCY_VALUE"),
                            ("buffer_gb", "AZCOPY_BUFFER_GB"),
                        ):
                            if self.azcopy[name] is not None:
                                environment[variable] = str(self.azcopy[name])
                        for name, variable in (
                            ("logs", "AZCOPY_LOG_LOCATION"),
                            ("plans", "AZCOPY_JOB_PLAN_LOCATION"),
                        ):
                            directory = Path(state) / name
                            directory.mkdir()
                            environment[variable] = str(directory)
                        command = [
                            executable,
                            "copy",
                            source,
                            str(temporary_path.resolve()),
                            "--from-to=BlobLocal",
                            "--overwrite=true",
                            "--check-length=true",
                            "--check-md5=FailIfDifferent",
                            "--output-type=json",
                            "--output-level=essential",
                            "--log-level=NONE",
                            f"--cap-mbps={self.azcopy['cap_mbps']}",
                        ]
                        try:
                            result = subprocess.run(
                                command,
                                shell=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                env=environment,
                                timeout=self.azcopy["timeout_seconds"],
                            )
                        except subprocess.TimeoutExpired:
                            raise TimeoutError("AzCopy download timed out") from None
                        except OSError as exc:
                            if exc.errno in {errno.ENOENT, errno.ENOEXEC, errno.EACCES}:
                                with _AZCOPY_LOCK:
                                    _AZCOPY_PATHS[key] = None
                            raise
                        if result.returncode != 0:
                            raise OSError(f"AzCopy exited with status {result.returncode}")
                    # AzCopy cannot use the SDK's If-Match condition. Confirm the
                    # source still matches the properties used for admission.
                    current = client.get_blob_properties(**self.property_options)
                    if int(current.size) != expected_size or (
                        etag is not None and current.etag != etag
                    ):
                        raise ResourceModifiedError("Azure blob changed during download")
                else:
                    options = self.request_options
                    if etag is not None:
                        options.update(
                            etag=etag, match_condition=MatchConditions.IfNotModified
                        )
                    with temporary_path.open("wb") as handle:
                        client.download_blob(**options).readinto(handle)

                if temporary_path.stat().st_size != expected_size:
                    raise OSError("Azure blob download size mismatch")
                content_settings = getattr(properties, "content_settings", None)
                content_md5 = getattr(content_settings, "content_md5", None)
                if content_md5:
                    digest = hashlib.md5(usedforsecurity=False)
                    with temporary_path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024**2), b""):
                            digest.update(chunk)
                    if digest.digest() != bytes(content_md5):
                        raise OSError("Azure blob download MD5 mismatch")
                return
            except Exception as exc:
                if backend != "azcopy" or not self.fallback_to_sdk:
                    raise
                # subprocess.run kills and waits on timeout before returning;
                # SDK fallback can now truncate the same staging file safely.
                logger.warning(
                    "AzCopy download failed (%s); using the SDK for this file",
                    type(exc).__name__,
                )
