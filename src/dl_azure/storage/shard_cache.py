"""Process-safe local cache for raw Azure tar shards and sidecar indexes."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dl_azure.storage.client import AzureClientService


class AzureShardCache:
    """Materialize immutable Azure blobs into an identity-validated local cache."""

    def __init__(self, cache_dir: str | Path, *, lock_timeout_seconds: float = 300) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.lock_timeout_seconds = lock_timeout_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, container_name: str, blob_path: str) -> Path:
        """Return a traversal-safe path preserving the blob filename and suffix."""

        parts = [
            quote(part, safe="-_.()")
            for part in blob_path.replace("\\", "/").split("/")
            if part
        ]
        if not parts:
            raise ValueError("blob_path must identify a file")
        return self.cache_dir / quote(container_name, safe="-_.()") / Path(*parts)

    @staticmethod
    def _identity(blob_properties: Any) -> dict[str, Any]:
        size = getattr(blob_properties, "size", None)
        if size is None:
            size = getattr(blob_properties, "content_length", None)
        return {
            "etag": str(getattr(blob_properties, "etag", "")),
            "size": int(size) if size is not None else None,
            "version_id": getattr(blob_properties, "version_id", None),
        }

    @staticmethod
    def _matches(path: Path, metadata_path: Path, identity: dict[str, Any]) -> bool:
        if not path.is_file() or not metadata_path.is_file():
            return False
        if identity["size"] is not None and path.stat().st_size != identity["size"]:
            return False
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle) == identity
        except (OSError, json.JSONDecodeError):
            return False

    def materialize(
        self,
        service: AzureClientService,
        container_name: str,
        blob_path: str,
    ) -> Path:
        """Download a blob once and return its validated local path."""

        destination = self.cache_path(container_name, blob_path)
        metadata_path = destination.with_name(f"{destination.name}.azure.json")
        lock_path = destination.with_name(f"{destination.name}.lock")
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob_client = service.get_blob_client_pooled(container_name, blob_path)
        identity = self._identity(blob_client.get_blob_properties())
        if self._matches(destination, metadata_path, identity):
            return destination

        deadline = time.monotonic() + self.lock_timeout_seconds
        lock_descriptor: int | None = None
        while lock_descriptor is None:
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                if self._matches(destination, metadata_path, identity):
                    return destination
                try:
                    stale = time.time() - lock_path.stat().st_mtime
                    if stale > self.lock_timeout_seconds:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for shard cache lock: {lock_path}")
                time.sleep(0.1)

        try:
            if self._matches(destination, metadata_path, identity):
                return destination
            if not service.download_blob(container_name, blob_path, destination):
                raise OSError(f"Failed to download Azure blob: {blob_path}")
            if identity["size"] is not None and destination.stat().st_size != identity["size"]:
                destination.unlink(missing_ok=True)
                raise OSError(f"Downloaded Azure blob has an unexpected size: {blob_path}")

            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=metadata_path.parent,
                prefix=f".{metadata_path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                    json.dump(identity, handle, separators=(",", ":"))
                os.replace(temporary_name, metadata_path)
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            return destination
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            lock_path.unlink(missing_ok=True)


__all__ = ["AzureShardCache"]
