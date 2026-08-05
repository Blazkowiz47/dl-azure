"""Azure-mounted and blob-cached wrappers for indexed tar shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.datasets import TarShardWrapper

from dl_azure.datasets.base import AzureBlobMixin, AzureComputeMixin
from dl_azure.storage import AzureShardCache


class AzureComputeTarShardWrapper(AzureComputeMixin, TarShardWrapper):
    """Read indexed tar shards from an Azure ML mount or compatible local root."""

    @property
    def shard_root(self) -> Path:
        """Resolve relative shard paths beneath the mounted dataset root."""

        return self.root_dir


class AzureStreamingTarShardWrapper(AzureBlobMixin, TarShardWrapper):
    """Cache Azure tar shards locally before indexed sample reads."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        cache_config = self.config.get("cache", {})
        configured_cache_dir = Path(
            cache_config.get("cache_dir", "~/.cache/dl-azure")
        ).expanduser()
        shard_cache_dir = Path(
            cache_config.get("shard_cache_dir", configured_cache_dir / "shards")
        ).expanduser()
        self.shard_cache = AzureShardCache(
            shard_cache_dir,
            lock_timeout_seconds=float(cache_config.get("lock_timeout_seconds", 300)),
        )

    def get_shards(self, split: str) -> list[dict[str, Any]]:
        """Resolve configured or discovered Azure blobs into local cached shards."""

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
                for blob_path in self.scan_paths(prefix, extension="tar")
            ]

        local_shards: list[dict[str, Any]] = []
        for shard in remote_shards:
            blob_path = str(shard["path"])
            if not blob_path.lower().endswith(".tar"):
                raise ValueError(
                    f"Indexed Azure shards must be uncompressed .tar blobs: {blob_path}"
                )
            local_tar = self.shard_cache.materialize(
                self.azure_service,
                self.container_name,
                blob_path,
            )
            remote_index = str(
                shard.get("index_path", f"{blob_path}{self.index_suffix}")
            )
            local_index: Path | None = None
            if self.azure_service.blob_exists(self.container_name, remote_index):
                local_index = self.shard_cache.materialize(
                    self.azure_service,
                    self.container_name,
                    remote_index,
                )

            local_shard = {
                **shard,
                "path": str(local_tar),
                "source_path": blob_path,
            }
            if local_index is not None:
                local_shard["index_path"] = str(local_index)
            else:
                local_shard.pop("index_path", None)
            local_shards.append(local_shard)
        return local_shards


__all__ = [
    "AzureComputeTarShardWrapper",
    "AzureStreamingTarShardWrapper",
]
