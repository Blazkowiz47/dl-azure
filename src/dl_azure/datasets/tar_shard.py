"""Azure shard path providers for WebDataset-backed tar datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.datasets import TarShardWrapper

from dl_azure.datasets.base import AzureBlobMixin, AzureComputeMixin


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
        if cache_config and cache_config.get("enabled", True):
            webdataset_config = dict(self.webdataset_config)
            webdataset_config.setdefault(
                "cache_dir",
                str(
                    Path(
                        cache_config.get("cache_dir", "~/.cache/dl-azure/shards")
                    ).expanduser()
                ),
            )
            if "cache_size" in cache_config:
                webdataset_config.setdefault("cache_size", cache_config["cache_size"])
            self.webdataset_config = webdataset_config

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
