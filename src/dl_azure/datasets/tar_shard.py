"""Azure shard path providers for WebDataset-backed tar datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dl_core.datasets import TarShardWrapper
from torch.utils.data import Dataset

from dl_azure.datasets.base import AzureBlobMixin, AzureComputeMixin
from dl_azure.storage.shard_cache import AzureShardCache
from dl_azure.storage.shard_prefetch import ShardPrefetcher


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
        cache_config = self.config.get("cache") or {}
        if not isinstance(cache_config, dict):
            raise TypeError("Azure tar cache configuration must be a mapping")
        if not cache_config.get("enabled", True):
            raise ValueError(
                "Azure streaming tar requires its shard cache so signed URLs "
                "do not enter sample metadata"
            )
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
        retry_backoff_seconds = float(cache_config.get("retry_backoff_seconds", 1))
        retry_backoff_max_seconds = float(
            cache_config.get("retry_backoff_max_seconds", 30)
        )
        if retry_backoff_seconds < 0 or retry_backoff_max_seconds < 0:
            raise ValueError("Azure tar cache retry backoff cannot be negative")
        cache_dir = str(
            Path(
                cache_config.get("cache_dir")
                or self.webdataset_config.get("cache_dir")
                or "~/.cache/dl-azure/shards"
            ).expanduser()
        )
        cache_size_bytes = int(cache_size_gb * 1024**3)
        self.webdataset_config = {
            **self.webdataset_config,
            "cache_dir": cache_dir,
            "cache_size": cache_size_bytes,
        }
        self._azure_shard_cache_options: dict[str, Any] = {
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

    def create_shard_prefetcher(self) -> ShardPrefetcher:
        """Create a trainer-owned controller for explicit shard/cycle plans.

        Use it as a context manager in the trainer, outside DataLoader workers.
        Paths passed to plan() are logical container-relative blob paths.
        """
        options = self.config.get("prefetch", {})
        if not isinstance(options, dict):
            raise TypeError("dataset.prefetch must be a mapping")

        def resolve_url(blob_path: str) -> str:
            if not blob_path.lower().endswith((".tar", ".tar.gz", ".tgz")):
                raise ValueError("Prefetch shard paths must identify tar archives")
            return self.azure_service.get_blob_sas_url(
                self.container_name,
                blob_path,
                expiry_hours=int(self.config.get("sas_expiry_hours", 168)),
            )

        return ShardPrefetcher(
            AzureShardCache(**self._azure_shard_cache_options),
            resolve_url=resolve_url,
            **{"enabled": False, **options},
        )

    def _transform_webdataset_sample(
        self,
        sample: dict[str, Any],
        *,
        split: str,
        metadata_by_shard: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve public shard URLs against metadata from either core layout."""
        public_url = str(sample.get("__url__", ""))
        if public_url not in metadata_by_shard:
            for signed_url, metadata in tuple(metadata_by_shard.items()):
                parsed_url = urlsplit(signed_url)
                if urlunsplit(
                    (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "")
                ) == public_url:
                    metadata_by_shard[public_url] = metadata
                    break
        return super()._transform_webdataset_sample(
            sample,
            split=split,
            metadata_by_shard=metadata_by_shard,
        )

    def build_dataset(self, data: list[dict], split: str) -> Dataset:
        """Replace WebDataset's cache stage with the retrying Azure cache."""

        dataset = super().build_dataset(data, split)
        from webdataset.cache import FileCache

        pipelines = getattr(dataset, "datasets", [dataset])
        replaced = 0
        for pipeline in pipelines:
            for index, stage in enumerate(pipeline.pipeline):
                if isinstance(stage, FileCache):
                    pipeline.pipeline[index] = AzureShardCache(
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
                authenticated_url = self.azure_service.get_blob_sas_url(
                    self.container_name,
                    blob_path,
                    expiry_hours=int(self.config.get("sas_expiry_hours", 168)),
                )
                parsed_url = urlsplit(authenticated_url)
                authenticated_shards.append(
                    {
                        **shard,
                        "path": authenticated_url,
                        "public_url": urlunsplit(
                            (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "")
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
