"""Tests for Azure WebDataset shard path wrappers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from azure.core.exceptions import ServiceResponseTimeoutError
from dl_azure.datasets import (
    AzureComputeTarShardWrapper,
    AzureStreamingTarShardWrapper,
)


def _write_tar(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in {
            "sample.png": b"png-bytes",
            "sample.json": b'{"label":1}',
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class _ComputeTarWrapper(AzureComputeTarShardWrapper):
    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return {**file_dict, "split": split}


class _StreamingTarWrapper(AzureStreamingTarShardWrapper):
    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return {**file_dict, "split": split}


class _DynamicStreamingTarWrapper(_StreamingTarWrapper):
    def build_shard_sources(self, split: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "attack",
                "weight": 0.75,
                "shards": [
                    {"path": path, "group": "attack"}
                    for path in self.config["dynamic_shards"].get(split, [])
                ],
            }
        ]


class _FakeAzureService:
    def __init__(self, paths: dict[str, Path | str]) -> None:
        self.paths = paths
        self.sas_requests: list[tuple[str, str, int]] = []

    def get_blob_sas_url(
        self,
        container_name: str,
        blob_path: str,
        expiry_hours: int = 24,
        permissions: str = "r",
    ) -> str:
        assert permissions == "r"
        self.sas_requests.append((container_name, blob_path, expiry_hours))
        return str(self.paths[blob_path])


def test_compute_tar_wrapper_reads_from_resolved_mount(tmp_path: Path) -> None:
    tar_path = tmp_path / "train" / "demo.tar"
    tar_path.parent.mkdir()
    _write_tar(tar_path)
    wrapper = _ComputeTarWrapper(
        {
            "root_dir": str(tmp_path),
            "allow_local_fallback": False,
            "shards": {"train": [{"path": "train/demo.tar", "group": "real"}]},
            "required_extensions": ["png", "json"],
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "auto_split": False,
        }
    )

    loader = wrapper.get_split("train")
    assert loader is not None
    batch = next(iter(loader))
    assert batch["key"] == ["sample"]
    assert batch["group"] == ["real"]
    assert batch["members"][0] == {
        "json": b'{"label":1}',
        "png": b"png-bytes",
    }


def test_streaming_tar_wrapper_provides_authenticated_shard_urls(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tar_path = tmp_path / "source.tar"
    _write_tar(tar_path)
    service = _FakeAzureService({"train/demo.tar": tar_path})
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService",
        lambda config: service,
    )
    wrapper = _DynamicStreamingTarWrapper(
        {
            "account_name": "demo",
            "container_name": "datasets",
            "azure_config_path": str(tmp_path / "missing-azure-config.json"),
            "dynamic_shards": {"train": ["train/demo.tar"]},
            "required_extensions": ["png", "json"],
            "cache": {"cache_dir": str(tmp_path / "cache")},
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "auto_split": False,
        }
    )

    loader = wrapper.get_split("train")
    assert loader is not None
    batch = next(iter(loader))
    assert batch["source_path"] == ["train/demo.tar"]
    assert batch["group"] == ["attack"]
    assert batch["source_name"] == ["attack"]
    assert batch["source_weight"].tolist() == [0.75]
    assert service.sas_requests == [("datasets", "train/demo.tar", 168)]
    assert wrapper.webdataset_config["cache_dir"] == str(tmp_path / "cache")
    assert wrapper.webdataset_config["cache_size"] == 3000 * 1024**3


def test_streaming_tar_cache_retries_whole_azure_downloads(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tar_path = tmp_path / "source.tar"
    _write_tar(tar_path)
    payload = tar_path.read_bytes()
    shard_url = (
        "https://demo.blob.core.windows.net/datasets/train/demo.tar?sig=secret"
    )
    service = _FakeAzureService({"train/demo.tar": shard_url})
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService",
        lambda config: service,
    )
    attempts: list[str] = []

    class _FakeBlobClient:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def get_blob_properties(self, **kwargs: Any) -> Any:
            assert kwargs == {"connection_timeout": 7.0, "read_timeout": 11.0}
            return SimpleNamespace(
                etag='"etag-1"',
                size=len(payload),
            )

        def download_blob(self, **kwargs: Any) -> Any:
            assert kwargs["connection_timeout"] == 7.0
            assert kwargs["read_timeout"] == 11.0
            assert kwargs["validate_content"] is True

            def readinto(handle: Any) -> int:
                if self.fail:
                    handle.write(payload[:100])
                    raise ServiceResponseTimeoutError("read timed out")
                return handle.write(payload)

            return SimpleNamespace(readinto=readinto)

        def close(self) -> None:
            return None

    def from_blob_url(url: str) -> _FakeBlobClient:
        attempts.append(url)
        return _FakeBlobClient(fail=len(attempts) == 1)

    monkeypatch.setattr(
        "dl_azure.datasets.tar_shard.BlobClient.from_blob_url",
        from_blob_url,
    )
    wrapper = _DynamicStreamingTarWrapper(
        {
            "account_name": "demo",
            "container_name": "datasets",
            "azure_config_path": str(tmp_path / "missing-azure-config.json"),
            "dynamic_shards": {"train": ["train/demo.tar"]},
            "required_extensions": ["png", "json"],
            "cache": {
                "cache_dir": str(tmp_path / "cache"),
                "cache_size_gb": 2,
                "download_retries": 2,
                "retry_backoff_seconds": 0,
                "retry_jitter": False,
                "connection_timeout_seconds": 7,
                "read_timeout_seconds": 11,
            },
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "auto_split": False,
        }
    )

    loader = wrapper.get_split("train")
    assert loader is not None
    batch = next(iter(loader))
    assert batch["key"] == ["sample"]
    assert attempts == [shard_url, shard_url]
    assert len(list((tmp_path / "cache").glob("*.tar"))) == 1
    assert not list((tmp_path / ".cache.parts").glob("*.part"))

    next(iter(loader))
    assert attempts == [shard_url, shard_url]


def test_streaming_tar_cache_size_is_configured_in_gb(tmp_path: Path) -> None:
    common = {
        "account_name": "demo",
        "container_name": "datasets",
        "azure_config_path": str(tmp_path / "missing-azure-config.json"),
        "dynamic_shards": {"train": ["train/demo.tar"]},
        "auto_split": False,
    }
    wrapper = _DynamicStreamingTarWrapper(
        {
            **common,
            "cache": {"enabled": True, "cache_size_gb": 1.5},
        }
    )
    assert wrapper.webdataset_config["cache_size"] == int(1.5 * 1024**3)

    with pytest.raises(ValueError, match="replace cache_size with cache_size_gb"):
        _DynamicStreamingTarWrapper(
            {
                **common,
                "cache": {"enabled": True, "cache_size": 1000},
            }
        )
