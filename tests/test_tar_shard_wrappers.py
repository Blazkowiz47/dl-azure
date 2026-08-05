"""Tests for mounted and blob-cached Azure tar shard wrappers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dl_core.datasets import TarShardIndex

from dl_azure.datasets import (
    AzureComputeTarShardWrapper,
    AzureStreamingTarShardWrapper,
)
from dl_azure.storage import AzureShardCache


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


class _FakeAzureService:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs
        self.downloads: list[str] = []

    def get_blob_client_pooled(self, container_name: str, blob_path: str) -> Any:
        del container_name
        payload = self.blobs[blob_path]
        return SimpleNamespace(
            get_blob_properties=lambda: SimpleNamespace(
                etag=f'"{len(payload)}"',
                size=len(payload),
                version_id=None,
            )
        )

    def download_blob(
        self, container_name: str, blob_path: str, local_path: Path
    ) -> bool:
        del container_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.blobs[blob_path])
        self.downloads.append(blob_path)
        return True

    def blob_exists(self, container_name: str, blob_path: str) -> bool:
        del container_name
        return blob_path in self.blobs


def test_compute_tar_wrapper_reads_from_resolved_mount(tmp_path: Path) -> None:
    tar_path = tmp_path / "train" / "demo.tar"
    tar_path.parent.mkdir()
    _write_tar(tar_path)
    TarShardIndex.build(tar_path).write(f"{tar_path}.idx.json")
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


def test_shard_cache_reuses_etag_validated_file(tmp_path: Path) -> None:
    service = _FakeAzureService({"train/demo.tar": b"first"})
    cache = AzureShardCache(tmp_path / "cache")

    first = cache.materialize(service, "datasets", "train/demo.tar")
    second = cache.materialize(service, "datasets", "train/demo.tar")

    assert first == second
    assert first.read_bytes() == b"first"
    assert service.downloads == ["train/demo.tar"]

    service.blobs["train/demo.tar"] = b"second-version"
    refreshed = cache.materialize(service, "datasets", "train/demo.tar")
    assert refreshed.read_bytes() == b"second-version"
    assert service.downloads == ["train/demo.tar", "train/demo.tar"]


def test_streaming_tar_wrapper_caches_tar_and_remote_index(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tar_path = tmp_path / "source.tar"
    index_path = tmp_path / "source.tar.idx.json"
    _write_tar(tar_path)
    TarShardIndex.build(tar_path).write(index_path)
    service = _FakeAzureService(
        {
            "train/demo.tar": tar_path.read_bytes(),
            "train/demo.tar.idx.json": index_path.read_bytes(),
        }
    )
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService",
        lambda config: service,
    )
    wrapper = _StreamingTarWrapper(
        {
            "account_name": "demo",
            "container_name": "datasets",
            "azure_config_path": str(tmp_path / "missing-azure-config.json"),
            "shards": {"train": [{"path": "train/demo.tar", "group": "attack"}]},
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
    assert service.downloads == [
        "train/demo.tar",
        "train/demo.tar.idx.json",
    ]
