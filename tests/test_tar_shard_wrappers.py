"""Tests for Azure WebDataset shard path wrappers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

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
    def __init__(self, paths: dict[str, Path]) -> None:
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
