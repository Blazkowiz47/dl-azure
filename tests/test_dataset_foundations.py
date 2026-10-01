"""Tests for the reusable Azure dataset foundations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dl_azure.datasets.base import (
    AzureComputeMultiFrameWrapper,
    AzureComputeWrapper,
    AzureStreamingWrapper,
    sort_frame_paths,
)


class DummyComputeWrapper(AzureComputeWrapper):
    """Minimal concrete wrapper for compute foundation tests."""

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        return []

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        del split
        return file_dict


class DummyStreamingWrapper(AzureStreamingWrapper):
    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        return []

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return file_dict


class DummyMultiFrameWrapper(AzureComputeMultiFrameWrapper):
    """Minimal concrete multiframe wrapper for foundation tests."""

    def get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        del split
        return {}

    def build_frame_record(
        self, frame_path: str, dataset_name: str, video_id: str
    ) -> dict[str, Any]:
        return {
            "path": frame_path,
            "label": 1,
            "dataset": dataset_name,
            "video_id": video_id,
            "attack_type": "print",
            "attack_dimension": "2D",
        }


def test_compute_wrapper_uses_explicit_root_dir(tmp_path: Path) -> None:
    """The compute foundation should honour an explicit dataset root."""

    dataset_root = tmp_path / "dataset"
    metadata_dir = dataset_root / "data" / "paths" / "Train" / "Attack"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "demo.json"
    metadata_file.write_text("{}", encoding="utf-8")

    wrapper = DummyComputeWrapper(
        {
            "root_dir": str(dataset_root),
            "allow_local_fallback": False,
        }
    )

    assert wrapper.resolve_path("data/paths/Train/Attack/demo.json") == metadata_file
    assert wrapper.scan_paths("data/paths/Train", extension="json") == [
        "data/paths/Train/Attack/demo.json"
    ]


def test_compute_wrapper_uses_mount_env_for_relative_root(tmp_path: Path) -> None:
    """The compute foundation should resolve a relative root under the Azure mount."""

    mount_root = tmp_path / "mount"
    data_root = mount_root / "data"
    data_root.mkdir(parents=True)

    old_mount = os.environ.get("AZURE_ML_INPUT_dataset_path")
    os.environ["AZURE_ML_INPUT_dataset_path"] = str(mount_root)
    try:
        wrapper = DummyComputeWrapper(
            {
                "root_dir": "data",
                "allow_local_fallback": False,
            }
        )
    finally:
        if old_mount is None:
            os.environ.pop("AZURE_ML_INPUT_dataset_path", None)
        else:
            os.environ["AZURE_ML_INPUT_dataset_path"] = old_mount

    assert wrapper.root_dir == data_root
    assert wrapper.resolve_path("data/frames/Test/frame_001.png") == (
        data_root / "frames" / "Test" / "frame_001.png"
    )


def test_sort_frame_paths_tolerates_mixed_names() -> None:
    """Frame sorting should tolerate common naming variants."""

    frames = [
        "frames/frame_010.png",
        "frames/frame2.png",
        "frames/frames_001.png",
        "frames/alpha.png",
    ]

    assert sort_frame_paths(frames) == [
        "frames/frames_001.png",
        "frames/frame2.png",
        "frames/frame_010.png",
        "frames/alpha.png",
    ]


def test_multiframe_wrapper_builds_consecutive_samples() -> None:
    """The multiframe foundation should build grouped consecutive samples."""

    wrapper = DummyMultiFrameWrapper(
        {
            "root_dir": ".",
            "allow_local_fallback": True,
            "multiframe": {
                "mode": "consecutive",
                "num_frames": 2,
                "frame_stride": 1,
            },
        }
    )

    files = wrapper.convert_groups_to_files(
        {
            "demo": {
                "video-1": [
                    "frames/frame_001.png",
                    "frames/frame_002.png",
                    "frames/frame_003.png",
                    "frames/frame_004.png",
                    "frames/frame_005.png",
                ]
            }
        },
        "train",
    )

    assert len(files) == 2
    assert files[0]["paths"] == (
        "frames/frame_001.png",
        "frames/frame_002.png",
    )
    assert files[1]["paths"] == (
        "frames/frame_004.png",
        "frames/frame_005.png",
    )


def test_processed_frame_cache_isolated_by_margin_and_size() -> None:
    """Changing crop settings must not reuse images from another configuration."""
    images: dict[str, np.ndarray] = {}
    cache = SimpleNamespace(
        get_cached_image=images.get,
        cache_image_async=lambda key, image: images.__setitem__(key, image),
    )
    wrappers = [
        DummyMultiFrameWrapper({
            "root_dir": ".",
            "face_detected_and_resized_cache": True,
            "margin": margin,
            "resize_height": size,
            "resize_width": size,
        })
        for margin, size in [(0, 64), (25, 64), (25, 128)]
    ]
    for wrapper in wrappers:
        wrapper.cache = cache

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    wrappers[0]._maybe_store_resized_cache("frame.jpg", image)
    assert wrappers[0]._maybe_load_resized_cache("frame.jpg") is image
    assert wrappers[1]._maybe_load_resized_cache("frame.jpg") is None
    wrappers[1]._maybe_store_resized_cache("frame.jpg", image)
    assert wrappers[2]._maybe_load_resized_cache("frame.jpg") is None
    assert len(images) == 2


def test_nested_azure_config_merges_defaults_legacy_and_dataset_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "azure.json"
    config_file.write_text(json.dumps({"azure": {
        "account_name": "project-account", "container_name": "project-container",
        "download": {"backend": "sdk", "sdk": {
            "max_concurrency": 8, "read_buffer_kib": 64,
        }},
        "cache": {"enabled": False, "cache_splits": ["test"]},
    }}))
    captured = []
    monkeypatch.setattr("dl_azure.datasets.base.AzureClientService", lambda config: captured.append(config))
    wrapper = DummyStreamingWrapper({
        "account_name": "legacy-account", "container_name": "legacy-container",
        "azure_config_path": str(tmp_path / "wrong.json"),
        "cache": {"cache_splits": ["train"]},
        "azure": {
            "config_path": str(config_file), "account_name": "dataset-account",
            "download": {"sdk": {"max_concurrency": 32}},
            "cache": {"cache_dir": str(tmp_path / "cache")},
        },
    })
    assert wrapper.azure_config_path == config_file
    assert wrapper.container_name == "legacy-container"
    assert wrapper.cache is None and wrapper.cache_splits == {"train"}
    assert captured[0]["account_name"] == "dataset-account"
    assert captured[0]["download"] == {"backend": "sdk", "sdk": {
        "max_concurrency": 32, "read_buffer_kib": 64,
    }}
    assert captured[0]["cache"]["cache_dir"] == str(tmp_path / "cache")


def test_images_and_json_use_configured_sdk_without_azcopy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    from dl_azure.storage.download import AzureDownloader

    calls = []
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    payloads = {"image.png": encoded, "metadata.json": b'{"label": 1}'}
    downloader = AzureDownloader({"sdk": {"max_concurrency": 8}})

    def get_blob_client(container: str, path: str) -> Any:
        def download_blob(**kwargs: Any) -> Any:
            calls.append((path, kwargs))
            return SimpleNamespace(readall=lambda: payloads[path])
        return SimpleNamespace(download_blob=download_blob)

    service = SimpleNamespace(downloader=downloader, get_blob_client_pooled=get_blob_client)
    monkeypatch.setattr("dl_azure.datasets.base.AzureClientService", lambda config: service)
    monkeypatch.setattr(
        "dl_azure.storage.download.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Memory reads must not invoke AzCopy"),
    )
    wrapper = DummyStreamingWrapper({"azure": {
        "account_name": "demo", "container_name": "data",
        "config_path": str(tmp_path / "missing.json"), "cache": {"enabled": False},
    }})
    assert wrapper.load_json_data("metadata.json") == {"label": 1}
    assert np.array_equal(wrapper.load_image_data("image.png"), image)
    assert [path for path, _ in calls] == ["metadata.json", "image.png"]
    assert all(options["max_concurrency"] == 8 for _, options in calls)
