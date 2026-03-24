"""Tests for the reusable Azure dataset foundations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import dl_core
from dl_azure.datasets.base import (
    AzureComputeMultiFrameWrapper,
    AzureComputeWrapper,
    sort_frame_paths,
)
from dl_azure.datasets.pad import AzureComputeMultiframePADWrapper


class DummyComputeWrapper(AzureComputeWrapper):
    """Minimal concrete wrapper for compute foundation tests."""

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        return []

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        del split
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


def _write_dummy_pad_metadata(
    root: Path,
    split: str,
    metadata_suffix: str,
    blob_paths: list[str],
) -> None:
    """Write one PAD metadata JSON file and its backing frame images."""

    metadata_path = root / "data" / "paths" / split / metadata_suffix
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({"blobs": blob_paths}, indent=2),
        encoding="utf-8",
    )

    for index, blob_path in enumerate(blob_paths, start=1):
        frame_path = root / blob_path.replace("Raw_Frames", "data/frames")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        image[..., 0] = (index * 30) % 255
        image[..., 1] = (index * 50) % 255
        image[..., 2] = (index * 70) % 255
        cv2.imwrite(str(frame_path), image)


def test_compute_multiframe_pad_wrapper_stacks_frames(tmp_path: Path) -> None:
    """The PAD multiframe wrapper should emit stacked frames and preserve paths."""

    dl_core.load_builtin_components()
    dataset_root = tmp_path / "pad_dataset"

    _write_dummy_pad_metadata(
        dataset_root,
        "Train",
        "Attack/2D/Print/DemoSet.json",
        [
            "Raw_Frames/Train/Attack/2D/Print/DemoSet/attack_vid/frame_0001.png",
            "Raw_Frames/Train/Attack/2D/Print/DemoSet/attack_vid/frame_0002.png",
            "Raw_Frames/Train/Attack/2D/Print/DemoSet/attack_vid/frame_0003.png",
            "Raw_Frames/Train/Attack/2D/Print/DemoSet/attack_vid/frame_0004.png",
        ],
    )
    _write_dummy_pad_metadata(
        dataset_root,
        "Train",
        "Bonafide/DemoSet.json",
        [
            "Raw_Frames/Train/Bonafide/DemoSet/real_vid/frame_0001.png",
            "Raw_Frames/Train/Bonafide/DemoSet/real_vid/frame_0002.png",
            "Raw_Frames/Train/Bonafide/DemoSet/real_vid/frame_0003.png",
            "Raw_Frames/Train/Bonafide/DemoSet/real_vid/frame_0004.png",
        ],
    )

    wrapper = AzureComputeMultiframePADWrapper(
        {
            "root_dir": str(dataset_root),
            "allow_local_fallback": False,
            "classes": ["real", "attack"],
            "num_classes": 2,
            "batch_size": 1,
            "num_workers": 0,
            "height": 32,
            "width": 32,
            "metadata_dir": "data/paths",
            "use_face_detection": False,
            "multiframe": {
                "num_frames": 2,
                "mode": "consecutive",
                "frame_stride": 0,
            },
        }
    )

    train_loader = wrapper.get_split("train")
    assert train_loader is not None
    batch = next(iter(train_loader))

    assert isinstance(batch["image"], torch.Tensor)
    assert tuple(batch["image"].shape) == (1, 2, 3, 32, 32)
    assert "paths" in batch
    assert len(batch["paths"][0]) == 2
