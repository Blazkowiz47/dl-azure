"""Reusable Azure dataset foundations for dl-azure."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dl_core.core.base_dataset import BaseWrapper, FrameWrapper
from dl_core.utils import crop_face_with_bbox

from dl_azure.storage import AzureBlobCache, AzureClientService

logger = logging.getLogger(__name__)


def sort_frame_paths(frame_paths: list[str]) -> list[str]:
    """Sort frame paths numerically while tolerating inconsistent names."""

    valid_frames: list[tuple[int | float, str]] = []
    for path in frame_paths:
        filename = Path(path).name
        frame_name = Path(filename).stem
        frame_name = frame_name.removeprefix("frames_")
        frame_name = frame_name.removeprefix("frame_")
        frame_name = frame_name.removeprefix("frame")
        frame_name = frame_name.removeprefix("frames")
        try:
            frame_num = int(frame_name)
            valid_frames.append((frame_num, path))
            continue
        except ValueError:
            pass

        match = re.search(r"\d+", frame_name)
        if match:
            valid_frames.append((int(match.group()), path))
            continue

        valid_frames.append((float("inf"), path))

    valid_frames.sort(key=lambda item: item[0])
    return [path for _, path in valid_frames]


class AzureComputeMixin(ABC):
    """Shared helpers for Azure ML mounted datasets."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.input_name: str = self.config.get("input_name", "dataset_path")
        self.allow_local_fallback: bool = self.config.get(
            "allow_local_fallback", True
        )
        self.local_fallback_root: str = self.config.get("local_fallback_root", ".")
        self.root_dir = self._resolve_compute_root_dir(self.config.get("root_dir"))

    def _resolve_compute_root_dir(self, configured_root: str | None) -> Path:
        """Resolve the dataset root from explicit config or Azure ML mounts."""

        mount_path = os.environ.get(f"AZURE_ML_INPUT_{self.input_name}")
        if configured_root:
            root_dir = Path(configured_root).expanduser()
            if root_dir.is_absolute():
                return root_dir
            if mount_path:
                return Path(mount_path) / root_dir
            return root_dir

        if mount_path:
            return Path(mount_path)

        if self.allow_local_fallback:
            return Path(self.local_fallback_root).expanduser()

        raise FileNotFoundError(
            "Could not resolve Azure compute dataset root. "
            f"Set dataset.root_dir or AZURE_ML_INPUT_{self.input_name}."
        )

    def resolve_path(self, relative_path: str | Path) -> Path:
        """Resolve a dataset-relative path against the configured root."""

        path = Path(relative_path)
        if path.is_absolute():
            return path

        if path.parts and path.parts[0] == self.root_dir.name:
            return self.root_dir.parent / path
        return self.root_dir / path

    def scan_paths(
        self,
        prefix: str | Path,
        *,
        extension: str | None = None,
        recursive: bool = True,
    ) -> list[str]:
        """Scan a local directory and return dataset-relative file paths."""

        target_dir = self.resolve_path(prefix)
        if not target_dir.exists():
            self.logger.warning(f"Scan path does not exist: {target_dir}")
            return []

        pattern = f"*.{extension}" if extension else "*"
        iterator = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
        results = [
            str(path.relative_to(self.root_dir).as_posix())
            for path in iterator
            if path.is_file()
        ]
        results.sort()
        return results

    def load_json_data(
        self, relative_path: str, use_cache: bool = False
    ) -> dict[str, Any]:
        """Load JSON metadata from the mounted dataset root."""

        del use_cache
        with open(self.resolve_path(relative_path), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_image_data(
        self, relative_path: str, use_cache: bool = False
    ) -> np.ndarray | None:
        """Load an RGB image from the mounted dataset root."""

        del use_cache
        image = cv2.imread(str(self.resolve_path(relative_path)), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def should_use_cache(self, split: str) -> bool:
        """Return whether caching should be enabled for the given split."""

        del split
        return False


class AzureStreamingMixin(ABC):
    """Shared helpers for Azure blob-backed datasets."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.azure_config_path = Path(
            self.config.get("azure_config_path", "azure-config.json")
        ).expanduser()
        self.azure_config = self._load_azure_config()
        self.container_name = self.config.get("container_name")
        if not self.container_name:
            raise ValueError(
                "Azure streaming datasets require 'container_name' in dataset "
                "config."
            )
        self.azure_service = AzureClientService(self.azure_config)

        cache_config = self.config.get("cache", {})
        cache_dir = Path(
            cache_config.get("cache_dir", "~/.cache/dl-azure")
        ).expanduser()
        self.cache: AzureBlobCache | None = None
        if cache_config.get("enabled", True):
            self.cache = AzureBlobCache(str(cache_dir))
        self.cache_splits = set(
            cache_config.get("cache_splits", ["train", "validation", "test"])
        )

    def _load_azure_config(self) -> dict[str, Any]:
        """Load Azure storage configuration from file and dataset config."""

        azure_config: dict[str, Any] = {}
        if self.azure_config_path.exists():
            with open(self.azure_config_path, "r", encoding="utf-8") as handle:
                azure_config.update(json.load(handle))

        for key in [
            "account_name",
            "subscription_id",
            "resource_group",
            "workspace_name",
            "tenant_id",
        ]:
            if key in self.config:
                azure_config[key] = self.config[key]

        if "account_name" not in azure_config:
            raise ValueError(
                "Azure streaming datasets require 'account_name' in dataset config "
                "or azure-config.json."
            )

        return azure_config

    def scan_paths(
        self,
        prefix: str | Path,
        *,
        extension: str | None = None,
        recursive: bool = True,
    ) -> list[str]:
        """List blob paths under a prefix, optionally filtering by extension."""

        del recursive
        normalized_prefix = str(prefix).strip("/")
        container_client = self.azure_service.get_container_client(self.container_name)
        blobs = container_client.list_blob_names(name_starts_with=normalized_prefix)
        results: list[str] = []
        for blob_name in blobs:
            if extension and Path(blob_name).suffix.lstrip(".") != extension:
                continue
            results.append(str(blob_name))
        results.sort()
        return results

    def load_json_data(
        self, relative_path: str, use_cache: bool = False
    ) -> dict[str, Any]:
        """Load JSON metadata from Azure blob storage."""

        if use_cache and self.cache is not None:
            cached_json = self.cache.get_cached_json(relative_path)
            if cached_json is not None:
                return cached_json

        blob_client = self.azure_service.get_blob_client_pooled(
            self.container_name, relative_path
        )
        data = json.loads(blob_client.download_blob().readall())
        if use_cache and self.cache is not None:
            self.cache.cache_json_async(relative_path, data)
        return data

    def load_image_data(
        self, relative_path: str, use_cache: bool = False
    ) -> np.ndarray | None:
        """Load an RGB image from Azure blob storage."""

        if use_cache and self.cache is not None:
            cached_image = self.cache.get_cached_image(relative_path)
            if cached_image is not None:
                return cached_image

        blob_client = self.azure_service.get_blob_client_pooled(
            self.container_name, relative_path
        )
        with BytesIO(blob_client.download_blob().readall()) as buffer:
            nparr = np.frombuffer(buffer.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if use_cache and self.cache is not None:
            self.cache.cache_image_async(relative_path, image)
        return image

    def should_use_cache(self, split: str) -> bool:
        """Return whether caching should be enabled for the given split."""

        return split in self.cache_splits


class AzureComputeWrapper(AzureComputeMixin, BaseWrapper):
    """Generic Azure ML mounted dataset wrapper."""

    @property
    def file_extensions(self) -> list[str]:
        """Supported image file extensions for generic Azure datasets."""

        return ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


class AzureStreamingWrapper(AzureStreamingMixin, BaseWrapper):
    """Generic Azure blob-backed dataset wrapper."""

    @property
    def file_extensions(self) -> list[str]:
        """Supported image file extensions for generic Azure datasets."""

        return ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


class AzureFrameMixin(ABC):
    """Shared frame-level helpers for Azure dataset wrappers."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.height = self.config.get("height", 224)
        self.width = self.config.get("width", 224)
        self.resize_height = self.config.get("resize_height", self.height)
        self.resize_width = self.config.get("resize_width", self.width)
        self.use_face_detection = self.config.get("use_face_detection", True)
        self.face_detected_and_resized_cache = self.config.get(
            "face_detected_and_resized_cache", False
        )
        self.margin = self._normalize_margin(self.config.get("margin", (0, 0)))

    @property
    def file_extensions(self) -> list[str]:
        """Supported image file extensions."""

        return ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    def _normalize_margin(self, margin: Any) -> tuple[int, int]:
        """Normalize face crop margin config to a bounded tuple."""

        if isinstance(margin, (int, float)):
            normalized = (int(margin), int(margin))
        elif isinstance(margin, dict):
            normalized = (margin.get("height", 0), margin.get("width", 0))
        elif isinstance(margin, (tuple, list)) and len(margin) == 2:
            normalized = (int(margin[0]), int(margin[1]))
        else:
            self.logger.warning(f"Invalid margin format: {margin}. Using (0, 0)")
            normalized = (0, 0)
        return tuple(max(0, min(100, value)) for value in normalized)

    def metadata_path_for_image(self, image_path: str) -> str:
        """Map a frame path to its associated metadata path."""

        metadata_path = image_path.replace("Raw_Frames", "data/metadata")
        metadata_path = metadata_path.replace("data/frames", "data/metadata")
        return f"{Path(metadata_path).with_suffix('')}.json"

    def _maybe_load_resized_cache(self, image_path: str) -> np.ndarray | None:
        """Load a cached resized frame if that cache is enabled."""

        if not self.face_detected_and_resized_cache:
            return None
        if not hasattr(self, "cache") or self.cache is None:
            return None

        cache_prefix = "face_detected_image" if self.use_face_detection else (
            "resized_image"
        )
        return self.cache.get_cached_image(f"{cache_prefix}/{image_path}")

    def _maybe_store_resized_cache(self, image_path: str, image: np.ndarray) -> None:
        """Persist a resized frame if that cache is enabled."""

        if not self.face_detected_and_resized_cache:
            return
        if not hasattr(self, "cache") or self.cache is None:
            return

        cache_prefix = "face_detected_image" if self.use_face_detection else (
            "resized_image"
        )
        self.cache.cache_image_async(f"{cache_prefix}/{image_path}", image)

    def _load_image_and_face_detect(
        self, image_path: str, *, use_cache: bool
    ) -> tuple[np.ndarray | None, list[float] | None, bool]:
        """Load a frame and optionally crop the detected face region."""

        if use_cache:
            cached_image = self._maybe_load_resized_cache(image_path)
            if cached_image is not None:
                return (
                    cached_image,
                    [0, 0, self.resize_width, self.resize_height],
                    True,
                )

        image = self.load_image_data(image_path, use_cache=use_cache)
        if image is None:
            return None, None, False

        if not self.use_face_detection:
            if self.resize_width > 0 and self.resize_height > 0:
                image = cv2.resize(image, (self.resize_width, self.resize_height))
                if use_cache:
                    self._maybe_store_resized_cache(image_path, image)
            return image, [0, 0, image.shape[1], image.shape[0]], False

        metadata = self.load_json_data(
            self.metadata_path_for_image(image_path), use_cache=use_cache
        )
        if not metadata or "bboxes" not in metadata:
            return image, None, False

        face_image, _ = crop_face_with_bbox(image, metadata["bboxes"], self.margin)
        if face_image is None or face_image.size == 0:
            return image, None, False

        if self.resize_width > 0 and self.resize_height > 0:
            face_image = cv2.resize(face_image, (self.resize_width, self.resize_height))
            if use_cache:
                self._maybe_store_resized_cache(image_path, face_image)
        return face_image, [0, 0, face_image.shape[1], face_image.shape[0]], True

    def empty_frame_tensor(self) -> torch.Tensor:
        """Return a zero tensor matching the configured frame shape."""

        return torch.zeros(3, self.height, self.width)

    def load_frame_tensor(
        self, image_path: str, split: str
    ) -> tuple[torch.Tensor, list[float] | None, bool]:
        """Load a single frame and convert it into a tensor."""

        image, bbox, has_metadata = self._load_image_and_face_detect(
            image_path,
            use_cache=self.should_use_cache(split),
        )
        if image is None or image.size == 0:
            return self.empty_frame_tensor(), [0, 0, self.width, self.height], False

        if self.augmentation:
            tensor = self.augmentation.apply(image, split)
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        del image
        return tensor.float(), bbox, has_metadata


class AzureComputeFrameWrapper(AzureFrameMixin, AzureComputeMixin, FrameWrapper):
    """Azure ML mounted frame dataset wrapper."""


class AzureStreamingFrameWrapper(
    AzureFrameMixin, AzureStreamingMixin, FrameWrapper
):
    """Azure blob-backed frame dataset wrapper."""


class AzureMultiFrameMixin(ABC):
    """Shared multiframe logic for Azure frame datasets."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        multiframe_config = self.config.get("multiframe", {})
        self.mode = multiframe_config.get("mode", "random")
        self.num_frames = multiframe_config.get("num_frames", 5)
        self.frame_stride = multiframe_config.get("frame_stride", 0)

        if self.mode not in {"random", "consecutive"}:
            raise ValueError(
                f"Invalid multiframe mode: {self.mode}. "
                "Must be 'random' or 'consecutive'."
            )

    def sample_frames(
        self, frames: list[str], max_frames: int, split: str
    ) -> list[str]:
        """Keep all frames for multiframe datasets while sorting them."""

        del max_frames, split
        return sort_frame_paths(frames)

    @abstractmethod
    def build_frame_record(
        self, frame_path: str, dataset_name: str, video_id: str
    ) -> dict[str, Any]:
        """Build the metadata dictionary for one frame record."""

    def convert_groups_to_files(
        self,
        video_groups: dict[str, dict[str, list[str]]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Convert grouped videos into multiframe samples."""

        files: list[dict[str, Any]] = []
        skipped_videos = 0
        rng = random.Random(self.seed + self.current_epoch)

        for dataset_name, videos in video_groups.items():
            for video_id, frames in videos.items():
                sorted_frames = sort_frame_paths(frames)
                if len(sorted_frames) < self.num_frames:
                    skipped_videos += 1
                    continue

                base_record = self.build_frame_record(
                    sorted_frames[0], dataset_name, video_id
                )
                common_fields = {
                    key: value
                    for key, value in base_record.items()
                    if key not in {"path", "paths"}
                }

                if self.mode == "random":
                    num_samples = len(sorted_frames) // self.num_frames
                    for _ in range(num_samples):
                        selected_frames = tuple(
                            rng.sample(sorted_frames, self.num_frames)
                        )
                        files.append(
                            {
                                **common_fields,
                                "path": selected_frames[0],
                                "paths": selected_frames,
                            }
                        )
                    continue

                stride = self.num_frames + self.frame_stride
                for start_idx in range(
                    0, len(sorted_frames) - self.num_frames + 1, stride
                ):
                    selected_frames = tuple(
                        sorted_frames[start_idx : start_idx + self.num_frames]
                    )
                    files.append(
                        {
                            **common_fields,
                            "path": selected_frames[0],
                            "paths": selected_frames,
                        }
                    )

        self.logger.info(
            f"Created {len(files)} multiframe samples for {split} "
            f"(skipped {skipped_videos} videos with < {self.num_frames} frames)"
        )
        return files

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Load and stack multiple frames into one sample tensor."""

        frame_tensors: list[torch.Tensor] = []
        valid_paths: list[str] = []
        for path in file_dict["paths"]:
            tensor, _, _ = self.load_frame_tensor(path, split)
            frame_tensors.append(tensor)
            valid_paths.append(path)

        if frame_tensors:
            stacked_frames = torch.stack(frame_tensors, dim=0)
        else:
            stacked_frames = torch.zeros(self.num_frames, 3, self.height, self.width)

        return {
            **{
                key: value
                for key, value in file_dict.items()
                if key not in {"path", "paths"}
            },
            "image": stacked_frames,
            "paths": tuple(valid_paths),
        }


class AzureComputeMultiFrameWrapper(
    AzureMultiFrameMixin, AzureComputeFrameWrapper
):
    """Azure ML mounted multiframe dataset wrapper."""


class AzureStreamingMultiFrameWrapper(
    AzureMultiFrameMixin, AzureStreamingFrameWrapper
):
    """Azure blob-backed multiframe dataset wrapper."""
