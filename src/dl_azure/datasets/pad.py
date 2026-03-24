"""PAD-specific Azure dataset wrappers built on the reusable foundations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from dl_core.core.registry import register_dataset

from dl_azure.datasets.base import (
    AzureComputeFrameWrapper,
    AzureComputeMultiFrameWrapper,
    AzureStreamingFrameWrapper,
    AzureStreamingMultiFrameWrapper,
    sort_frame_paths,
)


class AzurePADMixin:
    """Common PAD dataset scanning, filtering, and transform helpers."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        if not self.classes:
            self.classes = ["real", "attack"]

        self.attack_label = self.classes.index("attack")
        self.real_label = self.classes.index("real")
        self.allowed_datasets = self.config.get("allowed_datasets")
        self.allowed_attacks = self.config.get("allowed_attacks")
        self.allowed_dimensions = self.config.get("allowed_dimensions")
        self.skipped_datasets = self.config.get("skipped_datasets")
        self.metadata_dir = self.config.get("metadata_dir", "data/paths")
        self.metadata_paths: dict[
            str, dict[str, list[str] | dict[str, dict[str, list[str]]]]
        ] = {
            "train": {"attack": {}, "real": []},
            "validation": {"attack": {}, "real": []},
            "test": {"attack": {}, "real": []},
        }

    def _clean_frame_path(self, blob_path: str) -> str:
        """Normalize raw PAD frame paths into dataset-relative frame paths."""

        return blob_path.replace("Raw_Frames/Train", "data/frames/Train").replace(
            "Raw_Frames/Test", "data/frames/Test"
        )

    def _parse_path(self, path: str) -> dict[str, Any]:
        """Parse a PAD metadata or frame path into structured metadata."""

        if "/Train/" in path:
            is_train = True
            parts = path.split("/Train/")[1].split("/")
        elif "/Test/" in path:
            is_train = False
            parts = path.split("/Test/")[1].split("/")
        elif path.startswith("Train/"):
            is_train = True
            parts = path.split("Train/")[1].split("/")
        elif path.startswith("Test/"):
            is_train = False
            parts = path.split("Test/")[1].split("/")
        else:
            raise ValueError(f"Path does not contain Train or Test: {path}")

        is_attack = parts[0] == "Attack"
        if is_attack:
            return {
                "is_train": is_train,
                "is_attack": True,
                "attack_type": parts[2].lower(),
                "attack_dimension": parts[1],
                "dataset_name": parts[3],
            }
        return {
            "is_train": is_train,
            "is_attack": False,
            "attack_type": "real",
            "attack_dimension": "NA",
            "dataset_name": parts[1],
        }

    def _extract_video_id(self, file_path: str) -> str:
        """Extract the PAD video identifier from a frame path."""

        split_part = "/Train/" if "/Train/" in file_path else "/Test/"
        return os.path.split(file_path.split(split_part)[1])[0]

    def _should_include_dataset(self, dataset: str, split: str) -> bool:
        """Check whether a dataset passes the configured allow/skip filters."""

        if self.skipped_datasets:
            if isinstance(self.skipped_datasets, dict):
                split_skips = self.skipped_datasets.get(split, [])
                if dataset in split_skips:
                    return False
            elif dataset in self.skipped_datasets:
                return False

        if not self.allowed_datasets:
            return True

        if isinstance(self.allowed_datasets, dict):
            split_allowed = self.allowed_datasets.get(split)
            if split_allowed is None:
                return True
            return dataset in split_allowed

        return dataset in self.allowed_datasets

    def _should_include_attack(
        self, attack_type: str, attack_dimension: str, split: str
    ) -> bool:
        """Check whether an attack passes the configured allow filters."""

        attack_allowed = True
        dimension_allowed = True

        if self.allowed_dimensions is not None:
            if isinstance(self.allowed_dimensions, dict):
                split_dimensions = self.allowed_dimensions.get(split)
                if split_dimensions is not None:
                    dimension_allowed = attack_dimension in split_dimensions
            else:
                dimension_allowed = attack_dimension in self.allowed_dimensions

        if self.allowed_attacks is not None:
            if isinstance(self.allowed_attacks, dict):
                split_attacks = self.allowed_attacks.get(split)
                if split_attacks is not None:
                    attack_allowed = attack_type in split_attacks
            else:
                attack_allowed = attack_type in self.allowed_attacks

        return attack_allowed and dimension_allowed

    def _clean_paths_json(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Convert PAD metadata blobs into grouped frame lists."""

        paths = [self._clean_frame_path(blob) for blob in data.get("blobs", [])]
        video_grouped: dict[str, list[str]] = {}
        for path in paths:
            if Path(path).stem.lower().endswith("_depth"):
                continue
            video_id = self._extract_video_id(path)
            video_grouped.setdefault(video_id, []).append(path)

        for video_id, frame_paths in video_grouped.items():
            video_grouped[video_id] = sort_frame_paths(frame_paths)
        return video_grouped

    def _get_paths_metadata(self, split: str) -> None:
        """Discover PAD metadata JSON files for the requested split."""

        if self.metadata_paths[split]["real"] or self.metadata_paths[split]["attack"]:
            return

        metadata_prefix = f"{self.metadata_dir}/{split.capitalize()}"
        metadata_paths = self.scan_paths(metadata_prefix, extension="json")
        for path in metadata_paths:
            reminder_url = path.replace(f"{self.metadata_dir}/", "", 1)
            if not reminder_url.startswith(("Train/", "Test/")):
                reminder_url = f"{split.capitalize()}/{reminder_url}"

            path_info = self._parse_path(reminder_url)
            if path_info["is_attack"]:
                attack_dimension = str(path_info["attack_dimension"])
                attack_type = str(path_info["attack_type"])
                attacks = self.metadata_paths[split]["attack"]
                if isinstance(attacks, list):
                    raise ValueError("Expected attack metadata structure to be a dict.")
                attacks.setdefault(attack_dimension, {}).setdefault(attack_type, [])
                attacks[attack_dimension][attack_type].append(path)
                continue

            reals = self.metadata_paths[split]["real"]
            if isinstance(reals, dict):
                raise ValueError("Expected real metadata structure to be a list.")
            reals.append(path)

    def build_frame_record(
        self, frame_path: str, dataset_name: str, video_id: str
    ) -> dict[str, Any]:
        """Build the file metadata dictionary for a single PAD frame."""

        path_info = self._parse_path(frame_path)
        is_attack = path_info["is_attack"]
        return {
            "path": frame_path,
            "label": self.attack_label if is_attack else self.real_label,
            "video_id": video_id,
            "dataset": dataset_name,
            "attack_type": path_info["attack_type"],
            "attack_dimension": path_info["attack_dimension"],
        }

    def get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        """Collect PAD video groups for the requested split."""

        if split == "validation":
            self.logger.info("No dedicated validation split in PAD metadata.")
            return {}

        self._get_paths_metadata(split)
        video_groups: dict[str, dict[str, list[str]]] = {}

        attacks = self.metadata_paths[split]["attack"]
        if isinstance(attacks, list):
            raise ValueError("Expected attacks metadata structure to be a dict.")

        for attack_dimension, attack_map in attacks.items():
            for attack_type, paths in attack_map.items():
                if not self._should_include_attack(
                    attack_type, attack_dimension, split
                ):
                    continue
                for path in paths:
                    dataset = Path(path).stem
                    if not self._should_include_dataset(dataset, split):
                        continue
                    data = self.load_json_data(
                        path, use_cache=self.should_use_cache(split)
                    )
                    if not data:
                        continue
                    video_groups.setdefault(dataset, {}).update(
                        self._clean_paths_json(data)
                    )

        bonafides = self.metadata_paths[split]["real"]
        if isinstance(bonafides, dict):
            raise ValueError("Expected real metadata structure to be a list.")

        for path in bonafides:
            dataset = Path(path).stem
            if not self._should_include_dataset(dataset, split):
                continue
            data = self.load_json_data(path, use_cache=self.should_use_cache(split))
            if not data:
                continue
            video_groups.setdefault(dataset, {}).update(self._clean_paths_json(data))

        return video_groups

    def convert_groups_to_files(
        self,
        video_groups: dict[str, dict[str, list[str]]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Convert grouped PAD videos into frame-level file records."""

        files: list[dict[str, Any]] = []
        for dataset_name, videos in video_groups.items():
            for video_id, frames in videos.items():
                for frame_path in frames:
                    try:
                        files.append(
                            self.build_frame_record(frame_path, dataset_name, video_id)
                        )
                    except Exception as exc:
                        self.logger.warning(f"Failed to parse path {frame_path}: {exc}")
        self.logger.info(f"Converted {len(files)} PAD frames for {split}")
        return files

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Load and preprocess a single PAD frame."""

        image_tensor, bbox, has_metadata = self.load_frame_tensor(
            file_dict["path"], split
        )
        return {
            **file_dict,
            "image": image_tensor,
            "bboxes": bbox,
            "metadata": has_metadata,
        }


@register_dataset("azure_compute_pad")
class AzureComputePADWrapper(AzurePADMixin, AzureComputeFrameWrapper):
    """PAD dataset wrapper for Azure ML mounted datasets."""


@register_dataset("azure_streaming_pad")
class AzureStreamingPADWrapper(AzurePADMixin, AzureStreamingFrameWrapper):
    """PAD dataset wrapper for Azure blob-backed datasets."""


@register_dataset("azure_compute_multiframe_pad")
class AzureComputeMultiframePADWrapper(
    AzurePADMixin, AzureComputeMultiFrameWrapper
):
    """Multiframe PAD dataset wrapper for Azure ML mounted datasets."""


@register_dataset("azure_streaming_multiframe_pad")
class AzureStreamingMultiframePADWrapper(
    AzurePADMixin, AzureStreamingMultiFrameWrapper
):
    """Multiframe PAD dataset wrapper for Azure blob-backed datasets."""
