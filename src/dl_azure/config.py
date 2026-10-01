"""Merge Azure project defaults and experiment overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_azure_config(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively without modifying the supplied configs."""
    merged: dict[str, Any] = {}
    for config in configs:
        if not isinstance(config, dict):
            raise TypeError("Azure configuration must be a mapping")
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key, {}), dict):
                merged[key] = merge_azure_config(merged.get(key, {}), value)
            else:
                merged[key] = deepcopy(value)
    return merged


def normalize_azure_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept flat config or an azure block, with nested values taking precedence."""
    if not isinstance(config, dict):
        raise TypeError("Azure configuration must be a mapping")
    return merge_azure_config(
        {key: value for key, value in config.items() if key != "azure"},
        config.get("azure", {}),
    )
