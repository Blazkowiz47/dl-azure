"""Tests for safe, hierarchical Azure blob caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_azure.storage.cache import AzureBlobCache


def test_cache_paths_cannot_escape_the_configured_root(tmp_path: Path) -> None:
    """Absolute and parent path segments should be encoded as blob-name data."""
    cache = AzureBlobCache(str(tmp_path / "cache"))

    cache_path = cache._get_cache_path("/../../outside/example.jpg")

    assert cache_path.resolve().is_relative_to(cache.images_dir.resolve())
    assert "%2E%2E" in cache_path.parts


def test_cache_stats_count_nested_files_and_clear_them(tmp_path: Path) -> None:
    """Statistics and cleanup should include the hierarchical cache layout."""
    cache = AzureBlobCache(str(tmp_path / "cache"))
    cache.cache_image(
        "data/frames/test/video/example.jpg",
        np.zeros((4, 4, 3), dtype=np.uint8),
    )
    cache.cache_json(
        "data/metadata/test/video/example.json",
        {"label": "class0"},
    )

    stats = cache.get_cache_stats()
    assert stats["cached_images"] == 1
    assert stats["cached_metadata"] == 1
    assert stats["cache_size_mb"] > 0

    cache.clear_cache()

    cleared_stats = cache.get_cache_stats()
    assert cleared_stats["cached_images"] == 0
    assert cleared_stats["cached_metadata"] == 0
