"""
Azure blob cache utility for caching test data locally.

This module provides a simple file-based cache for Azure blob storage operations,
specifically designed to cache TEST data only to improve performance during
model evaluation and testing.
"""

import atexit
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


# Module-level executor for async cache writes
_cache_executor: Optional[ThreadPoolExecutor] = None


def _get_cache_executor() -> ThreadPoolExecutor:
    """Get or create the shared ThreadPoolExecutor for async cache writes."""
    global _cache_executor
    if _cache_executor is None:
        _cache_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache_")
        atexit.register(_shutdown_cache_executor)
    return _cache_executor


def _shutdown_cache_executor() -> None:
    """Shutdown the cache executor gracefully."""
    global _cache_executor
    if _cache_executor is not None:
        _cache_executor.shutdown(wait=True)
        _cache_executor = None


class AzureBlobCache:
    """
    Simple file-based cache for Azure blob storage operations.

    Caches both images and JSON metadata with hash-based filenames
    to avoid collisions and provide fast lookup.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize Azure blob cache.

        Args:
            cache_dir: Directory to store cached files
            log: Logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.log = logging.getLogger(__name__)

        # Create cache directories
        self.images_dir = self.cache_dir / "images"
        self.metadata_dir = self.cache_dir / "metadata"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.log.debug(f"Azure cache initialized at: {self.cache_dir}")

    def _get_cache_path(self, blob_path: str, file_extension: str = ".png") -> Path:
        """
        Convert blob path to hierarchical cache path.

        Args:
            blob_path: Azure blob path (e.g., "data/frames/Test/Attack/file.jpg"
                       or "face_detected_image/data/frames/Test/Attack/file.jpg")
            file_extension: Extension for cached file (default: .png)

        Returns:
            Path object for the cache file with hierarchical directory structure
        """
        # Handle face detected images
        is_face_detected = blob_path.startswith("face_detected_image/")
        if is_face_detected:
            # Remove the face_detected_image/ prefix
            blob_path = blob_path[len("face_detected_image/") :]

        # Remove 'data/' prefix if present
        if blob_path.startswith("data/"):
            relative_path = blob_path[len("data/") :]
        else:
            # Fallback to full path if data/ prefix not found
            relative_path = blob_path

        # Split path into directory and filename
        path_obj = Path(relative_path)
        directory = path_obj.parent
        filename = path_obj.stem  # filename without extension

        # Add prefix for face detected images
        if is_face_detected:
            directory = f"facedetected/{directory}"

        # Construct final cache path with new extension
        if file_extension == ".json":
            cache_path = self.metadata_dir / directory / f"{filename}{file_extension}"
        else:
            cache_path = self.images_dir / directory / f"{filename}{file_extension}"

        return cache_path

    def get_cached_image(self, blob_path: str) -> Optional[np.ndarray]:
        """
        Get cached image if it exists.

        Args:
            blob_path: Azure blob path for the image

        Returns:
            Cached image as numpy array or None if not cached
        """
        cache_file = self._get_cache_path(blob_path, ".png")

        if cache_file.exists():
            try:
                # Load cached image
                image = cv2.imread(str(cache_file), cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.log.debug(f"Cache hit for image: {blob_path}")
                    return image
                else:
                    # cv2.imread returned None - file is corrupted
                    self.log.warning(f"Cached image is corrupted: {cache_file}")
                    cache_file.unlink(missing_ok=True)
            except Exception as e:
                self.log.warning(f"Failed to load cached image {cache_file}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)

        return None

    def cache_image(self, blob_path: str, image: np.ndarray) -> None:
        """
        Cache image to local storage.

        Args:
            blob_path: Azure blob path for the image
            image: Image as numpy array (RGB format)
        """
        cache_file = self._get_cache_path(blob_path, ".png")

        try:
            # Create parent directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Write directly to cache file
            success = cv2.imwrite(str(cache_file), image_bgr)

            if success:
                self.log.debug(f"Cached image: {blob_path} -> {cache_file}")
            else:
                self.log.warning(f"Failed to write image cache: {blob_path}")

        except Exception as e:
            self.log.warning(f"Failed to cache image {blob_path}: {e}")

    def get_cached_json(self, json_url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached JSON metadata if it exists.

        Args:
            json_url: Azure blob URL for the JSON file

        Returns:
            Cached JSON data or None if not cached
        """
        cache_file = self._get_cache_path(json_url, ".json")

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self.log.debug(f"Cache hit for JSON: {json_url}")
                    return data
            except Exception as e:
                self.log.warning(f"Failed to load cached JSON {cache_file}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)

        return None

    def cache_json(self, json_url: str, data: Dict[str, Any]) -> None:
        """
        Cache JSON metadata to local storage.

        Args:
            json_url: Azure blob URL for the JSON file
            data: JSON data to cache
        """
        cache_file = self._get_cache_path(json_url, ".json")

        try:
            # Create parent directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Write directly to cache file
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            self.log.debug(f"Cached JSON: {json_url} -> {cache_file}")

        except Exception as e:
            self.log.warning(f"Failed to cache JSON {json_url}: {e}")

    def cache_image_async(self, blob_path: str, image: np.ndarray) -> None:
        """
        Cache image to local storage asynchronously (non-blocking).

        Submits the cache write to a background thread pool, allowing the
        main thread to continue without waiting for disk I/O.

        Args:
            blob_path: Azure blob path for the image
            image: Image as numpy array (RGB format)
        """
        # Make a copy of the image to avoid race conditions if the caller
        # modifies the array after this call returns
        image_copy = image.copy()
        executor = _get_cache_executor()
        executor.submit(self._cache_image_sync, blob_path, image_copy)

    def _cache_image_sync(self, blob_path: str, image: np.ndarray) -> None:
        """Internal synchronous cache image implementation."""
        self.cache_image(blob_path, image)

    def cache_json_async(self, json_url: str, data: Dict[str, Any]) -> None:
        """
        Cache JSON metadata to local storage asynchronously (non-blocking).

        Submits the cache write to a background thread pool, allowing the
        main thread to continue without waiting for disk I/O.

        Args:
            json_url: Azure blob URL for the JSON file
            data: JSON data to cache
        """
        # Make a deep copy of the data to avoid race conditions
        data_copy = json.loads(json.dumps(data))
        executor = _get_cache_executor()
        executor.submit(self._cache_json_sync, json_url, data_copy)

    def _cache_json_sync(self, json_url: str, data: Dict[str, Any]) -> None:
        """Internal synchronous cache json implementation."""
        self.cache_json(json_url, data)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            image_count = len(list(self.images_dir.glob("*.png")))
            json_count = len(list(self.metadata_dir.glob("*.json")))

            # Calculate cache size
            cache_size = 0
            for path in self.cache_dir.rglob("*"):
                if path.is_file():
                    cache_size += path.stat().st_size

            return {
                "cached_images": image_count,
                "cached_metadata": json_count,
                "cache_size_mb": cache_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            self.log.warning(f"Failed to get cache stats: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear all cached files."""
        try:
            # Remove all cached files (recursive search for hierarchical structure)
            for cache_file in self.images_dir.rglob("*.png"):
                cache_file.unlink()
            for cache_file in self.metadata_dir.rglob("*.json"):
                cache_file.unlink()

            self.log.info("Cache cleared successfully")
        except Exception as e:
            self.log.warning(f"Failed to clear cache: {e}")
