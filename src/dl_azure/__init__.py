"""Generic Azure integration layer for dl-core."""

from . import callbacks, datasets, executors, metrics_sources, storage, trackers

__version__ = "0.0.2"

__all__ = [
    "__version__",
    "callbacks",
    "datasets",
    "executors",
    "metrics_sources",
    "storage",
    "trackers",
]
