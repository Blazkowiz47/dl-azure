"""Generic Azure integration layer for dl-core."""

from . import datasets, executors, storage

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "datasets",
    "executors",
    "storage",
]
