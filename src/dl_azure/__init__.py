"""Generic Azure integration layer for dl-core."""

from . import datasets, executors, storage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "datasets",
    "executors",
    "storage",
]
