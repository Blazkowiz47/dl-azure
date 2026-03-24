"""Compatibility exports for Azure compute PAD dataset wrappers."""

from dl_azure.datasets.pad import (
    AzureComputeMultiframePADWrapper,
    AzureComputePADWrapper,
)

__all__ = [
    "AzureComputePADWrapper",
    "AzureComputeMultiframePADWrapper",
]
