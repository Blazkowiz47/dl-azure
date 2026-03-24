"""Compatibility exports for Azure streaming PAD dataset wrappers."""

from dl_azure.datasets.pad import (
    AzureStreamingMultiframePADWrapper,
    AzureStreamingPADWrapper,
)

__all__ = [
    "AzureStreamingPADWrapper",
    "AzureStreamingMultiframePADWrapper",
]
