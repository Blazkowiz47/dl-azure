"""Generic Azure dataset base wrappers for dl-azure."""

from dl_azure.datasets.base import (
    AzureComputeFrameWrapper,
    AzureComputeMultiFrameWrapper,
    AzureComputeWrapper,
    AzureStreamingFrameWrapper,
    AzureStreamingMultiFrameWrapper,
    AzureStreamingWrapper,
)
from dl_azure.datasets.tar_shard import (
    AzureComputeTarShardWrapper,
    AzureStreamingTarShardWrapper,
)

__all__ = [
    "AzureComputeWrapper",
    "AzureStreamingWrapper",
    "AzureComputeFrameWrapper",
    "AzureStreamingFrameWrapper",
    "AzureComputeMultiFrameWrapper",
    "AzureStreamingMultiFrameWrapper",
    "AzureComputeTarShardWrapper",
    "AzureStreamingTarShardWrapper",
]
