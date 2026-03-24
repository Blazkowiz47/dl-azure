"""Azure dataset integrations for dl-azure."""

from dl_azure.datasets.azure_compute_pad import (
    AzureComputeMultiframePADWrapper,
    AzureComputePADWrapper,
)
from dl_azure.datasets.azure_streaming_pad import (
    AzureStreamingMultiframePADWrapper,
    AzureStreamingPADWrapper,
)
from dl_azure.datasets.base import (
    AzureComputeFrameWrapper,
    AzureComputeMultiFrameWrapper,
    AzureComputeWrapper,
    AzureStreamingFrameWrapper,
    AzureStreamingMultiFrameWrapper,
    AzureStreamingWrapper,
)

__all__ = [
    "AzureComputeWrapper",
    "AzureStreamingWrapper",
    "AzureComputeFrameWrapper",
    "AzureStreamingFrameWrapper",
    "AzureComputeMultiFrameWrapper",
    "AzureStreamingMultiFrameWrapper",
    "AzureComputePADWrapper",
    "AzureStreamingPADWrapper",
    "AzureComputeMultiframePADWrapper",
    "AzureStreamingMultiframePADWrapper",
]
