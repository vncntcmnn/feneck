import torch
import torch.nn as nn

from feneck._errors import StrideOrderError


class BaseNeck(nn.Module):
    """Base class for neck modules.

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all levels)
    """

    def __init__(self, in_channels: list[int], in_strides: list[int], out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self._out_channels = out_channels

        # Validate input strides are increasing
        if not all(in_strides[i] < in_strides[i + 1] for i in range(len(in_strides) - 1)):
            raise StrideOrderError()

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through the neck module."""
        raise NotImplementedError

    @property
    @torch.jit.unused
    def out_channels(self) -> list[int]:
        """Output channels for each feature level."""
        return [self._out_channels] * len(self.out_strides)

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        """Output strides for each feature level."""
        # Default implementation returns input strides
        # Subclasses should override if they add/modify levels
        return self.in_strides.copy()
