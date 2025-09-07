from typing import Literal

import torch.nn as nn


class ConvNormLayer(nn.Module):
    """Convolution layer with optional normalization.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to all four sides of the input. If None, auto-calculated
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels
        norm_type: Type of normalization ('batch', 'group', None)
        norm_groups: Number of groups for GroupNorm (only used when norm_type='group')
        bias: Whether to use bias in convolution (disabled when using normalization)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        norm_type: Literal["batch", "group"] | None = None,
        norm_groups: int = 32,
        bias: bool = False,
    ):
        super().__init__()

        # Auto-disable bias when using normalization
        use_bias = bias if norm_type is None else False

        # Auto-calculate padding to maintain spatial dimensions if not specified
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=use_bias
        )

        self.norm = None
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(norm_groups, out_channels)

    def forward(self, x):
        """Forward pass through convolution and normalization."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
