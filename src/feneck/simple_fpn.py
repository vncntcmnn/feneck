from typing import Literal

import torch
import torch.nn as nn

from feneck._errors import ConfigurationError, FeatureCountError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer


class SimpleFPN(BaseNeck):
    """Simple Feature Pyramid Network for transformer-based backbones.

    SimpleFPN is designed for non-hierarchical backbones (like Vision Transformers) that output single-scale features.
    It creates a feature pyramid by applying convolutions at different strides to generate multi-scale
    representations.

    !!! info "Reference"
        **Exploring Plain Vision Transformer Backbones for Object Detection**
        *Li et al., ECCV 2022*
        [:material-file-document: Paper](https://arxiv.org/abs/2203.16527)

    Args:
        in_channels: Number of input channels (int or single-element list)
        in_strides: Input stride (int or single-element list)
        out_channels: Number of output channels (same for all pyramid levels)
        num_levels: Number of pyramid levels to generate
        start_level: Index of pyramid level that matches input resolution. Levels below this index will have higher
            resolution (upsampled), levels above will have lower resolution (downsampled)
        norm_type: Normalization type applied to all convolution layers

    ??? example "Usage Examples"
        ```python
        >>> # ViT outputs 14x14 features at stride 16
        >>> # We want 5 levels: stride 4, 8, 16, 32, 64
        >>> # Input stride 16 matches level 2, so start_level=2
        >>> neck = SimpleFPN(
        ...     in_channels=768, in_strides=16, out_channels=256,
        ...     num_levels=5, start_level=2
        ... )
        >>> # Output: [stride4, stride8, stride16, stride32, stride64]
        >>> #         [level0,  level1, level2,   level3,   level4]
        >>> #         [upsample, upsample, same, downsample, downsample]
        ```
    """

    def __init__(
        self,
        in_channels: int | list[int],
        in_strides: int | list[int],
        out_channels: int = 256,
        num_levels: int = 5,
        start_level: int = 2,
        norm_type: Literal["batch", "group"] | None = None,
    ):
        # Convert to lists if needed
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if isinstance(in_strides, int):
            in_strides = [in_strides]

        # For SimpleFPN, we expect single input from transformer backbone
        if len(in_channels) != 1 or len(in_strides) != 1:
            raise ConfigurationError("SimpleFPN expects single input feature from non-hierarchical backbone")

        super().__init__(in_channels, in_strides, out_channels)

        self.num_levels = num_levels
        self.start_level = start_level
        self.norm_type = norm_type

        # Input projection to adjust channels
        self.input_proj = ConvNormLayer(in_channels[0], out_channels, kernel_size=1, norm_type=norm_type)

        # Build pyramid convolutions for different scales
        self.pyramid_convs = nn.ModuleList()

        for level_idx in range(num_levels):
            if level_idx < start_level:
                # Upsampling levels - use transposed conv with normalization
                stride = 2 ** (start_level - level_idx)
                conv = self._build_upsample_layer(out_channels, stride, norm_type)
            elif level_idx == start_level:
                # Same resolution level - use 3x3 conv
                conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
            else:
                # Downsampling levels - use strided conv
                stride = 2 ** (level_idx - start_level)
                conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, stride=stride, norm_type=norm_type)

            self.pyramid_convs.append(conv)

    def _build_upsample_layer(
        self, out_channels: int, stride: int, norm_type: Literal["batch", "group"] | None
    ) -> nn.Module:
        """Build upsampling layer with optional normalization."""
        transpose_conv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=stride * 2, stride=stride, padding=stride // 2, bias=False
        )

        if norm_type == "batch":
            norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            norm_layer = nn.GroupNorm(32, out_channels)  # Using default 32 groups
        else:
            norm_layer = None

        if norm_layer is not None:
            return nn.Sequential(transpose_conv, norm_layer)
        else:
            return transpose_conv

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass generating multi-scale pyramid from single input feature."""
        if len(features) != 1:
            raise FeatureCountError("SimpleFPN expects single input feature")

        # Project input to target channels
        base_feature = self.input_proj(features[0])

        # Generate pyramid levels
        pyramid_features = []

        for level_idx in range(self.num_levels):
            if level_idx < self.start_level:
                # Upsample for higher resolution levels
                pyramid_feature = self.pyramid_convs[level_idx](base_feature)
            elif level_idx == self.start_level:
                # Same resolution as input
                pyramid_feature = self.pyramid_convs[level_idx](base_feature)
            else:
                # Downsample for lower resolution levels
                pyramid_feature = self.pyramid_convs[level_idx](base_feature)

            pyramid_features.append(pyramid_feature)

        return pyramid_features

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        base_stride = self.in_strides[0]
        strides = []

        for level_idx in range(self.num_levels):
            if level_idx < self.start_level:
                # Higher resolution levels
                stride = base_stride // (2 ** (self.start_level - level_idx))
            else:
                # Same or lower resolution levels
                stride = base_stride * (2 ** (level_idx - self.start_level))
            strides.append(stride)

        return strides
