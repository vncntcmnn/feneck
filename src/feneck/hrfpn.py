# Adapted from Microsoft Research implementation
# Original Copyright (c) 2019 Microsoft Research
# Licensed under the MIT License

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import FeatureCountError, InputLevelError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer


class HRFPN(BaseNeck):
    """High-Resolution Feature Pyramid Network neck module.

    HRFPN aggregates multi-resolution feature representations by upsampling all feature levels
    to the highest resolution, concatenating them, then generating pyramid levels through pooling.
    Originally designed for HRNet but works with any multi-scale backbone.

    !!! info "Reference"
        **Deep High-Resolution Representation Learning for Visual Recognition**
        *Wang et al., TPAMI 2021*
        [:material-file-document: Paper](https://arxiv.org/abs/1908.07919)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all pyramid levels)
        num_levels: Number of output pyramid levels to generate
        pooling_type: Pooling operation for generating lower resolution levels ('max' or 'avg')
        norm_type: Normalization type applied to convolution layers
        stride: Stride for the final 3x3 convolutions in each pyramid level
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_levels: int = 5,
        pooling_type: Literal["max", "avg"] = "avg",
        norm_type: Literal["batch", "group"] | None = None,
        stride: int = 1,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        self.num_levels = num_levels
        self.pooling_type = pooling_type
        self.norm_type = norm_type
        self.stride = stride

        # Validate that we have enough input levels for the requested output levels
        if num_levels > len(in_channels) + 2:
            raise InputLevelError(f"Cannot generate {num_levels} output levels from {len(in_channels)} input levels")

        # Channel reduction conv - combines all feature levels into single representation
        self.reduction_conv = ConvNormLayer(sum(in_channels), out_channels, kernel_size=1, norm_type=norm_type)

        # Final convolutions for each pyramid level
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_levels):
            fpn_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, stride=stride, norm_type=norm_type)
            self.fpn_convs.append(fpn_conv)

        self.pooling = F.max_pool2d if pooling_type == "max" else F.avg_pool2d

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through HRFPN."""
        if len(features) != len(self.in_channels):
            raise FeatureCountError(f"Expected {len(self.in_channels)} input features, got {len(features)}")

        # Get the highest resolution (first feature map)
        highest_resolution = features[0]
        target_size = highest_resolution.shape[2:]

        # Upsample all features to the highest resolution
        upsampled_features = [features[0]]  # First feature is already at target resolution

        for feature in features[1:]:
            upsampled = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            upsampled_features.append(upsampled)

        # Concatenate all upsampled features
        concat_features = torch.cat(upsampled_features, dim=1)

        # Apply channel reduction
        reduced_features = self.reduction_conv(concat_features)

        # Generate pyramid levels
        pyramid_features = []
        current_features = reduced_features

        for level_idx in range(self.num_levels):
            if level_idx == 0:
                # Highest resolution level - apply conv directly
                pyramid_feature = self.fpn_convs[level_idx](current_features)
                pyramid_features.append(pyramid_feature)
            else:
                # Lower resolution levels - apply pooling then conv
                pooled = self.pooling(current_features, kernel_size=2, stride=2)
                pyramid_feature = self.fpn_convs[level_idx](pooled)
                pyramid_features.append(pyramid_feature)
                current_features = pooled

        return pyramid_features

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        """Output strides for each pyramid level."""
        # Start from the highest resolution input stride
        base_stride = self.in_strides[0]
        strides = []

        for level_idx in range(self.num_levels):
            stride = base_stride * (2**level_idx)
            strides.append(stride)

        return strides
