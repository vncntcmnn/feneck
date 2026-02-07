# Adapted from EfficientDet BiFPN implementation
# Original Copyright (c) 2020 Google Research. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# https://github.com/google/automl/tree/master/efficientdet

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import InputLevelError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer

__all__ = ["BiFPN"]


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_type: Literal["batch", "group"] | None = None,
        activation: str = "swish",
    ):
        super().__init__()
        self.depthwise = ConvNormLayer(
            in_channels, in_channels, kernel_size, stride, groups=in_channels, norm_type=norm_type
        )
        self.pointwise = ConvNormLayer(in_channels, out_channels, kernel_size=1, norm_type=norm_type)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.activation == "swish":
            x = F.silu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        return x


class BiFPNBlock(nn.Module):
    def __init__(
        self,
        num_levels: int,
        out_channels: int,
        norm_type: Literal["batch", "group"] | None = None,
        use_fast_attention: bool = True,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon
        self.use_fast_attention = use_fast_attention

        # Top-down pathway convolutions (P7->P6, P6->P5, etc.)
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, norm_type=norm_type, activation="swish")
            for _ in range(num_levels - 1)
        ])

        # Bottom-up pathway convolutions (P3->P4, P4->P5, etc.)
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, norm_type=norm_type, activation="swish")
            for _ in range(num_levels - 1)
        ])

        # Fast normalized fusion weights (learnable)
        if use_fast_attention:
            self.td_weights = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)])
            self.bu_weights = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(num_levels - 2)])
            self.bu_weights.append(nn.Parameter(torch.ones(2)))  # P7 only has 2 inputs

    def _fuse(self, feat1: torch.Tensor, feat2: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """Fuse two features with optional size matching and attention weights."""
        if feat1.shape[-2:] != feat2.shape[-2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode="nearest")

        if self.use_fast_attention and weights is not None:
            # Fast normalized fusion: w = ReLU(w) / (sum(w) + ε)
            w = F.relu(weights)
            w = w / (w.sum() + self.epsilon)
            return w[0] * feat1 + w[1] * feat2
        return feat1 + feat2

    def _fuse_multiple(self, features: list[torch.Tensor], weights: torch.Tensor = None) -> torch.Tensor:
        """Fuse multiple features (for 3-input nodes in bottom-up path)."""
        if len(features) == 2:
            if self.use_fast_attention and weights is not None:
                w = F.relu(weights[:2])
                w = w / (w.sum() + self.epsilon)
                return w[0] * features[0] + w[1] * features[1]
            return features[0] + features[1]

        if self.use_fast_attention and weights is not None:
            w = F.relu(weights)
            w = w / (w.sum() + self.epsilon)
            return sum(w[i] * feat for i, feat in enumerate(features))

        return sum(features)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Top-down pathway: start from highest level (P7), work down to P3
        td_feats = [features[-1]]  # P7 unchanged

        for i in range(self.num_levels - 2, -1, -1):
            higher_feat = F.interpolate(td_feats[0], scale_factor=2.0, mode="nearest")
            current_feat = features[i]
            weights = self.td_weights[self.num_levels - 2 - i] if self.use_fast_attention else None
            fused = self._fuse(current_feat, higher_feat, weights)
            td_feat = self.td_convs[self.num_levels - 2 - i](fused)
            td_feats.insert(0, td_feat)

        # Bottom-up pathway: start from lowest level (P3), work up to P7
        bu_feats = [td_feats[0]]  # P3 from top-down becomes final P3

        for i in range(1, self.num_levels):
            lower_feat = F.max_pool2d(bu_feats[-1], kernel_size=3, stride=2, padding=1)

            # Build inputs: original feature + same-level skip connection + bottom-up feature
            if i == self.num_levels - 1:  # P7: only 2 inputs (no skip connection)
                inputs = [features[i], lower_feat]
                weights = self.bu_weights[i - 1] if self.use_fast_attention else None
            else:  # P4,P5,P6: 3 inputs including same-level skip from original input
                inputs = [features[i], td_feats[i], lower_feat]
                weights = self.bu_weights[i - 1] if self.use_fast_attention else None

            fused = self._fuse_multiple(inputs, weights)
            bu_feat = self.bu_convs[i - 1](fused)
            bu_feats.append(bu_feat)

        return bu_feats


class BiFPN(BaseNeck):
    """Bidirectional Feature Pyramid Network for efficient multi-scale feature fusion.

    BiFPN introduces weighted bidirectional cross-scale connections for improved feature fusion compared to traditional
    FPN and PANet. Key innovations include fast normalized fusion, same-level skip connections, and depthwise separable convolutions.

    !!! info "Reference"
        **EfficientDet: Scalable and Efficient Object Detection**
        *Tan et al., CVPR 2020*
        [:material-file-document: Paper](https://arxiv.org/abs/1911.09070)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level
        out_channels: Number of output channels (same for all levels)
        num_levels: Total number of pyramid levels
        num_layers: Number of BiFPN blocks to stack
        norm_type: Normalization type applied to all convolution layers
        use_fast_attention: Whether to use fast attention for feature fusion
        epsilon: Small value for numerical stability in attention normalization
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_levels: int = 5,
        num_layers: int = 3,
        norm_type: Literal["batch", "group"] | None = None,
        use_fast_attention: bool = True,
        epsilon: float = 1e-4,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        self.num_levels = num_levels
        self.num_layers = num_layers
        self.num_input_levels = len(in_channels)

        if self.num_input_levels > num_levels:
            raise InputLevelError(f"Too many input levels ({self.num_input_levels}) for total levels ({num_levels})")

        self.input_convs = nn.ModuleList([
            ConvNormLayer(in_ch, out_channels, kernel_size=1, norm_type=norm_type)
            if in_ch != out_channels
            else nn.Identity()
            for in_ch in in_channels
        ])

        num_extra = num_levels - self.num_input_levels
        self.extra_convs = nn.ModuleList([
            ConvNormLayer(
                in_channels[-1] if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
            )
            for i in range(num_extra)
        ])

        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(num_levels, out_channels, norm_type, use_fast_attention, epsilon) for _ in range(num_layers)
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Project input features to common channel dimension
        projected = [conv(feat) for conv, feat in zip(self.input_convs, features, strict=True)]

        # Generate additional pyramid levels if needed (P6, P7, etc.)
        current = projected
        for i, conv in enumerate(self.extra_convs):
            source = features[-1] if i == 0 else current[-1]  # Use original C5 for P6, then generated features
            if i > 0:
                source = F.relu(source)  # Apply ReLU before subsequent levels
            extra = conv(source)
            current.append(extra)

        # Apply stacked BiFPN blocks for iterative feature refinement
        for block in self.bifpn_blocks:
            current = block(current)

        return current

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        strides = self.in_strides.copy()
        for _ in range(self.num_levels - self.num_input_levels):
            strides.append(strides[-1] * 2)
        return strides
