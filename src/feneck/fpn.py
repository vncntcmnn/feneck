from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer


class FPN(BaseNeck):
    """Feature Pyramid Network neck module.

    Paper:
        Feature Pyramid Networks for Object Detection
        Lin et al., CVPR 2017
        https://arxiv.org/abs/1612.03144

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all levels), default 256
        has_extra_convs: Whether to add extra conv layers for additional pyramid levels.
            False uses max pooling for one extra level (Faster R-CNN style).
            True uses strided convolutions for extra levels (RetinaNet/FCOS style)
        extra_stage: Number of additional pyramid levels beyond backbone features.
            Only used when has_extra_convs=True for multiple extra levels
        use_c5: Whether to use backbone's highest feature (c5) as input for first extra level.
            False uses FPN's highest feature (p5) instead. Only affects extra convolutions
        norm_type: Normalization type applied to all convolution layers
        relu_before_extra_convs: Whether to apply ReLU activation before extra convolutions
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        has_extra_convs: bool = False,
        extra_stage: int = 1,
        use_c5: bool = True,
        norm_type: Literal["batch", "group"] | None = None,
        relu_before_extra_convs: bool = True,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.norm_type = norm_type
        self.relu_before_extra_convs = relu_before_extra_convs

        # Build lateral convolutions (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            lateral_conv = ConvNormLayer(in_ch, out_channels, kernel_size=1, norm_type=norm_type)
            self.lateral_convs.append(lateral_conv)

        # Build FPN convolutions (3x3 conv to smooth features)
        self.fpn_convs = nn.ModuleList()
        num_backbone_levels = len(in_channels)
        for _ in range(num_backbone_levels):
            fpn_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
            self.fpn_convs.append(fpn_conv)

        # Build extra convolutions if needed
        if has_extra_convs:
            for stage_idx in range(extra_stage):
                if stage_idx == 0 and use_c5:
                    extra_in_channels = in_channels[-1]
                else:
                    extra_in_channels = out_channels

                extra_conv = ConvNormLayer(
                    extra_in_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type
                )
                self.fpn_convs.append(extra_conv)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through FPN."""
        # Apply lateral convolutions
        laterals = []
        for idx, feature in enumerate(features):
            lateral = self.lateral_convs[idx](feature)
            laterals.append(lateral)

        # Top-down pathway with lateral connections
        num_backbone_levels = len(features)
        for idx in range(1, num_backbone_levels):
            level_idx = num_backbone_levels - idx
            # Upsample higher level feature and add to current lateral
            upsampled = F.interpolate(laterals[level_idx], scale_factor=2.0, mode="nearest")
            laterals[level_idx - 1] = laterals[level_idx - 1] + upsampled

        # Apply FPN convolutions to smooth features
        fpn_features = []
        for idx in range(num_backbone_levels):
            fpn_feature = self.fpn_convs[idx](laterals[idx])
            fpn_features.append(fpn_feature)

        # Add extra levels if needed
        if self.extra_stage > 0:
            if not self.has_extra_convs:
                # Use max pooling for extra level (Faster R-CNN style)
                extra_feature = F.max_pool2d(fpn_features[-1], kernel_size=1, stride=2)
                fpn_features.append(extra_feature)
            else:
                # Use extra convolutions (RetinaNet/FCOS style)
                if self.use_c5:
                    extra_source = features[-1]  # Use backbone c5
                else:
                    extra_source = fpn_features[-1]  # Use FPN p5

                extra_feature = self.fpn_convs[num_backbone_levels](extra_source)
                fpn_features.append(extra_feature)

                # Add additional extra stages
                for stage_idx in range(1, self.extra_stage):
                    conv_idx = num_backbone_levels + stage_idx
                    if self.relu_before_extra_convs:
                        extra_input = F.relu(fpn_features[-1])
                    else:
                        extra_input = fpn_features[-1]

                    extra_feature = self.fpn_convs[conv_idx](extra_input)
                    fpn_features.append(extra_feature)

        return fpn_features

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        strides = self.in_strides.copy()

        # Add extra level strides
        if self.extra_stage > 0:
            for _ in range(self.extra_stage):
                next_stride = strides[-1] * 2
                strides.append(next_stride)

        return strides
