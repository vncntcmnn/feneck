from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import FeatureCountError, InputLevelError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer


class PAFPN(BaseNeck):
    """Path Aggregation Feature Pyramid Network neck module.

    PAFPN enhances FPN by adding a bottom-up path augmentation that shortens the information
    path between lower and top feature levels. This preserves fine-grained localization
    information through direct connections after the initial FPN processing.

    !!! info "Reference"
        **Path Aggregation Network for Instance Segmentation**
        *Liu et al., CVPR 2018*
        [:material-file-document: Paper](https://arxiv.org/abs/1803.01534)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all levels)
        num_levels: Total number of output pyramid levels. If None, uses len(in_channels).
            Must be >= len(in_channels)
        has_extra_convs: Whether to add extra conv layers for additional pyramid levels.
            False uses max pooling for one extra level (Faster R-CNN style).
            True uses strided convolutions for extra levels (RetinaNet/FCOS style)
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
        num_levels: int | None = None,
        has_extra_convs: bool = False,
        use_c5: bool = True,
        norm_type: Literal["batch", "group"] | None = None,
        relu_before_extra_convs: bool = True,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        # Set default num_levels to input levels if not specified
        if num_levels is None:
            num_levels = len(in_channels)

        if num_levels < len(in_channels):
            raise InputLevelError(f"num_levels ({num_levels}) cannot be less than input levels ({len(in_channels)})")

        self.num_levels = num_levels
        self.extra_levels = num_levels - len(in_channels)
        self.has_extra_convs = has_extra_convs
        self.use_c5 = use_c5
        self.norm_type = norm_type
        self.relu_before_extra_convs = relu_before_extra_convs
        self.num_backbone_levels = len(in_channels)

        # Build lateral convolutions (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            lateral_conv = ConvNormLayer(in_ch, out_channels, kernel_size=1, norm_type=norm_type)
            self.lateral_convs.append(lateral_conv)

        # Build FPN convolutions (3x3 conv to smooth features after top-down)
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_backbone_levels):
            fpn_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
            self.fpn_convs.append(fpn_conv)

        # Build bottom-up downsample convolutions (stride-2 3x3 conv)
        self.downsample_convs = nn.ModuleList()
        for _ in range(self.num_backbone_levels - 1):  # P2->P3, P3->P4, etc.
            downsample_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type)
            self.downsample_convs.append(downsample_conv)

        # Build PAFPN convolutions (3x3 conv to smooth features after bottom-up)
        self.pafpn_convs = nn.ModuleList()
        for _ in range(self.num_backbone_levels - 1):  # P3, P4, P5, ... (P2 uses fpn_conv directly)
            pafpn_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
            self.pafpn_convs.append(pafpn_conv)

        # Build extra convolutions if needed
        if has_extra_convs and self.extra_levels > 0:
            for stage_idx in range(self.extra_levels):
                extra_in_channels = in_channels[-1] if (stage_idx == 0 and use_c5) else out_channels
                extra_conv = ConvNormLayer(
                    extra_in_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type
                )
                self.fpn_convs.append(extra_conv)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through PAFPN."""
        if len(features) != len(self.in_channels):
            raise FeatureCountError(f"Expected {len(self.in_channels)} input features, got {len(features)}")

        # Step 1: Apply lateral convolutions
        laterals = []
        for idx, feature in enumerate(features):
            lateral = self.lateral_convs[idx](feature)
            laterals.append(lateral)

        # Step 2: Top-down pathway (standard FPN)
        for idx in range(self.num_backbone_levels - 1, 0, -1):
            prev_shape = laterals[idx - 1].shape[2:]
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(laterals[idx], size=prev_shape, mode="nearest")

        # Step 3: Apply FPN convolutions to get intermediate outputs
        inter_outs = []
        for idx in range(self.num_backbone_levels):
            inter_out = self.fpn_convs[idx](laterals[idx])
            inter_outs.append(inter_out)

        # Step 4: Bottom-up path augmentation
        for idx in range(self.num_backbone_levels - 1):
            # Add downsampled lower-level feature to higher-level feature
            inter_outs[idx + 1] = inter_outs[idx + 1] + self.downsample_convs[idx](inter_outs[idx])

        # Step 5: Build final outputs
        # P2 uses the FPN output directly (no additional PAFPN conv)
        outs = [inter_outs[0]]

        # P3, P4, P5, ... use PAFPN convolutions
        for idx in range(1, self.num_backbone_levels):
            pafpn_out = self.pafpn_convs[idx - 1](inter_outs[idx])
            outs.append(pafpn_out)

        # Step 6: Add extra levels if needed
        if self.extra_levels > 0:
            if not self.has_extra_convs:
                # Use max pooling for extra level (Faster R-CNN style)
                for _ in range(self.extra_levels):
                    extra_feature = F.max_pool2d(outs[-1], kernel_size=1, stride=2)
                    outs.append(extra_feature)
            else:
                # Use extra convolutions (RetinaNet/FCOS style)
                extra_source = features[-1] if self.use_c5 else outs[-1]
                extra_feature = self.fpn_convs[self.num_backbone_levels](extra_source)
                outs.append(extra_feature)

                # Add additional extra stages
                for stage_idx in range(1, self.extra_levels):
                    conv_idx = self.num_backbone_levels + stage_idx
                    extra_input = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
                    extra_feature = self.fpn_convs[conv_idx](extra_input)
                    outs.append(extra_feature)

        return outs

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        strides = self.in_strides.copy()

        # Add extra level strides
        for _ in range(self.extra_levels):
            next_stride = strides[-1] * 2
            strides.append(next_stride)

        return strides
