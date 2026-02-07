from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import FeatureCountError, InputLevelError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer

__all__ = ["CARAFE"]


class CARAFEUpsampler(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE) upsampler.

    CARAFE Algorithm:

    1. Channel compression: 1x1 conv to reduce channels
    2. Content encoding: Generate reassembly kernels for each spatial location
    3. Kernel normalization: Apply softmax to make valid convolution kernels
    4. Feature reassembly: Apply kernels to upsampled features

    This implementation uses standard PyTorch operations without CUDA dependencies.

    Args:
        channels: Number of input channels
        scale_factor: Upsampling scale factor
        up_kernel: Kernel size for reassembly operation
        encoder_kernel: Kernel size for content encoder
        encoder_dilation: Dilation for content encoder
        compressed_channels: Intermediate channels after compression
        norm_type: Normalization type for convolutions
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        compressed_channels: int = 64,
        norm_type: Literal["batch", "group"] | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        self.channel_compressor = ConvNormLayer(channels, compressed_channels, kernel_size=1, norm_type=norm_type)

        encoder_out_channels = up_kernel * up_kernel * scale_factor * scale_factor
        encoder_padding = ((encoder_kernel - 1) * encoder_dilation) // 2

        self.content_encoder = nn.Conv2d(
            compressed_channels,
            encoder_out_channels,
            kernel_size=encoder_kernel,
            padding=encoder_padding,
            dilation=encoder_dilation,
            bias=False,
        )

        nn.init.normal_(self.content_encoder.weight, std=0.001)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        N, C, *_ = features.shape

        compressed = self.channel_compressor(features)
        kernel_prediction = self.content_encoder(compressed)

        kernel_prediction = F.pixel_shuffle(kernel_prediction, self.scale_factor)
        N, k_squared, sH, sW = kernel_prediction.shape

        kernel_prediction = kernel_prediction.view(N, 1, k_squared, sH, sW)
        kernels = F.softmax(kernel_prediction, dim=2)

        features_up = F.interpolate(features, scale_factor=self.scale_factor, mode="nearest")

        patches = F.unfold(
            features_up,
            kernel_size=self.up_kernel,
            dilation=self.scale_factor,
            padding=(self.up_kernel // 2) * self.scale_factor,
        )

        patches = patches.view(N, C, k_squared, sH * sW)
        patches = patches.view(N, C, k_squared, sH, sW)

        output = (patches * kernels).sum(dim=2)

        return output


class CARAFE(BaseNeck):
    """CARAFE Feature Pyramid Network neck module.

    Replaces standard FPN upsampling with Content-Aware ReAssembly of FEatures (CARAFE) for improved feature
    upsampling and multi-scale feature fusion.

    This implementation uses standard PyTorch operations without CUDA dependencies.

    !!! info "Reference"
        **CARAFE: Content-Aware ReAssembly of FEatures**
        *Wang et al., ICCV 2019*
        [:material-file-document: Paper](https://arxiv.org/abs/1905.02188)

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
        carafe_scale_factor: Scale factor for CARAFE upsampling
        carafe_up_kernel: Kernel size for CARAFE reassembly operation
        carafe_encoder_kernel: Kernel size for CARAFE content encoder
        carafe_encoder_dilation: Dilation for CARAFE content encoder
        carafe_compressed_channels: Intermediate channels in CARAFE
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
        carafe_scale_factor: int = 2,
        carafe_up_kernel: int = 5,
        carafe_encoder_kernel: int = 3,
        carafe_encoder_dilation: int = 1,
        carafe_compressed_channels: int = 64,
        norm_type: Literal["batch", "group"] | None = None,
        relu_before_extra_convs: bool = True,
    ):
        super().__init__(in_channels, in_strides, out_channels)

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

        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            lateral_conv = ConvNormLayer(in_ch, out_channels, kernel_size=1, norm_type=norm_type)
            self.lateral_convs.append(lateral_conv)

        self.carafe_upsamplers = nn.ModuleList()
        num_backbone_levels = len(in_channels)
        for _ in range(num_backbone_levels - 1):
            carafe = CARAFEUpsampler(
                channels=out_channels,
                scale_factor=carafe_scale_factor,
                up_kernel=carafe_up_kernel,
                encoder_kernel=carafe_encoder_kernel,
                encoder_dilation=carafe_encoder_dilation,
                compressed_channels=carafe_compressed_channels,
                norm_type=norm_type,
            )
            self.carafe_upsamplers.append(carafe)

        self.fpn_convs = nn.ModuleList()
        for _ in range(num_backbone_levels):
            fpn_conv = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
            self.fpn_convs.append(fpn_conv)

        if has_extra_convs and self.extra_levels > 0:
            for stage_idx in range(self.extra_levels):
                extra_in_channels = in_channels[-1] if (stage_idx == 0 and use_c5) else out_channels
                extra_conv = ConvNormLayer(
                    extra_in_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type
                )
                self.fpn_convs.append(extra_conv)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.in_channels):
            raise FeatureCountError(f"Expected {len(self.in_channels)} input features, got {len(features)}")

        laterals = []
        for idx, feature in enumerate(features):
            lateral = self.lateral_convs[idx](feature)
            laterals.append(lateral)

        num_backbone_levels = len(features)
        for idx in range(num_backbone_levels - 1, 0, -1):
            upsampled = self.carafe_upsamplers[idx - 1](laterals[idx])
            laterals[idx - 1] = laterals[idx - 1] + upsampled

        fpn_features = []
        for idx in range(num_backbone_levels):
            fpn_feature = self.fpn_convs[idx](laterals[idx])
            fpn_features.append(fpn_feature)

        if self.extra_levels > 0:
            if not self.has_extra_convs:
                for _ in range(self.extra_levels):
                    extra_feature = F.max_pool2d(fpn_features[-1], kernel_size=1, stride=2)
                    fpn_features.append(extra_feature)
            else:
                extra_source = features[-1] if self.use_c5 else fpn_features[-1]
                extra_feature = self.fpn_convs[num_backbone_levels](extra_source)
                fpn_features.append(extra_feature)

                for stage_idx in range(1, self.extra_levels):
                    conv_idx = num_backbone_levels + stage_idx
                    extra_input = F.relu(fpn_features[-1]) if self.relu_before_extra_convs else fpn_features[-1]
                    extra_feature = self.fpn_convs[conv_idx](extra_input)
                    fpn_features.append(extra_feature)

        return fpn_features

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        strides = self.in_strides.copy()

        for _ in range(self.extra_levels):
            next_stride = strides[-1] * 2
            strides.append(next_stride)

        return strides
