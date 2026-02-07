# Adapted from Microsoft DynamicHead
# Original Copyright (c) Microsoft Corporation. All Rights Reserved.
# Licensed under the MIT License
# https://github.com/microsoft/DynamicHead
#
# Includes fix for deformable convolution dimension mismatch from MMDetection from issue #25
# https://github.com/microsoft/DynamicHead/issues/25

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from feneck._errors import ChannelMismatchError, UnsupportedOperationError
from feneck.base import BaseNeck

__all__ = ["DyHead"]


def make_divisible(value: int, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.relu(input_tensor + 3) / 6


class DyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        lambda_a: float = 1.0,
        use_bias: bool = True,
        init_a: list[float] | None = None,
        init_b: list[float] | None = None,
    ):
        super().__init__()
        if init_a is None:
            init_a = [1.0, 0.0]
        if init_b is None:
            init_b = [0.0, 0.0]

        self.out_channels = out_channels
        self.lambda_a = lambda_a * 2
        self.use_bias = use_bias
        self.init_a = init_a
        self.init_b = init_b

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Determine squeeze channels
        squeeze = in_channels // reduction if reduction == 4 else make_divisible(in_channels // reduction, 4)

        # exp=4 for piecewise linear with bias: a1, b1, a2, b2
        self.exp = 4 if use_bias else 2

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, out_channels * self.exp),
            HSigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, channels, *_ = input_tensor.shape

        # Global average pooling and fully connected
        coeffs = self.avg_pool(input_tensor).view(batch_size, channels)
        coeffs = self.fc(coeffs).view(batch_size, self.out_channels * self.exp, 1, 1)

        if self.exp == 4:
            # Piecewise linear with bias: max(x*a1 + b1, x*a2 + b2)
            a1, b1, a2, b2 = torch.split(coeffs, self.out_channels, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            output = torch.max(input_tensor * a1 + b1, input_tensor * a2 + b2)
        elif self.exp == 2:
            # Piecewise linear without bias: max(x*a1, x*a2)
            a1, a2 = torch.split(coeffs, self.out_channels, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            output = torch.max(input_tensor * a1, input_tensor * a2)
        else:
            raise UnsupportedOperationError(f"Unsupported exp value: {self.exp}")

        return output


class DeformableConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: Literal["batch", "group"] | None = "group",
        norm_groups: int = 16,
    ):
        super().__init__()

        use_bias = norm_type is None

        self.conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )

        self.norm = None
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(norm_groups, out_channels)

    def forward(self, input_tensor: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.conv(input_tensor.contiguous(), offset, mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DyConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: Literal["batch", "group"] | None = "group"):
        super().__init__()

        self.conv_list = nn.ModuleList([
            DeformableConvNorm(in_channels, out_channels, stride=1, norm_type=norm_type),  # high-res (next level)
            DeformableConvNorm(in_channels, out_channels, stride=1, norm_type=norm_type),  # current level
            DeformableConvNorm(in_channels, out_channels, stride=2, norm_type=norm_type),  # low-res (prev level)
        ])

        # Scale attention
        self.attention_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, kernel_size=1), nn.ReLU(inplace=True)
        )

        # Offset and mask prediction
        self.offset_conv = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)

        # Task attention
        self.dyrelu = DyReLU(out_channels, out_channels)
        self.hsigmoid = HSigmoid()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []

        for level_idx, feature in enumerate(features):
            # Predict offset and mask
            offset_mask = self.offset_conv(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()

            # Collect features from different scales
            temp_features = []
            attention_features = []

            # Current level
            current_feat = self.conv_list[1](feature, offset, mask)
            temp_features.append(current_feat)
            attention_features.append(self.attention_conv(current_feat))

            # Previous level (lower resolution)
            if level_idx > 0:
                low_feat = self.conv_list[2](features[level_idx - 1], offset, mask)
                temp_features.append(low_feat)
                attention_features.append(self.attention_conv(low_feat))

            # Next level (higher resolution)
            # Fix from MMDetection, DyHead issue #25: upsample input before deformable conv
            if level_idx < len(features) - 1:
                upsampled_input = F.interpolate(
                    features[level_idx + 1],
                    size=(feature.size(2), feature.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
                high_feat = self.conv_list[0](upsampled_input, offset, mask)
                temp_features.append(high_feat)
                attention_features.append(self.attention_conv(high_feat))

            # Apply scale attention
            feature_stack = torch.stack(temp_features)
            attention_stack = self.hsigmoid(torch.stack(attention_features))
            weighted_features = feature_stack * attention_stack
            mean_feature = torch.mean(weighted_features, dim=0)

            # Apply task attention
            output = self.dyrelu(mean_feature)
            outputs.append(output)

        return outputs


class DyHead(BaseNeck):
    """Dynamic Head neck with attention mechanisms for object detection.

    Adapted from Microsoft's DynamicHead implementation with three types of attention:
    - Scale-aware attention: Combines features from different pyramid levels
    - Spatial-aware attention: Uses deformable convolutions for spatial modeling
    - Task-aware attention: Uses dynamic ReLU for channel-wise adaptive activation

    DyHead is typically used after an FPN neck, which is why all input levels are required to have the same number of
    channels (standard FPN output).

    !!! info "Reference"
        **Dynamic Head: Unifying Object Detection Heads with Attentions**
        *Dai et al., CVPR 2021*
        [:material-file-document: Paper](https://arxiv.org/abs/2106.08322)

    Args:
        in_channels: Number of input channels for each feature level (must be same for all levels)
        in_strides: Stride values for each input feature level
        out_channels: Number of output channels (same for all levels)
        num_blocks: Number of DyHead blocks to stack
        norm_type: Normalization type applied to convolutions
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_blocks: int = 6,
        norm_type: Literal["batch", "group"] | None = "group",
    ):
        super().__init__(in_channels, in_strides, out_channels)

        if len(set(in_channels)) != 1:
            raise ChannelMismatchError("DyHead requires all input levels to have the same number of channels")

        self.num_blocks = num_blocks

        dyhead_tower = []
        for block_idx in range(num_blocks):
            block_in_channels = in_channels[0] if block_idx == 0 else out_channels
            block = DyConv(block_in_channels, out_channels, norm_type)
            dyhead_tower.append(block)

        self.dyhead_tower = nn.Sequential(*dyhead_tower)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = features
        for block in self.dyhead_tower:
            outputs = block(outputs)
        return outputs

    @property
    @torch.jit.unused
    def out_channels(self) -> list[int]:
        return [self._out_channels] * len(self.in_strides)
