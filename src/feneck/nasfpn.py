# Adapted from TensorFlow Models NAS-FPN implementation
# Original Copyright (c) 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# https://github.com/tensorflow/models/blob/v2.15.0/official/vision/modeling/decoders/nasfpn.py

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import ConfigurationError, FeatureCountError, UnsupportedOperationError
from feneck.base import BaseNeck

__all__ = ["NASFPN"]

NUM_PYRAMID_NODES = 5  # P3-P7


@dataclass(frozen=True)
class BlockSpec:
    """Represents a NAS-FPN block specification.

    Following the original TensorFlow implementation for fidelity,
    not simplified or hardcoded for optimization.
    """

    level: int
    op: str
    inputs: tuple[int, int]
    is_output: bool


# The discovered NAS-FPN architecture (Ghiasi et al., 2019).
# We keep this for reference, but the implementation supports custom specs.
DISCOVERED_NASFPN_CONFIG = [
    BlockSpec(4, "attention", (1, 3), False),
    BlockSpec(4, "sum", (1, 5), False),
    BlockSpec(3, "sum", (0, 6), True),
    BlockSpec(4, "sum", (6, 7), True),
    BlockSpec(5, "attention", (7, 8), True),
    BlockSpec(7, "attention", (6, 9), True),
    BlockSpec(6, "attention", (9, 10), True),
]


class NASFPNOperationBlock(nn.Module):
    """NAS-FPN operation block for feature fusion."""

    def __init__(
        self,
        out_channels: int,
        inputs: tuple[int, int],
        input_levels: list[int],
        target_level: int,
        op: str,
        norm_type: str | None = "batch",
    ):
        super().__init__()
        self.inputs = inputs
        self.input_levels = input_levels
        self.target_level = target_level
        self.op_type = op

        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=(norm_type is None))
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def resample(feature: torch.Tensor, from_level: int, to_level: int) -> torch.Tensor:
        """Resample feature map to the target level using pooling/upsampling."""
        if from_level == to_level:
            return feature
        elif from_level < to_level:
            stride = 2 ** (to_level - from_level)
            return F.max_pool2d(feature, kernel_size=stride, stride=stride)
        else:
            scale = 2 ** (from_level - to_level)
            return F.interpolate(feature, scale_factor=scale, mode="nearest")

    @staticmethod
    def global_attention(feat0: torch.Tensor, feat1: torch.Tensor) -> torch.Tensor:
        """Channel attention as defined in NAS-FPN."""
        attention = torch.amax(feat0, dim=[2, 3], keepdim=True).sigmoid()
        return feat0 + feat1 * attention

    def forward(self, nodes: list[torch.Tensor]) -> torch.Tensor:
        resampled = [
            self.resample(nodes[idx], from_level=level, to_level=self.target_level)
            for idx, level in zip(self.inputs, self.input_levels, strict=True)
        ]

        if self.op_type == "sum":
            fused = resampled[0] + resampled[1]
        elif self.op_type == "attention":
            level0, level1 = self.input_levels
            if level0 >= level1:
                fused = self.global_attention(resampled[0], resampled[1])
            else:
                fused = self.global_attention(resampled[1], resampled[0])
        else:
            raise UnsupportedOperationError(f"Unknown op type {self.op_type}")

        output = self.activation(fused)
        output = self.conv(output)
        output = self.norm(output)
        return output


class NASFPNCell(nn.Module):
    """Single NAS-FPN cell with dynamic recycling.

    Matches the original TensorFlow reference implementation for fidelity.
    """

    def __init__(self, out_channels: int, norm_type: str | None = "batch", block_config: list[BlockSpec] | None = None):
        super().__init__()
        if block_config is None:
            block_config = DISCOVERED_NASFPN_CONFIG

        self.specs = block_config
        self.blocks = nn.ModuleList()

        # Build operation blocks - we need to track node levels as we build
        node_levels = [3 + i for i in range(NUM_PYRAMID_NODES)]  # Initial P3-P7 levels

        for spec in self.specs:
            # Get input levels for the two parent nodes
            input_levels = [node_levels[idx] for idx in spec.inputs]

            self.blocks.append(
                NASFPNOperationBlock(out_channels, spec.inputs, input_levels, spec.level, spec.op, norm_type)
            )

            # Add this new node's level for future blocks
            node_levels.append(spec.level)

    def forward(self, pyramid_features: list[torch.Tensor]) -> list[torch.Tensor]:
        nodes = list(pyramid_features)
        node_levels = [3 + i for i in range(NUM_PYRAMID_NODES)]  # P3, P4, P5, P6, P7
        num_output_connections = [0] * len(nodes)

        for spec, block in zip(self.specs, self.blocks, strict=True):
            # Build node
            output = block(nodes)

            # Update input usage count
            for inp in spec.inputs:
                num_output_connections[inp] += 1

            # Recycle unused nodes of the same level if this is an output block
            if spec.is_output:
                for i, (feat, level, conn) in enumerate(zip(nodes, node_levels, num_output_connections, strict=True)):
                    if conn == 0 and level == spec.level:
                        resampled = NASFPNOperationBlock.resample(feat, level, spec.level)
                        output = output + resampled
                        num_output_connections[i] += 1

            # Append new node
            nodes.append(output)
            node_levels.append(spec.level)
            num_output_connections.append(0)

        # Collect last NUM_PYRAMID_NODES nodes and their levels
        final_nodes = nodes[-NUM_PYRAMID_NODES:]
        final_levels = node_levels[-NUM_PYRAMID_NODES:]

        # Re-order by increasing level (P3-P7)
        ordered = [feat for _, feat in sorted(zip(final_levels, final_nodes, strict=True), key=lambda x: x[0])]
        return ordered


class NASFPN(BaseNeck):
    """Neural Architecture Search Feature Pyramid Network.

    NAS-FPN uses learned feature fusion patterns discovered through neural architecture search.
    It employs dynamic feature recycling and channel attention for efficient multi-scale feature fusion.
    Generates 5 pyramid levels (P3-P7) regardless of input configuration.

    !!! info "Reference"
        **NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection**
        *Ghiasi et al., CVPR 2019*
        [:material-file-document: Paper](https://arxiv.org/abs/1904.07392)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level
        out_channels: Number of output channels (same for all levels)
        num_repeats: Number of NAS-FPN cells to stack
        norm_type: Normalization type applied to convolution layers
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_repeats: int = 5,
        norm_type: str | None = "batch",
    ):
        super().__init__(in_channels, in_strides, out_channels)
        if len(in_channels) < 3:
            raise ConfigurationError(f"Expected at least 3 input levels, got {len(in_channels)}")

        self.num_repeats = num_repeats

        # Lateral 1x1 convs to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
            for channels in in_channels
        ])

        # Extra convs for levels above provided features to reach NUM_PYRAMID_NODES
        self.extra_convs = nn.ModuleList()
        for i in range(max(0, NUM_PYRAMID_NODES - len(in_channels))):
            src_channels = in_channels[-1] if i == 0 else out_channels
            self.extra_convs.append(nn.Conv2d(src_channels, out_channels, 3, stride=2, padding=1))

        # NAS-FPN cells
        self.nasfpn_cells = nn.ModuleList([
            NASFPNCell(out_channels, norm_type, DISCOVERED_NASFPN_CONFIG) for _ in range(num_repeats)
        ])

    def _build_pyramid(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        pyramid_features = [conv(feature) for feature, conv in zip(features, self.lateral_convs, strict=True)]
        for i, conv in enumerate(self.extra_convs):
            source = F.relu(features[-1]) if i == 0 else F.relu(pyramid_features[-1])
            pyramid_features.append(conv(source))
        return pyramid_features

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.in_channels):
            raise FeatureCountError(f"Expected {len(self.in_channels)} features, got {len(features)}")

        pyramid_features = self._build_pyramid(features)
        if len(pyramid_features) != NUM_PYRAMID_NODES:
            raise ConfigurationError(f"Expected {NUM_PYRAMID_NODES} pyramid levels, got {len(pyramid_features)}")

        current_features = pyramid_features
        for cell in self.nasfpn_cells:
            current_features = cell(current_features)
        return current_features

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        # Always returns strides for P3-P7: [8, 16, 32, 64, 128]
        base_stride = self.in_strides[0] if len(self.in_strides) >= 3 else 8
        return [base_stride * (2**i) for i in range(NUM_PYRAMID_NODES)]
