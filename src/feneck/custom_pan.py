# Adapted from PaddleDetection CustomCSPPAN
# Original Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# https://github.com/PaddlePaddle/PaddleDetection

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import DimensionError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer

__all__ = ["CustomCSPPAN"]


def _get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Literal["batch", "group"] | None = None,
        activation: str = "relu",
        shortcut: bool = True,
        use_alpha: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvNormLayer(in_channels, out_channels, kernel_size=3, norm_type=norm_type)
        self.conv2 = ConvNormLayer(out_channels, out_channels, kernel_size=3, norm_type=norm_type)
        self.shortcut = shortcut and in_channels == out_channels
        self.activation = activation
        self.use_alpha = use_alpha

        if use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._get_activation(self.conv1(x))
        y = self._get_activation(self.conv2(y))
        if self.shortcut:
            if self.use_alpha:
                return self.alpha * x + y
            return x + y
        return y

    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "swish":
            return F.silu(x)
        elif self.activation == "mish":
            return F.mish(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x)
        else:
            return F.relu(x)


class SPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_sizes: list[int],
        norm_type: Literal["batch", "group"] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes])
        self.conv = ConvNormLayer(in_channels, out_channels, kernel_size, norm_type=norm_type)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for pool in self.pools:
            outs.append(pool(x))
        y = torch.cat(outs, dim=1)
        return self._get_activation(self.conv(y))

    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "swish":
            return F.silu(x)
        elif self.activation == "mish":
            return F.mish(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x)
        else:
            return F.relu(x)


class CSPStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        norm_type: Literal["batch", "group"] | None = None,
        activation: str = "relu",
        spp: bool = False,
        use_alpha: bool = False,
    ):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvNormLayer(in_channels, mid_channels, kernel_size=1, norm_type=norm_type)
        self.conv2 = ConvNormLayer(in_channels, mid_channels, kernel_size=1, norm_type=norm_type)
        self.convs = self._build_blocks(mid_channels, num_blocks, norm_type, activation, spp, use_alpha)
        self.conv3 = ConvNormLayer(mid_channels * 2, out_channels, kernel_size=1, norm_type=norm_type)
        self.activation = activation

    def _build_blocks(
        self,
        mid_channels: int,
        num_blocks: int,
        norm_type: Literal["batch", "group"] | None,
        activation: str,
        spp: bool,
        use_alpha: bool,
    ) -> nn.Sequential:
        convs = nn.Sequential()
        next_channels_in = mid_channels

        for i in range(num_blocks):
            block = BasicBlock(
                next_channels_in,
                mid_channels,
                norm_type=norm_type,
                activation=activation,
                shortcut=False,
                use_alpha=use_alpha,
            )
            convs.add_module(str(i), block)

            # Add SPP module at the middle of the stage if requested
            if i == (num_blocks - 1) // 2 and spp:
                spp_module = SPP(
                    mid_channels * 4, mid_channels, 1, [5, 9, 13], norm_type=norm_type, activation=activation
                )
                convs.add_module("spp", spp_module)

            next_channels_in = mid_channels

        return convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self._get_activation(self.conv1(x))
        y2 = self._get_activation(self.conv2(x))
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], dim=1)
        return self._get_activation(self.conv3(y))

    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "swish":
            return F.silu(x)
        elif self.activation == "mish":
            return F.mish(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x)
        else:
            return F.relu(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        if pos_embed is not None:
            src = src + pos_embed

        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(src, src, src)
        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = residual + self.dropout2(src)

        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos_embed)
        return output


class CustomCSPPAN(BaseNeck):
    """Custom CSP-PAN neck with transformer enhancement.

    Adapted from PaddleDetection CustomCSPPAN implementation.

    !!! info "Reference"
        **PP-YOLOv2: A Practical Object Detector**
        *Huang et al., 2021*
        [:material-file-document: Paper](https://arxiv.org/abs/2104.10419)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level
        out_channels: Number of output channels (same for all levels)
        norm_type: Normalization type applied to all convolution layers
        activation: Activation function type
        stage_num: Number of stages in each CSP block
        block_num: Number of basic blocks in each CSP stage
        spp: Whether to use SPP (Spatial Pyramid Pooling) in the deepest level
        use_alpha: Whether to use learnable alpha parameter in residual connections
        width_mult: Width multiplier for channel scaling
        depth_mult: Depth multiplier for block number scaling
        use_transformer: Whether to use transformer enhancement on deepest feature
        transformer_num_heads: Number of attention heads in transformer
        transformer_num_layers: Number of transformer encoder layers
        transformer_dim_feedforward: Feedforward dimension in transformer
        transformer_dropout: Dropout rate in transformer
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        norm_type: Literal["batch", "group"] | None = None,
        activation: str = "relu",
        stage_num: int = 1,
        block_num: int = 3,
        spp: bool = False,
        use_alpha: bool = False,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_transformer: bool = False,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 4,
        transformer_dim_feedforward: int = 2048,
        transformer_dropout: float = 0.1,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        # Apply multipliers
        self.actual_out_channels = max(round(out_channels * width_mult), 1)
        self.block_num = max(round(block_num * depth_mult), 1)
        self.stage_num = stage_num
        self.spp = spp
        self.use_alpha = use_alpha
        self.norm_type = norm_type
        self.activation = activation
        self.use_transformer = use_transformer
        self.num_levels = len(in_channels)

        # Initialize transformer if needed
        if use_transformer:
            self.hidden_dim = in_channels[-1]
            encoder_layer = TransformerEncoderLayer(
                self.hidden_dim, transformer_num_heads, transformer_dim_feedforward, transformer_dropout
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, transformer_num_layers)

        self._build_fpn_stages()
        self._build_pan_stages()

    def _build_fpn_stages(self) -> None:
        fpn_stages = []
        fpn_routes = []

        reversed_in_channels = self.in_channels[::-1]

        channels_pre = 0
        for i, channels_in in enumerate(reversed_in_channels):
            if i > 0:
                channels_in += channels_pre // 2

            stage = nn.Sequential()
            for j in range(self.stage_num):
                input_channels = channels_in if j == 0 else self.actual_out_channels
                stage_layer = CSPStage(
                    input_channels,
                    self.actual_out_channels,
                    self.block_num,
                    norm_type=self.norm_type,
                    spp=(self.spp and i == 0),
                )
                stage.add_module(str(j), stage_layer)

            fpn_stages.append(stage)

            if i < self.num_levels - 1:
                route = ConvNormLayer(
                    self.actual_out_channels, self.actual_out_channels // 2, kernel_size=1, norm_type=self.norm_type
                )
                fpn_routes.append(route)

            channels_pre = self.actual_out_channels

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

    def _build_pan_stages(self) -> None:
        pan_stages = []
        pan_routes = []

        for _ in reversed(range(self.num_levels - 1)):
            route = ConvNormLayer(
                self.actual_out_channels, self.actual_out_channels, kernel_size=3, stride=2, norm_type=self.norm_type
            )
            pan_routes.append(route)

            channels_in = self.actual_out_channels * 2
            stage = nn.Sequential()
            for j in range(self.stage_num):
                input_channels = channels_in if j == 0 else self.actual_out_channels
                stage_layer = CSPStage(
                    input_channels, self.actual_out_channels, self.block_num, norm_type=self.norm_type, spp=False
                )
                stage.add_module(str(j), stage_layer)

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def _build_2d_sincos_position_embedding(
        self, width: int, height: int, embed_dim: int, temperature: float = 10000.0
    ) -> torch.Tensor:
        grid_w = torch.arange(width, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

        if embed_dim % 4 != 0:
            raise DimensionError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten().unsqueeze(-1) @ omega.unsqueeze(0)
        out_h = grid_h.flatten().unsqueeze(-1) @ omega.unsqueeze(0)

        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1).unsqueeze(
            0
        )

        return pos_emb

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.use_transformer:
            last_feat = features[-1]
            batch_size, channels, height, width = last_feat.shape

            src_flatten = last_feat.flatten(2).transpose(1, 2)

            pos_embed = self._build_2d_sincos_position_embedding(width, height, self.hidden_dim).to(last_feat.device)

            memory = self.transformer_encoder(src_flatten, pos_embed)

            last_feat_enhanced = memory.transpose(1, 2).reshape(batch_size, channels, height, width)
            features = [*features[:-1], last_feat_enhanced]

        blocks = features[::-1]
        fpn_feats = []

        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                route = F.interpolate(route, scale_factor=2.0, mode="nearest")
                block = torch.cat([route, block], dim=1)

            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_levels - 1:
                route = F.relu(self.fpn_routes[i](route))

        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]

        for i in reversed(range(self.num_levels - 1)):
            block = fpn_feats[i]
            route = F.relu(self.pan_routes[i](route))
            block = torch.cat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]

    @property
    @torch.jit.unused
    def out_channels(self) -> list[int]:
        return [self.actual_out_channels] * self.num_levels
