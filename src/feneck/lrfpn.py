from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import ConfigurationError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer

__all__ = ["LRFPN"]


class SPIEMBlock(nn.Module):
    """Single-scale Shallow Position Information Extraction (SPIEM) block.

    Implements Eq. (5)-(9) from the LR-FPN paper. Enhances a higher-level feature with shallow spatial position
    information.

    Args:
        shallow_channels: Number of channels in shallow feature F1
        out_channels: Number of output channels
        scale_factor: Downscaling factor from shallow to target resolution
        norm_type: Normalization type ('batch', 'group', None)
    """

    def __init__(
        self,
        shallow_channels: int,
        out_channels: int,
        scale_factor: int,
        norm_type: Literal["batch", "group"] | None = None,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.aap_weight = nn.Parameter(torch.ones(1, shallow_channels, 1, 1))
        self.amp_weight = nn.Parameter(torch.ones(1, shallow_channels, 1, 1))
        self.align_conv = ConvNormLayer(shallow_channels, out_channels, kernel_size=1, norm_type=norm_type)

    def forward(self, shallow_feat: torch.Tensor) -> torch.Tensor:
        h, w = shallow_feat.shape[2:]
        target_h, target_w = h // self.scale_factor, w // self.scale_factor

        aap = F.adaptive_avg_pool2d(shallow_feat, (target_h, target_w))
        amp = F.adaptive_max_pool2d(shallow_feat, (target_h, target_w))
        fused = self.aap_weight * aap + self.amp_weight * amp
        return self.align_conv(fused)


class ChannelInteraction(nn.Module):
    """Channel interaction branch inside CIM."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2 * channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(n, c)
        gmp = F.adaptive_max_pool2d(x, 1).view(n, c)
        a = self.relu(self.fc1(gap))
        m = self.relu(self.fc1(gmp))
        w = self.sigmoid(self.fc2(torch.cat([a, m], dim=1))).view(n, c, 1, 1)
        return x * w + x


class CIMBlock(nn.Module):
    """Contextual Interaction Module (CIM) block.

    Implements Eq. (10)-(12) from the LR-FPN paper. Replaces the 1x1 lateral conv in FPN with local depthwise, dilated
    depthwise, and channel interaction branches.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int = 3, norm_type: Literal["batch", "group"] | None = None
    ):
        super().__init__()
        self.dw_local = ConvNormLayer(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, norm_type=norm_type
        )
        self.dw_nonlocal = ConvNormLayer(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            groups=in_channels,
            norm_type=norm_type,
        )

        self.channel = ChannelInteraction(in_channels)
        self.proj = ConvNormLayer(in_channels, out_channels, kernel_size=1, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw_local(x)
        nonlocal_ = self.dw_nonlocal(x)
        ch = self.channel(x)
        fused = local + nonlocal_ + ch
        return self.proj(fused)


class LRFPN(BaseNeck):
    """Location-Refined Feature Pyramid Network (LR-FPN).

    The first input feature is always used as F1 (shallow). All subsequent inputs are treated as higher-level
    features F2, F3, ..., each refined by SPIEM and CIM. The last CIM output serves as the base of the pyramid.
    Additional pyramid levels are built top-down (if multiple higher features are available) and with stride-2
    convolutions until `num_levels` outputs are produced.

    !!! info "Reference"
        **LR-FPN: Enhancing Remote Sensing Object Detection with Location Refined Feature Pyramid Network**
        *Li et al., 2024*
        [:material-file-document: Paper](https://arxiv.org/abs/2404.01614)

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all levels)
        num_levels: Total number of output pyramid levels (>= 1).
            Defaults to len(in_channels) - 1 + 2 (paper's setting).
        norm_type: Normalization type applied to ConvNormLayer
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_levels: int | None = None,
        norm_type: Literal["batch", "group"] | None = None,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        if len(in_channels) < 2:
            raise ConfigurationError("LR-FPN requires at least 2 input features (F1 + one higher level)")

        # default num_levels = (#higher feats) + 2 (as in paper)
        if num_levels is None:
            num_levels = len(in_channels) - 1 + 2
        if num_levels < 1:
            raise ConfigurationError("num_levels must be >= 1")
        self.num_levels = num_levels

        shallow_ch = in_channels[0]
        shallow_stride = in_strides[0]
        higher_chs = in_channels[1:]
        higher_strides = in_strides[1:]

        # compute scale factors for SPIEM
        scale_factors = [stride // shallow_stride for stride in higher_strides]

        self.spiem_blocks = nn.ModuleList([
            SPIEMBlock(shallow_ch, ch, scale_factor, norm_type=norm_type)
            for ch, scale_factor in zip(higher_chs, scale_factors, strict=False)
        ])
        self.cim_blocks = nn.ModuleList([
            CIMBlock(ch, out_channels, dilation=3, norm_type=norm_type) for ch in higher_chs
        ])

        # maximum extra convs we might need
        max_extra = max(0, num_levels - (len(in_channels) - 1))
        self.extra_convs = nn.ModuleList([
            ConvNormLayer(out_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type)
            for _ in range(max_extra)
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        shallow, higher_feats = features[0], features[1:]

        # SPIEM: Fi* for each higher feature
        fi_stars = [blk(shallow) for blk in self.spiem_blocks]

        # CIM: refine Fi + Fi*
        refined = [blk(Fi + Fi_star) for blk, Fi, Fi_star in zip(self.cim_blocks, higher_feats, fi_stars, strict=False)]

        # base = last CIM output
        outs = [refined[-1]]

        # build top-down fusion (only if >1 higher feature)
        for i in range(len(refined) - 2, -1, -1):
            up = F.interpolate(outs[0], size=refined[i].shape[-2:], mode="nearest")
            outs.insert(0, refined[i] + up)

        # cut/expand to num_levels
        if len(outs) > self.num_levels:
            outs = outs[-self.num_levels :]
        while len(outs) < self.num_levels:
            outs.append(self.extra_convs[len(outs) - (len(refined))](outs[-1]))

        return outs

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        base_strides = self.in_strides[1:]  # skip shallow
        # pyramid strides: match refined levels (top-down), then extend with *2
        strides = base_strides[-len(self.cim_blocks) :]
        while len(strides) < self.num_levels:
            strides.append(strides[-1] * 2)
        return strides[-self.num_levels :]
