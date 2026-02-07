from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from feneck._errors import FeatureCountError, InputLevelError
from feneck.base import BaseNeck
from feneck.common_layers import ConvNormLayer


class FeaturePyramidExtender(BaseNeck):
    """Extends and transforms feature pyramids by adding levels and optionally projecting channels.

    This neck can:

    1. Add extra feature levels at higher or lower resolutions
    2. Optionally project all features to uniform channel count
    3. Or do both simultaneously

    Useful for preprocessing before other necks that need specific level counts or channel uniformity, or for
    extending necks like CustomCSPPAN that can't create additional levels themselves.

    Args:
        in_channels: Number of input channels for each feature level
        in_strides: Stride values for each input feature level (must be increasing)
        out_channels: Number of output channels (same for all levels). If None, keeps original channels
        num_levels: Total number of output levels. If None, uses len(in_channels)
        add_higher_res: Whether to add higher resolution levels (lower strides) at the beginning.
            If False, only adds lower resolution levels (higher strides) at the end
        project_channels: Whether to project input features to out_channels. If False, only
            extra levels use out_channels while original features keep their channels
        norm_type: Normalization type applied to convolution layers
        relu_before_downsample: Whether to apply ReLU before downsampling convolutions

    ??? example "Usage Examples"

        ```python
        >>> # Just add levels, keep original channels
        >>> extender = FeaturePyramidExtender(
        ...     in_channels=[256, 512, 1024],
        ...     in_strides=[8, 16, 32],
        ...     out_channels=256,  # Only for new levels
        ...     num_levels=5,
        ...     project_channels=False
        ... )
        >>> # Output channels: [256, 512, 1024, 256, 256]

        >>> # Add levels + unify channels
        >>> extender = FeaturePyramidExtender(
        ...     in_channels=[256, 512, 1024],
        ...     in_strides=[8, 16, 32],
        ...     out_channels=256,
        ...     num_levels=5,
        ...     project_channels=True  # Project all to 256 channels
        ... )
        >>> # Output channels: [256, 256, 256, 256, 256]

        >>> # Just unify channels without adding levels
        >>> extender = FeaturePyramidExtender(
        ...     in_channels=[256, 512, 1024],
        ...     in_strides=[8, 16, 32],
        ...     out_channels=256,
        ...     project_channels=True
        ... )
        >>> # Output channels: [256, 256, 256]

        >>> # Add higher resolution levels
        >>> extender = FeaturePyramidExtender(
        ...     in_channels=[512, 1024],
        ...     in_strides=[16, 32],
        ...     out_channels=256,
        ...     num_levels=4,  # Add P2 (stride 4) and P3 (stride 8)
        ...     add_higher_res=True
        ... )
        >>> # Output strides: [4, 8, 16, 32]
        ```
    """

    def __init__(
        self,
        in_channels: list[int],
        in_strides: list[int],
        out_channels: int = 256,
        num_levels: int | None = None,
        add_higher_res: bool = False,
        project_channels: bool = True,
        norm_type: Literal["batch", "group"] | None = None,
        relu_before_downsample: bool = True,
    ):
        super().__init__(in_channels, in_strides, out_channels)

        # Set default num_levels to input levels if not specified
        if num_levels is None:
            num_levels = len(in_channels)

        if num_levels < len(in_channels):
            raise InputLevelError(f"num_levels ({num_levels}) cannot be less than input levels ({len(in_channels)})")

        self.num_levels = num_levels
        self.add_higher_res = add_higher_res
        self.project_channels = project_channels
        self.relu_before_downsample = relu_before_downsample
        self.extra_levels = num_levels - len(in_channels)

        # Channel projection for input features (optional)
        self.input_convs = nn.ModuleList()
        if project_channels:
            for in_ch in in_channels:
                if in_ch != out_channels:
                    conv = ConvNormLayer(in_ch, out_channels, kernel_size=1, norm_type=norm_type)
                else:
                    conv = nn.Identity()
                self.input_convs.append(conv)
        else:
            # No projection - use identity for all
            self.input_convs.extend([nn.Identity() for _ in in_channels])

        # Extra level convolutions
        self.extra_convs = nn.ModuleList()
        if self.extra_levels > 0:
            if add_higher_res:
                self._build_higher_resolution_layers(in_channels, out_channels, norm_type)
            else:
                self._build_lower_resolution_layers(in_channels, out_channels, norm_type)

    def _build_higher_resolution_layers(
        self, in_channels: list[int], out_channels: int, norm_type: Literal["batch", "group"] | None
    ):
        """Build layers for adding higher resolution levels (upsampling with transposed conv)."""
        for i in range(self.extra_levels):
            # Use first input feature as source for all higher res levels
            if self.project_channels:
                source_channels = out_channels if i > 0 else out_channels  # noqa: RUF034
            else:
                source_channels = in_channels[0] if i == 0 else out_channels

            stride = 2 ** (self.extra_levels - i)

            conv = nn.ConvTranspose2d(
                source_channels,
                out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
                bias=norm_type is None,
            )

            if norm_type == "batch":
                extra_conv = nn.Sequential(conv, nn.BatchNorm2d(out_channels))
            elif norm_type == "group":
                extra_conv = nn.Sequential(conv, nn.GroupNorm(32, out_channels))
            else:
                extra_conv = conv

            self.extra_convs.append(extra_conv)

    def _build_lower_resolution_layers(
        self, in_channels: list[int], out_channels: int, norm_type: Literal["batch", "group"] | None
    ):
        """Build layers for adding lower resolution levels (downsampling with strided conv)."""
        for i in range(self.extra_levels):
            # Use last input feature as source for first extra level, then previous extra levels
            if self.project_channels:
                source_channels = out_channels if i > 0 else out_channels  # noqa: RUF034
            else:
                source_channels = in_channels[-1] if i == 0 else out_channels

            conv = ConvNormLayer(source_channels, out_channels, kernel_size=3, stride=2, norm_type=norm_type)
            self.extra_convs.append(conv)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass adding extra levels and optionally projecting channels."""
        if len(features) != len(self.in_channels):
            raise FeatureCountError(f"Expected {len(self.in_channels)} input features, got {len(features)}")

        # Project input features to target channels (if enabled)
        projected_features = []
        for feature, conv in zip(features, self.input_convs, strict=True):
            projected_features.append(conv(feature))

        if self.extra_levels == 0:
            return projected_features

        if self.add_higher_res:
            return self._forward_with_higher_levels(features, projected_features)
        return self._forward_with_lower_levels(features, projected_features)

    def _forward_with_higher_levels(
        self, features: list[torch.Tensor], projected_features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Forward pass when adding higher resolution levels."""
        output_features = []
        # Use projected feature if channel projection is enabled, otherwise original
        source_feature = projected_features[0] if self.project_channels else features[0]

        for i, conv in enumerate(self.extra_convs):
            extra_feature = conv(source_feature) if i == 0 else conv(output_features[-1])
            output_features.append(extra_feature)

        # Add original features after extra higher-res levels
        output_features.extend(projected_features)
        return output_features

    def _forward_with_lower_levels(
        self, features: list[torch.Tensor], projected_features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Forward pass when adding lower resolution levels."""
        # Add original features first
        output_features = projected_features.copy()

        # Use projected feature if channel projection is enabled, otherwise original
        source_feature = projected_features[-1] if self.project_channels else features[-1]

        for i, conv in enumerate(self.extra_convs):
            if i == 0:
                if self.relu_before_downsample:
                    source_feature = F.relu(source_feature)
                extra_feature = conv(source_feature)
            else:
                source_feature = output_features[-1]
                if self.relu_before_downsample:
                    source_feature = F.relu(source_feature)
                extra_feature = conv(source_feature)

            output_features.append(extra_feature)

        return output_features

    @property
    @torch.jit.unused
    def out_channels(self) -> list[int]:
        """Output channels for each feature level."""
        if self.project_channels:
            # All levels have same out_channels
            return [self._out_channels] * self.num_levels
        else:
            # Original levels keep their channels, extra levels use out_channels
            channels = self.in_channels.copy()
            for _ in range(self.extra_levels):
                channels.append(self._out_channels)
            if self.add_higher_res:
                # Extra levels come first when adding higher resolution
                extra_channels = [self._out_channels] * self.extra_levels
                return extra_channels + self.in_channels
            return channels

    @property
    @torch.jit.unused
    def out_strides(self) -> list[int]:
        """Output strides for each feature level."""
        if self.add_higher_res:
            return self._compute_higher_res_strides()
        return self._compute_lower_res_strides()

    def _compute_higher_res_strides(self) -> list[int]:
        """Compute output strides when adding higher resolution levels."""
        extra_strides = []
        base_stride = self.in_strides[0]
        for i in range(self.extra_levels):
            stride = base_stride // (2 ** (self.extra_levels - i))
            extra_strides.append(stride)
        return extra_strides + self.in_strides

    def _compute_lower_res_strides(self) -> list[int]:
        """Compute output strides when adding lower resolution levels."""
        extra_strides = []
        for i in range(self.extra_levels):
            stride = self.in_strides[-1] * (2 ** (i + 1))
            extra_strides.append(stride)
        return self.in_strides + extra_strides
