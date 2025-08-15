# 🦊 FE-Neck

Feature extraction neck modules for computer vision models.

## Installation

Clone the repository and install with uv:

```bash
git clone <repository-url>
cd feneck

# CPU version
uv sync --extra cpu

# CUDA 11.8
uv sync --extra cu118

# CUDA 12.4
uv sync --extra cu124
```

## Available Necks

- **FPN**: Classic Feature Pyramid Network
- **PAFPN**: Path Aggregation FPN with bottom-up augmentation
- **BiFPN**: Bidirectional FPN with weighted feature fusion
- **SimpleFPN**: FPN for single-scale transformer backbones
- **CustomCSPPAN**: CSP-PAN with optional transformer enhancement
- **HRFPN**: High-Resolution FPN for multi-scale aggregation
- **DyHead**: Dynamic Head with attention mechanisms
- **FeaturePyramidExtender**: Extends pyramids by adding levels and/or unifying channels

## Quick Start

```python
import torch

from feneck import FPN, PAFPN, CustomCSPPAN, SimpleFPN, BiFPN, HRFPN, DyHead, FeaturePyramidExtender

# Standard FPN for hierarchical backbones (ResNet, etc.)
fpn = FPN(in_channels=[256, 512, 1024, 2048], in_strides=[4, 8, 16, 32], out_channels=256, num_levels=5)

# PAFPN for enhanced feature fusion with bottom-up augmentation
pafpn = PAFPN(in_channels=[256, 512, 1024], in_strides=[8, 16, 32], out_channels=256, num_levels=5)

# BiFPN for efficient multi-scale fusion
bifpn = BiFPN(in_channels=[256, 512, 1024], in_strides=[8, 16, 32], out_channels=256, num_levels=5)

# SimpleFPN for transformer backbones (ViT, etc.)
simple_fpn = SimpleFPN(in_channels=768, in_strides=16, out_channels=256, num_levels=5, start_level=2)

# Custom CSP-PAN with transformer enhancement
csp_pan = CustomCSPPAN(
    in_channels=[256, 512, 1024, 2048], in_strides=[4, 8, 16, 32], out_channels=256, use_transformer=True
)

# HRFPN for high-resolution representation learning
hrfpn = HRFPN(in_channels=[256, 512, 1024], in_strides=[8, 16, 32], out_channels=256, num_levels=5)

# DyHead for attention-based feature refinement (requires same input channels)
dyhead = DyHead(in_channels=[256, 256, 256], in_strides=[8, 16, 32], out_channels=256, num_blocks=6)

# FeaturePyramidExtender for preprocessing (extend levels + unify channels)
extender = FeaturePyramidExtender(
    in_channels=[256, 512, 1024], in_strides=[8, 16, 32], out_channels=256, num_levels=5
)

# Forward pass example
backbone_features = [
    torch.randn(1, c, 64 // (s // 4), 64 // (s // 4))
    for c, s in zip([256, 512, 1024, 2048], [4, 8, 16, 32], strict=True)
]
pyramid_features = fpn(backbone_features)
```

## Architecture Support

- **Hierarchical backbones**: ResNet, RegNet, EfficientNet → use `FPN`, `PAFPN`, `BiFPN`, `CustomCSPPAN`, `HRFPN`
- **Plain Transformer backbones**: Vision Transformer → use `SimpleFPN`
- **Post-FPN refinement**: Any FPN output → use `DyHead` (requires uniform channels)
- **Preprocessing**: Any backbone → use `FeaturePyramidExtender` for level/channel preparation

## Requirements

- Python ≥ 3.10
- PyTorch

## License

Apache License 2.0

Some implementations are adapted from other repositories (PaddleDetection, Microsoft Research, Google Research) under compatible licenses.
