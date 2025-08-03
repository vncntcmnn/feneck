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

## Quick Start

```python
import torch

from feneck import FPN, CustomCSPPAN, SimpleFPN

# Standard FPN for hierarchical backbones (ResNet, etc.)
fpn = FPN(in_channels=[256, 512, 1024, 2048], in_strides=[4, 8, 16, 32], out_channels=256)

# SimpleFPN for transformer backbones (ViT, etc.)
simple_fpn = SimpleFPN(in_channels=768, in_strides=16, out_channels=256, num_levels=5, start_level=2)

# Custom CSP-PAN with transformer enhancement
csp_pan = CustomCSPPAN(
    in_channels=[256, 512, 1024, 2048], in_strides=[4, 8, 16, 32], out_channels=256, use_transformer=True
)

# Forward pass
backbone_features = [
    torch.randn(1, c, 64 // (s // 4), 64 // (s // 4))
    for c, s in zip(
        [256, 512, 1024, 2048],
        [4, 8, 16, 32],
        strict=True,
    )
]
pyramid_features = fpn(backbone_features)
```

## Architecture Support

- **Hierarchical backbones**: ResNet, RegNet, EfficientNet → use `FPN` or `CustomCSPPAN`
- **Plain Transformer backbones**: Vision Transformer → use `SimpleFPN`

## Requirements

- Python ≥ 3.10
- PyTorch

## License

Apache License 2.0

Some implementations are adapted from other repositories (PaddleDetection) under compatible licenses.
