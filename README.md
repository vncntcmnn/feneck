# 🦊 FE-Neck

Feature extraction neck modules for computer vision models.

## Installation

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

| Neck                       | Description                            | Use Case                        |
|----------------------------|----------------------------------------|---------------------------------|
| **FPN**                    | Classic Feature Pyramid Network        | Standard multi-scale detection  |
| **BiFPN**                  | Bidirectional FPN with weighted fusion | Efficient multi-scale fusion    |
| **NASFPN**                 | Neural Architecture Search FPN         | Learned fusion patterns         |
| **SimpleFPN**              | FPN for transformer backbones          | Single-scale to multi-scale     |
| **CustomCSPPAN**           | CSP-PAN with transformer enhancement   | Advanced feature aggregation    |
| **HRFPN**                  | High-Resolution FPN                    | Multi-scale aggregation         |
| **LRFPN**                  | Location-Refined FPN                   | Remote sensing object detection |
| **DyHead**                 | Dynamic Head with attention            | Post-FPN refinement             |
| **FeaturePyramidExtender** | Level/channel preprocessing            | Backbone adaptation             |

## Quick Examples

### Standard Hierarchical Backbones
```python
import torch
from feneck import FPN, BiFPN, NASFPN

# Classic FPN
fpn = FPN(
    in_channels=[256, 512, 1024, 2048],
    in_strides=[4, 8, 16, 32],
    out_channels=256
)

# Efficient BiFPN
bifpn = BiFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)

# NAS-discovered architecture
nasfpn = NASFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

### Transformer Backbones
```python
from feneck import SimpleFPN

# For Vision Transformers
simple_fpn = SimpleFPN(
    in_channels=768,
    in_strides=16,
    out_channels=256,
    start_level=2
)
```

### Specialized Applications
```python
from feneck import LRFPN

# For remote sensing object detection
lrfpn = LRFPN(
    in_channels=[256, 512, 1024],  # shallow, F2, F3
    in_strides=[4, 8, 16],
    out_channels=256
)
```

### Advanced Features
```python
from feneck import CustomCSPPAN, DyHead, FeaturePyramidExtender

# CSP-PAN with transformer
csp_pan = CustomCSPPAN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256,
    use_transformer=True
)

# Post-FPN refinement (requires uniform channels)
dyhead = DyHead(
    in_channels=[256, 256, 256],
    in_strides=[8, 16, 32],
    out_channels=256
)

# Preprocessing: extend levels + unify channels
extender = FeaturePyramidExtender(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256,
    num_levels=5
)
```

## Architecture Compatibility

| Backbone Type                    | Recommended Necks                              |
|----------------------------------|------------------------------------------------|
| **ResNet, RegNet, EfficientNet** | FPN, BiFPN, NASFPN, CustomCSPPAN, HRFPN, LRFPN |
| **Vision Transformer**           | SimpleFPN                                      |
| **Any backbone**                 | FeaturePyramidExtender (preprocessing)         |
| **Post-FPN processing**          | DyHead                                         |

## Forward Pass Example

```python
# Typical backbone features
backbone_features = [
    torch.randn(1, 256, 64, 64),    # stride 4
    torch.randn(1, 512, 32, 32),    # stride 8
    torch.randn(1, 1024, 16, 16),   # stride 16
    torch.randn(1, 2048, 8, 8),     # stride 32
]

pyramid_features = fpn(backbone_features)
# Output: 5 levels with 256 channels each
```

## Requirements

- Python ≥ 3.10
- PyTorch

## License

Apache License 2.0

Some implementations adapted from PaddleDetection, Microsoft Research, Google Research under compatible licenses.