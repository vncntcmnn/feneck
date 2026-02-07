# 🦊 FE-Neck

[![Release](https://img.shields.io/github/v/release/vncntcmnn/feneck)](https://img.shields.io/github/v/release/vncntcmnn/feneck)
[![Build status](https://img.shields.io/github/actions/workflow/status/vncntcmnn/feneck/main.yml?branch=main)](https://github.com/vncntcmnn/feneck/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vncntcmnn/feneck/branch/main/graph/badge.svg)](https://codecov.io/gh/vncntcmnn/feneck)
[![Commit activity](https://img.shields.io/github/commit-activity/m/vncntcmnn/feneck)](https://img.shields.io/github/commit-activity/m/vncntcmnn/feneck)
[![License](https://img.shields.io/github/license/vncntcmnn/feneck)](https://img.shields.io/github/license/vncntcmnn/feneck)

Feature extraction neck modules for computer vision models. This library provides a comprehensive collection of feature pyramid network architectures for multi-scale feature fusion in object detection, segmentation, and other computer vision tasks.

## Installation

```bash
# Clone the repository
git clone https://github.com/vncntcmnn/feneck.git
cd feneck

# Install dependencies (CPU version)
uv sync --extra cpu

# Or with CUDA support
uv sync --extra cu118  # CUDA 11.8
uv sync --extra cu124  # CUDA 12.4
```

## Quick Start

All necks can be imported directly from the `feneck` package:

```python
import torch
from feneck import FPN, PAFPN, BiFPN

# Create a Feature Pyramid Network
fpn = FPN(
    in_channels=[256, 512, 1024, 2048],
    in_strides=[4, 8, 16, 32],
    out_channels=256
)

# Forward pass with backbone features
backbone_features = [
    torch.randn(1, 256, 64, 64),   # stride 4
    torch.randn(1, 512, 32, 32),   # stride 8
    torch.randn(1, 1024, 16, 16),  # stride 16
    torch.randn(1, 2048, 8, 8),    # stride 32
]

pyramid_features = fpn(backbone_features)
# Output: 5 levels with 256 channels each
```

## Available Modules

### Standard FPN Variants

#### FPN - Feature Pyramid Network
Classic top-down architecture with lateral connections for multi-scale feature fusion.

```python
from feneck import FPN

fpn = FPN(
    in_channels=[256, 512, 1024, 2048],
    in_strides=[4, 8, 16, 32],
    out_channels=256
)
```

**Use Case:** Standard multi-scale object detection

---

#### PAFPN - Path Aggregation FPN
Enhances FPN with an additional bottom-up pathway for better feature propagation.

```python
from feneck import PAFPN

pafpn = PAFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** Enhanced feature fusion for detection tasks

---

#### BiFPN - Bidirectional Feature Pyramid Network
Efficient bidirectional cross-scale connections with learnable weights.

```python
from feneck import BiFPN

bifpn = BiFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** Efficient multi-scale fusion with weighted connections

---

#### NASFPN - Neural Architecture Search FPN
Feature pyramid architecture discovered through neural architecture search.

```python
from feneck import NASFPN

nasfpn = NASFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** Learned fusion patterns for optimal performance

---

### Specialized Architectures

#### SimpleFPN
Designed for transformer backbones that output single-scale features.

```python
from feneck import SimpleFPN

simple_fpn = SimpleFPN(
    in_channels=768,        # ViT output channels
    in_strides=16,          # ViT patch size
    out_channels=256,
    start_level=2
)
```

**Use Case:** Converting single-scale Vision Transformer outputs to multi-scale features

---

#### CustomCSPPAN
CSP-PAN architecture with optional transformer enhancement.

```python
from feneck import CustomCSPPAN

csp_pan = CustomCSPPAN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256,
    use_transformer=True
)
```

**Use Case:** Advanced feature aggregation with attention mechanisms

---

#### HRFPN - High-Resolution FPN
Maintains high-resolution representations throughout the network.

```python
from feneck import HRFPN

hrfpn = HRFPN(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** Tasks requiring fine-grained spatial details

---

#### LRFPN - Location-Refined FPN
Specialized for remote sensing and aerial image object detection.

```python
from feneck import LRFPN

lrfpn = LRFPN(
    in_channels=[256, 512, 1024],  # shallow, F2, F3
    in_strides=[4, 8, 16],
    out_channels=256
)
```

**Use Case:** Remote sensing object detection with location refinement

---

### Feature Enhancement Modules

#### CARAFE - Content-Aware ReAssembly of FEatures
Content-aware upsampling for better feature reconstruction.

```python
from feneck import CARAFE

carafe = CARAFE(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** High-quality feature upsampling with content awareness

---

#### DyHead - Dynamic Head
Post-FPN refinement with scale-aware, spatial-aware, and task-aware attention.

```python
from feneck import DyHead

# Requires uniform input channels
dyhead = DyHead(
    in_channels=[256, 256, 256],
    in_strides=[8, 16, 32],
    out_channels=256
)
```

**Use Case:** Post-processing FPN features with dynamic attention

---

### Utility Modules

#### FeaturePyramidExtender
Preprocesses backbone features by extending pyramid levels and unifying channels.

```python
from feneck import FeaturePyramidExtender

extender = FeaturePyramidExtender(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256,
    num_levels=5  # Extend to 5 pyramid levels
)
```

**Use Case:** Adapting backbone outputs for specific neck requirements

---

## Architecture Compatibility

| Backbone Type | Recommended Necks |
|---------------|-------------------|
| **ResNet, RegNet, EfficientNet** | FPN, PAFPN, BiFPN, NASFPN, CustomCSPPAN, HRFPN, LRFPN, CARAFE |
| **Vision Transformer (ViT, Swin)** | SimpleFPN |
| **Any backbone** | FeaturePyramidExtender (preprocessing) |
| **Post-FPN processing** | DyHead |

## Common Patterns

### Hierarchical Backbones (ResNet, etc.)
```python
from feneck import FPN

# Standard FPN usage
fpn = FPN(
    in_channels=[256, 512, 1024, 2048],  # C2, C3, C4, C5
    in_strides=[4, 8, 16, 32],
    out_channels=256
)
```

### Transformer Backbones
```python
from feneck import SimpleFPN

# Convert single-scale to multi-scale
simple_fpn = SimpleFPN(
    in_channels=768,
    in_strides=16,
    out_channels=256,
    start_level=2
)
```

### Extending Pyramid Levels
```python
from feneck import FeaturePyramidExtender, PAFPN

# Preprocess then apply neck
extender = FeaturePyramidExtender(
    in_channels=[256, 512, 1024],
    in_strides=[8, 16, 32],
    out_channels=256,
    num_levels=5
)

pafpn = PAFPN(
    in_channels=[256, 256, 256, 256, 256],
    in_strides=[4, 8, 16, 32, 64],
    out_channels=256
)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0

## License

Apache License 2.0

Some implementations adapted from:
- PaddleDetection (Apache 2.0)
- Microsoft Research
- Google Research

## Citation

If you use this library in your research, please cite:

```bibtex
@software{feneck2025,
  author = {Camenen, Vincent},
  title = {FE-Neck: Feature Extraction Neck Modules},
  year = {2025},
  url = {https://github.com/vncntcmnn/feneck}
}
```
