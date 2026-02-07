# 🦊 FE-Neck

[![Release](https://img.shields.io/github/v/release/vncntcmnn/feneck)](https://img.shields.io/github/v/release/vncntcmnn/feneck)
[![Build status](https://img.shields.io/github/actions/workflow/status/vncntcmnn/feneck/main.yml?branch=main)](https://github.com/vncntcmnn/feneck/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vncntcmnn/feneck/branch/main/graph/badge.svg)](https://codecov.io/gh/vncntcmnn/feneck)
[![Commit activity](https://img.shields.io/github/commit-activity/m/vncntcmnn/feneck)](https://img.shields.io/github/commit-activity/m/vncntcmnn/feneck)
[![License](https://img.shields.io/github/license/vncntcmnn/feneck)](https://img.shields.io/github/license/vncntcmnn/feneck)

Feature extraction neck modules for computer vision models.

- **Github repository**: <https://github.com/vncntcmnn/feneck/>
- **Documentation**: <https://vncntcmnn.github.io/feneck/>

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

```python
import torch
from feneck import FPN, PAFPN, BiFPN

# Classic FPN
fpn = FPN(
    in_channels=[256, 512, 1024, 2048],
    in_strides=[4, 8, 16, 32],
    out_channels=256
)

# Backbone features
features = [
    torch.randn(1, 256, 64, 64),   # stride 4
    torch.randn(1, 512, 32, 32),   # stride 8
    torch.randn(1, 1024, 16, 16),  # stride 16
    torch.randn(1, 2048, 8, 8),    # stride 32
]

# Forward pass
pyramid_features = fpn(features)
```

## Available Necks

| Module | Description | Best For |
|--------|-------------|----------|
| **FPN** | Feature Pyramid Network | Standard multi-scale detection |
| **PAFPN** | Path Aggregation FPN | Enhanced feature fusion |
| **BiFPN** | Bidirectional FPN | Efficient multi-scale fusion |
| **NASFPN** | NAS-discovered FPN | Learned fusion patterns |
| **SimpleFPN** | FPN for transformers | Single-scale to multi-scale |
| **CustomCSPPAN** | CSP-PAN + transformer | Advanced aggregation |
| **HRFPN** | High-Resolution FPN | Multi-scale aggregation |
| **LRFPN** | Location-Refined FPN | Remote sensing detection |
| **CARAFE** | Content-Aware upsampling | Adaptive feature reassembly |
| **DyHead** | Dynamic Head | Post-FPN refinement |
| **FeaturePyramidExtender** | Preprocessing utility | Backbone adaptation |

See the [documentation](https://vncntcmnn.github.io/feneck/) for detailed usage examples and API reference.

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
make install

# Run pre-commit hooks
uv run pre-commit run -a
```

### Running Tests

```bash
# Run tests
make test

# Run tests with coverage
make test-cov
```

## Requirements

- Python >= 3.10
- PyTorch

## License

Apache License 2.0

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
