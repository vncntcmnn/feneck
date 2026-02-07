import torch

from feneck import SimpleFPN
from tests.base_test import BaseNeckTest


class TestSimpleFPN(BaseNeckTest):
    """Test cases for SimpleFPN neck."""

    def create_neck(self):
        return SimpleFPN(
            in_channels=768,
            in_strides=16,
            out_channels=256,
            num_levels=5,
            start_level=2,
        )

    def get_test_features(self):
        # ViT-like feature at stride 16
        return [torch.randn(2, 768, 14, 14)]
