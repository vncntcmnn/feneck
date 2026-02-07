import torch

from feneck import CustomCSPPAN
from tests.base_test import BaseNeckTest


class TestCustomCSPPAN(BaseNeckTest):
    """Test cases for CustomCSPPAN neck."""

    def create_neck(self):
        return CustomCSPPAN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]
