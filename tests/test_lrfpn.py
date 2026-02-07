import pytest
import torch

from feneck import LRFPN
from tests.base_test import BaseNeckTest


class TestLRFPN(BaseNeckTest):
    """Test cases for LRFPN neck."""

    def create_neck(self):
        return LRFPN(
            in_channels=[256, 512, 1024],
            in_strides=[4, 8, 16],
            out_channels=256,
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 128, 128),  # stride 4 (shallow F1)
            torch.randn(2, 512, 64, 64),  # stride 8 (F2)
            torch.randn(2, 1024, 32, 32),  # stride 16 (F3)
        ]

    def test_insufficient_input_features(self):
        """Test that LRFPN raises error when fewer than 2 input features provided."""
        with pytest.raises(ValueError, match="LR-FPN requires at least 2 input features"):
            LRFPN(
                in_channels=[256],
                in_strides=[4],
                out_channels=256,
            )

    def test_different_num_levels(self):
        """Test LRFPN with different num_levels."""
        features = self.get_test_features()

        neck = LRFPN(
            in_channels=[256, 512, 1024],
            in_strides=[4, 8, 16],
            out_channels=256,
            num_levels=3,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 3
        assert all(out.shape[1] == 256 for out in outputs)

    def test_feature_map_shapes(self):
        """Test that output feature maps have correct spatial dimensions."""
        features = self.get_test_features()
        neck = self.create_neck()

        with torch.no_grad():
            outputs = neck(features)

        # Expected spatial dimensions based on strides: [8, 16, 32, 64]
        # Input is 128x128 at stride 4, so:
        # stride 8 -> 64x64, stride 16 -> 32x32, stride 32 -> 16x16, stride 64 -> 8x8
        expected_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]

        assert len(outputs) == 4
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes, strict=True)):
            assert output.shape[2:] == expected_shape, (
                f"Output {i} should have shape {expected_shape}, got {output.shape[2:]}"
            )
