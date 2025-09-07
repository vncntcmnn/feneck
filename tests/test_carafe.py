import torch

from feneck import CARAFE
from tests.base_test import BaseNeckTest


class TestCARAFE(BaseNeckTest):
    """Test cases for CARAFE neck."""

    def create_neck(self):
        return CARAFE(
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

    def test_output_shapes(self):
        """Test that CARAFE produces correct output shapes."""
        features = self.get_test_features()
        neck = self.create_neck()

        with torch.no_grad():
            outputs = neck(features)

        # Should have same number of outputs as inputs when num_levels not specified
        assert len(outputs) == 3

        # Check each output has correct channels and spatial dimensions
        expected_shapes = [
            (2, 256, 64, 64),  # stride 8
            (2, 256, 32, 32),  # stride 16
            (2, 256, 16, 16),  # stride 32
        ]

        for output, expected_shape in zip(outputs, expected_shapes, strict=True):
            assert output.shape == expected_shape

    def test_output_strides(self):
        """Test that output strides are correctly computed."""
        neck = CARAFE(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,
        )
        expected_strides = [8, 16, 32, 64, 128]
        assert neck.out_strides == expected_strides
