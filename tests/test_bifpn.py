import pytest
import torch

from feneck import BiFPN
from tests.base_test import BaseNeckTest


class TestBiFPN(BaseNeckTest):
    """Test cases for BiFPN neck."""

    def create_neck(self):
        return BiFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,
            num_layers=2,  # Fewer layers for faster testing
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

    def test_too_many_input_levels(self):
        """Test that BiFPN raises error when too many input levels provided."""
        with pytest.raises(ValueError, match="Too many input levels"):
            BiFPN(
                in_channels=[128, 256, 512, 1024, 2048, 4096],
                in_strides=[4, 8, 16, 32, 64, 128],
                out_channels=256,
                num_levels=5,
            )

    @pytest.mark.parametrize("num_levels,num_layers", [(3, 1), (5, 3), (6, 2)])
    def test_different_configurations(self, num_levels: int, num_layers: int):
        """Test different BiFPN configurations."""
        features = [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 512, 32, 32),  # stride 16
        ]

        neck = BiFPN(
            in_channels=[256, 512],
            in_strides=[8, 16],
            out_channels=256,
            num_levels=num_levels,
            num_layers=num_layers,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == num_levels
        assert all(out.shape[1] == 256 for out in outputs)

    def test_fast_attention_disabled(self):
        """Test BiFPN with fast attention disabled."""
        features = self.get_test_features()

        neck = BiFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            use_fast_attention=False,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 5
        assert all(out.shape[1] == 256 for out in outputs)

    def test_output_strides(self):
        """Test that output strides are correctly computed."""
        neck = self.create_neck()
        expected_strides = [8, 16, 32, 64, 128]
        assert neck.out_strides == expected_strides
