import pytest
import torch

from feneck import FPN
from tests.base_test import BaseNeckTest


class TestFPN(BaseNeckTest):
    """Test cases for FPN neck."""

    def create_neck(self):
        return FPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,  # Updated to use num_levels
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

    def test_num_levels_validation(self):
        """Test that FPN validates num_levels correctly."""
        with pytest.raises(ValueError, match=r"num_levels .* cannot be less than input levels"):
            FPN(
                in_channels=[256, 512, 1024],
                in_strides=[8, 16, 32],
                out_channels=256,
                num_levels=2,  # Less than 3 input levels
            )

    def test_default_num_levels(self):
        """Test that FPN uses input levels as default when num_levels not specified."""
        features = self.get_test_features()

        neck = FPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            # num_levels not specified - should default to 3
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 3  # Same as input levels
        assert neck.num_levels == 3
        assert neck.extra_levels == 0

    @pytest.mark.parametrize("num_levels,has_extra_convs", [(3, False), (5, True), (4, False)])
    def test_different_num_levels(self, num_levels: int, has_extra_convs: bool):
        """Test FPN with different num_levels configurations."""
        features = self.get_test_features()

        neck = FPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=num_levels,
            has_extra_convs=has_extra_convs,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == num_levels
        assert all(out.shape[1] == 256 for out in outputs)
        assert neck.extra_levels == num_levels - 3

    def test_output_strides(self):
        """Test that output strides are correctly computed."""
        neck = self.create_neck()
        expected_strides = [8, 16, 32, 64, 128]  # 3 input + 2 extra levels
        assert neck.out_strides == expected_strides
