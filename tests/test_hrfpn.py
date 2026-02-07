import pytest
import torch

from feneck import HRFPN
from tests.base_test import BaseNeckTest


class TestHRFPN(BaseNeckTest):
    """Test cases for HRFPN neck."""

    def create_neck(self):
        return HRFPN(
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

    def test_too_many_output_levels(self):
        """Test that HRFPN raises error when requesting too many output levels."""
        with pytest.raises(ValueError, match=r"Cannot generate .* output levels from .* input levels"):
            HRFPN(
                in_channels=[256, 512],
                in_strides=[8, 16],
                out_channels=256,
                num_levels=6,  # Too many levels for 2 inputs
            )

    @pytest.mark.parametrize("pooling_type", ["max", "avg"])
    def test_pooling_types(self, pooling_type: str):
        """Test different pooling types."""
        features = self.get_test_features()

        neck = HRFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            pooling_type=pooling_type,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 5
        assert all(out.shape[1] == 256 for out in outputs)
