import pytest
import torch

from feneck import PAFPN
from tests.base_test import BaseNeckTest


class TestPAFPN(BaseNeckTest):
    """Test cases for PAFPN neck."""

    def create_neck(self):
        return PAFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

    def test_num_levels_validation(self):
        """Test that PAFPN validates num_levels correctly."""
        with pytest.raises(ValueError, match=r"num_levels .* cannot be less than input levels"):
            PAFPN(
                in_channels=[256, 512, 1024],
                in_strides=[8, 16, 32],
                out_channels=256,
                num_levels=2,
            )

    def test_architecture_components(self):
        """Test that PAFPN has the correct number of components."""
        neck = PAFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=3,
        )

        # Check component counts
        assert len(neck.lateral_convs) == 3
        assert len(neck.fpn_convs) == 3
        assert len(neck.downsample_convs) == 2
        assert len(neck.pafpn_convs) == 2

    def test_output_strides(self):
        """Test that output strides are correctly computed."""
        neck = self.create_neck()
        expected_strides = [8, 16, 32, 64, 128]
        assert neck.out_strides == expected_strides
