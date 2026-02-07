import pytest
import torch

from feneck import NASFPN
from tests.base_test import BaseNeckTest


class TestNASFPN(BaseNeckTest):
    """Test cases for NASFPN neck."""

    def create_neck(self):
        return NASFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_repeats=2,  # Fewer repeats for faster testing
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8 (P3)
            torch.randn(2, 512, 32, 32),  # stride 16 (P4)
            torch.randn(2, 1024, 16, 16),  # stride 32 (P5)
        ]

    def test_insufficient_input_levels(self):
        """Test that NASFPN raises error when too few input levels provided."""
        with pytest.raises(ValueError, match="Expected at least 3 input levels"):
            NASFPN(
                in_channels=[256, 512],
                in_strides=[8, 16],
                out_channels=256,
            )

    def test_always_five_outputs(self):
        """Test that NASFPN always outputs 5 pyramid levels with correct spatial dimensions."""
        features = self.get_test_features()
        neck = self.create_neck()

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 5  # Always P3-P7
        assert all(out.shape[1] == 256 for out in outputs)

        # Check spatial dimensions: P3=64x64, P4=32x32, P5=16x16, P6=8x8, P7=4x4
        expected_sizes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
        for i, (output, expected_size) in enumerate(zip(outputs, expected_sizes, strict=True)):
            assert output.shape[2:] == expected_size, (
                f"P{i + 3} should have size {expected_size}, got {output.shape[2:]}"
            )

    def test_output_strides(self):
        """Test that output strides are P3-P7."""
        neck = self.create_neck()
        expected_strides = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7
        assert neck.out_strides == expected_strides

    @pytest.mark.parametrize("num_repeats", [1, 3])
    def test_different_repeats(self, num_repeats: int):
        """Test different numbers of NAS-FPN cell repeats."""
        features = self.get_test_features()

        neck = NASFPN(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_repeats=num_repeats,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 5
        assert all(out.shape[1] == 256 for out in outputs)
