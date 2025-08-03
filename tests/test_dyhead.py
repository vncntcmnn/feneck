import pytest
import torch

from feneck import DyHead
from tests.base_test import BaseNeckTest


class TestDyHead(BaseNeckTest):
    """Test cases for DyHead neck."""

    def create_neck(self):
        return DyHead(
            in_channels=[256, 256, 256],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_blocks=2,  # Fewer blocks for faster testing
            norm_type="group",
        )

    def get_test_features(self):
        return [
            torch.randn(2, 256, 64, 64),  # stride 8
            torch.randn(2, 256, 32, 32),  # stride 16
            torch.randn(2, 256, 16, 16),  # stride 32
        ]

    def test_channel_requirement(self):
        """Test that DyHead requires same input channels for all levels."""
        with pytest.raises(ValueError, match="DyHead requires all input levels to have the same number of channels"):
            DyHead(
                in_channels=[128, 256, 512],
                in_strides=[8, 16, 32],
                out_channels=256,
            )

    @pytest.mark.parametrize("num_blocks,norm_type", [(1, None), (6, "batch")])
    def test_different_configurations(self, num_blocks: int, norm_type: str | None):
        """Test different DyHead configurations."""
        features = self.get_test_features()

        neck = DyHead(
            in_channels=[256, 256, 256],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_blocks=num_blocks,
            norm_type=norm_type,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == len(features)
        assert all(out.shape[1] == 256 for out in outputs)

        # Check spatial dimensions preserved
        for input_feat, output_feat in zip(features, outputs, strict=True):
            assert input_feat.shape[2:] == output_feat.shape[2:]
