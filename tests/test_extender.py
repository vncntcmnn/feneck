import pytest
import torch

from feneck import FeaturePyramidExtender
from tests.base_test import BaseNeckTest


class TestFeaturePyramidExtender(BaseNeckTest):
    """Test cases for FeaturePyramidExtender neck."""

    def create_neck(self):
        return FeaturePyramidExtender(
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
        """Test that FeaturePyramidExtender validates num_levels correctly."""
        with pytest.raises(ValueError, match=r"num_levels .* cannot be less than input levels"):
            FeaturePyramidExtender(
                in_channels=[256, 512, 1024],
                in_strides=[8, 16, 32],
                out_channels=256,
                num_levels=2,
            )

    @pytest.mark.parametrize("project_channels", [True, False])
    def test_channel_projection(self, project_channels: bool):
        """Test channel projection behavior."""
        features = self.get_test_features()

        neck = FeaturePyramidExtender(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=128,
            project_channels=project_channels,
        )

        with torch.no_grad():
            outputs = neck(features)

        assert len(outputs) == 3
        if project_channels:
            assert all(out.shape[1] == 128 for out in outputs)
        else:
            assert outputs[0].shape[1] == 256
            assert outputs[1].shape[1] == 512
            assert outputs[2].shape[1] == 1024

    def test_output_strides(self):
        """Test that output strides are correctly computed."""
        neck = self.create_neck()
        expected_strides = [8, 16, 32, 64, 128]
        assert neck.out_strides == expected_strides
