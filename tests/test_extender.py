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

    def test_add_lower_levels_with_projection(self):
        """Test adding lower resolution levels with channel projection enabled."""
        features = self.get_test_features()

        neck = FeaturePyramidExtender(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,
            project_channels=True,
        )

        with torch.no_grad():
            outputs = neck(features)

        # Should have 5 output levels
        assert len(outputs) == 5

        # All outputs should have 256 channels (projected)
        assert all(out.shape[1] == 256 for out in outputs)

        # Check spatial dimensions
        assert outputs[0].shape[2:] == (64, 64)  # stride 8
        assert outputs[1].shape[2:] == (32, 32)  # stride 16
        assert outputs[2].shape[2:] == (16, 16)  # stride 32
        assert outputs[3].shape[2:] == (8, 8)  # stride 64
        assert outputs[4].shape[2:] == (4, 4)  # stride 128

        # Verify out_channels property
        assert neck.out_channels == [256, 256, 256, 256, 256]

    def test_add_lower_levels_without_projection(self):
        """Test adding lower resolution levels without channel projection."""
        features = self.get_test_features()

        neck = FeaturePyramidExtender(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=5,
            project_channels=False,
        )

        with torch.no_grad():
            outputs = neck(features)

        # Should have 5 output levels
        assert len(outputs) == 5

        # Original features keep their channels
        assert outputs[0].shape[1] == 256
        assert outputs[1].shape[1] == 512
        assert outputs[2].shape[1] == 1024

        # Extra levels use out_channels
        assert outputs[3].shape[1] == 256
        assert outputs[4].shape[1] == 256

        # Verify out_channels property
        assert neck.out_channels == [256, 512, 1024, 256, 256]

    def test_add_higher_levels_with_projection(self):
        """Test adding higher resolution levels with channel projection enabled."""
        features = [
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

        neck = FeaturePyramidExtender(
            in_channels=[512, 1024],
            in_strides=[16, 32],
            out_channels=256,
            num_levels=4,
            add_higher_res=True,
            project_channels=True,
        )

        with torch.no_grad():
            outputs = neck(features)

        # Should have 4 output levels
        assert len(outputs) == 4

        # All outputs should have 256 channels (projected)
        assert all(out.shape[1] == 256 for out in outputs)

        # Check spatial dimensions (approximate due to transposed conv)
        assert outputs[0].shape[2] >= 56  # stride 4 (from 32x32 upsampled)
        assert outputs[1].shape[2] >= 28  # stride 8
        assert outputs[2].shape[2:] == (32, 32)  # stride 16 (original)
        assert outputs[3].shape[2:] == (16, 16)  # stride 32 (original)

        # Verify out_channels property
        assert neck.out_channels == [256, 256, 256, 256]

        # Verify out_strides property
        assert neck.out_strides == [4, 8, 16, 32]

    def test_add_higher_levels_without_projection(self):
        """Test adding higher resolution levels without channel projection."""
        features = [
            torch.randn(2, 512, 32, 32),  # stride 16
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

        neck = FeaturePyramidExtender(
            in_channels=[512, 1024],
            in_strides=[16, 32],
            out_channels=256,
            num_levels=4,
            add_higher_res=True,
            project_channels=False,
        )

        with torch.no_grad():
            outputs = neck(features)

        # Should have 4 output levels
        assert len(outputs) == 4

        # Extra higher-res levels use out_channels
        assert outputs[0].shape[1] == 256
        assert outputs[1].shape[1] == 256

        # Original features keep their channels
        assert outputs[2].shape[1] == 512
        assert outputs[3].shape[1] == 1024

        # Verify out_channels property
        assert neck.out_channels == [256, 256, 512, 1024]

        # Verify out_strides property
        assert neck.out_strides == [4, 8, 16, 32]

    def test_multiple_extra_levels_lower_res(self):
        """Test that multiple extra lower resolution levels chain correctly."""
        features = self.get_test_features()

        neck = FeaturePyramidExtender(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=6,  # Add 3 extra levels
            project_channels=True,
        )

        # Verify we have the right number of extra convolutions
        assert len(neck.extra_convs) == 3

        with torch.no_grad():
            outputs = neck(features)

        # Should have 6 output levels
        assert len(outputs) == 6

        # All should have uniform channels
        assert all(out.shape[1] == 256 for out in outputs)

        # Verify strides
        assert neck.out_strides == [8, 16, 32, 64, 128, 256]

    def test_multiple_extra_levels_higher_res(self):
        """Test that multiple extra higher resolution levels chain correctly."""
        features = [
            torch.randn(2, 1024, 16, 16),  # stride 32
        ]

        neck = FeaturePyramidExtender(
            in_channels=[1024],
            in_strides=[32],
            out_channels=256,
            num_levels=4,  # Add 3 extra higher-res levels
            add_higher_res=True,
            project_channels=True,
        )

        # Verify we have the right number of extra convolutions
        assert len(neck.extra_convs) == 3

        with torch.no_grad():
            outputs = neck(features)

        # Should have 4 output levels
        assert len(outputs) == 4

        # All should have uniform channels
        assert all(out.shape[1] == 256 for out in outputs)

        # Verify strides
        assert neck.out_strides == [4, 8, 16, 32]

    def test_no_extra_levels(self):
        """Test behavior when num_levels equals input levels (no extra levels)."""
        features = self.get_test_features()

        neck = FeaturePyramidExtender(
            in_channels=[256, 512, 1024],
            in_strides=[8, 16, 32],
            out_channels=256,
            num_levels=None,  # Should default to len(in_channels)
            project_channels=True,
        )

        assert neck.extra_levels == 0
        assert len(neck.extra_convs) == 0

        with torch.no_grad():
            outputs = neck(features)

        # Should have same number of levels as input
        assert len(outputs) == 3

        # All should be projected to out_channels
        assert all(out.shape[1] == 256 for out in outputs)
