from abc import ABC, abstractmethod

import torch


class BaseNeckTest(ABC):
    """Base test case for neck modules."""

    @abstractmethod
    def create_neck(self):
        """Create neck instance for testing."""

    @abstractmethod
    def get_test_features(self):
        """Get test features for forward pass."""

    def test_forward_pass(self):
        """Test basic forward pass functionality."""
        neck = self.create_neck()
        features = self.get_test_features()

        with torch.no_grad():
            outputs = neck(features)

        assert isinstance(outputs, list)
        assert len(outputs) > 0
        assert all(isinstance(output, torch.Tensor) for output in outputs)
        assert all(output.dim() == 4 for output in outputs)  # NCHW format

    def test_output_properties(self):
        """Test output channels and strides properties."""
        neck = self.create_neck()

        out_channels = neck.out_channels
        out_strides = neck.out_strides

        assert isinstance(out_channels, list)
        assert isinstance(out_strides, list)
        assert len(out_channels) == len(out_strides)
