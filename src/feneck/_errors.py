"""Custom exceptions for the feneck package."""


class StrideOrderError(ValueError):
    """Input strides must be in increasing order."""


class InputLevelError(ValueError):
    """Invalid number of input levels for the architecture."""


class ChannelMismatchError(ValueError):
    """Channel configuration mismatch in the architecture."""


class UnsupportedOperationError(ValueError):
    """Unsupported operation or configuration."""


class FeatureCountError(ValueError):
    """Incorrect number of input features provided."""


class ConfigurationError(ValueError):
    """Invalid architecture configuration."""


class DimensionError(ValueError):
    """Invalid tensor dimensions."""
