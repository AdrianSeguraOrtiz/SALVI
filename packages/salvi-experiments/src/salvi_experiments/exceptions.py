"""Typed failures raised by scientific experiment protocols."""


class ExperimentError(RuntimeError):
    """Base error for invalid inputs, artifacts, or experiment execution."""


class ExperimentConfigurationError(ExperimentError):
    """Raised when an experiment YAML document is invalid."""


class ExperimentArtifactError(ExperimentError):
    """Raised when an experiment input or output artifact is invalid."""


__all__ = [
    "ExperimentArtifactError",
    "ExperimentConfigurationError",
    "ExperimentError",
]
