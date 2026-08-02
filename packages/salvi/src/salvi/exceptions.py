"""Typed exceptions exposed by SALVI."""


class SalviError(Exception):
    """Base class for expected SALVI failures."""


class ConfigurationError(SalviError):
    """Raised when a run configuration is invalid."""


class ComponentError(SalviError):
    """Raised when component registration or composition fails."""


class ArtifactError(SalviError):
    """Raised when a canonical artifact violates its contract."""


class ConversionError(ArtifactError):
    """Raised when external data cannot be converted canonically."""


class RunError(SalviError):
    """Raised when a run cannot be completed."""


class RunCancelledError(RunError):
    """Raised when a run is cancelled cooperatively."""


class EvaluationWorkerError(RunError):
    """Raised when a parallel evaluation worker cannot complete a candidate."""


class OptionalDependencyError(SalviError):
    """Raised when a requested optional feature is not installed."""
