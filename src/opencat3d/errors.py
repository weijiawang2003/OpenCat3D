"""Project-specific exceptions."""


class OpenCat3DError(RuntimeError):
    """Base class for recoverable OpenCat3D errors."""


class StageCacheMiss(OpenCat3DError):
    """Raised when a stage cache entry does not match current inputs/config."""


class LicenseGateError(OpenCat3DError):
    """Raised when a registry entry is not allowed for the requested profile."""


class AdapterUnavailableError(OpenCat3DError):
    """Raised when an optional local adapter is registered but not installed."""


class NotImplementedStageError(OpenCat3DError):
    """Raised by CLI commands reserved for later implementation tasks."""
