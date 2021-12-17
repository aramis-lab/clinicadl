class DownloadError(Exception):
    """Base class for download errors exceptions."""

    pass


class ClinicaDLArgumentError(ValueError):
    pass


class ConfigurationError(ValueError):
    pass


class ClinicaDLException(Exception):
    """Base class for Clinica exceptions."""


class MAPSError(ClinicaDLException):
    """Base class for MAPS errors."""


class ClinicaDLNetworksError(ClinicaDLException):
    """Base class for Networks errors."""


class ClinicaDLDataLeakageError(ClinicaDLException):
    """Base class for data leakage errors."""


class ClinicaDLTSVError(ClinicaDLException):
    """Base class for tsv files errors."""
