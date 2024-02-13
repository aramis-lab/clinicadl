class DownloadError(Exception):
    """Base class for download errors exceptions."""


class ClinicaDLArgumentError(ValueError):
    """Base class for ClinicaDL CLI Arguments error."""


class ClinicaDLConfigurationError(ValueError):
    """Base class for ClinicaDL configurations error."""


class ClinicaDLException(Exception):
    """Base class for ClinicaDL exceptions."""


class MAPSError(ClinicaDLException):
    """Base class for MAPS exceptions."""


class ClinicaDLNetworksError(ClinicaDLException):
    """Base class for Networks exceptions."""


class ClinicaDLDataLeakageError(ClinicaDLException):
    """Base class for data leakage exceptions."""


class ClinicaDLTSVError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class ClinicaDLBIDSError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class ClinicaDLCAPSError(ClinicaDLException):
    """Base class for tsv files exceptions."""
