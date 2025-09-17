import pydantic

from clinicadl.utils.typing import PathType


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


class ClinicaDLTensorConversionError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class ClinicaDLTrainingException(ClinicaDLException):
    """Base class for training exceptions."""


class MetricsHandlerError(ClinicaDLException):
    """Base class for training exceptions."""


class ClinicaDLMAPSError(ClinicaDLException):
    """Base class for training exceptions."""


class ClinicaDLTestingError(ClinicaDLException):
    """Base class for testing exceptions."""


class NotInterpretableJson(ClinicaDLException):
    """When a json cannot be interpreted by an object in ClinicaDL."""

    def __init__(self, json_path: PathType, object_name: str):
        error_msg = f"{object_name} cannot read {(str(json_path))}"
        super().__init__(error_msg)


class NotInterpretableJsonField(ClinicaDLException):
    """When some fields in a json cannot be interpreted by an object in ClinicaDL."""

    def __init__(
        self, error: pydantic.ValidationError, json_path: PathType, object_name: str
    ):
        wrong_fields = set([f"'{err['loc'][0]}'" for err in error.errors()])
        wrong_fields_str = ", ".join(wrong_fields)

        error_msg = (
            f"{object_name} cannot read the following fields in {str(json_path)}: {wrong_fields_str}\n"
            f"Please pass these fields via kwargs."
        )
        super().__init__(error_msg)


class NotInterpretableDictField(ClinicaDLException):
    """When some values in a dict cannot be interpreted by an object in ClinicaDL."""

    def __init__(self, error: pydantic.ValidationError, object_name: str):
        self.error = error
        wrong_fields = set([f"'{err['loc'][0]}'" for err in error.errors()])
        wrong_fields_str = ", ".join(wrong_fields)

        error_msg = (
            f"{object_name} cannot read the following fields in the dictionary: {wrong_fields_str}\n"
            f"Please pass these fields via kwargs."
        )
        super().__init__(error_msg)
