from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError


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


class DataLeakageError(ClinicaDLException):
    """Base class for data leakage exceptions."""


class ClinicaDLTSVError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class DataFrameError(ClinicaDLException):
    """Base class for exceptions on the DataFrames."""


class ClinicaDLBIDSError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class ClinicaDLCAPSError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class TensorConversionError(ClinicaDLException):
    """Base class for tsv files exceptions."""


class ClinicaDLTrainingException(ClinicaDLException):
    """Base class for training exceptions."""


class MetricsHandlerError(ClinicaDLException):
    """Base class for training exceptions."""


class ClinicaDLMAPSError(ClinicaDLException):
    """Base class for training exceptions."""


class ClinicaDLTestingError(ClinicaDLException):
    """Base class for testing exceptions."""


class NotInterpretableJsonError(ClinicaDLException):
    """When a json cannot be interpreted by an object in ClinicaDL."""

    def __init__(self, json_path: Path, object_name: str):
        error_msg = f"{object_name} cannot read {str(json_path)}"
        super().__init__(error_msg)


class MissingFieldsError(ClinicaDLException):
    """When some expected fields are missing."""

    def __init__(self, fields: Sequence[str]):
        self.fields = fields
        error_msg = f"Fields {fields} are missing."
        super().__init__(error_msg)


class WrongFieldsError(ClinicaDLException):
    """When some unknown fields are passed."""

    def __init__(self, fields: Sequence[str], object_name: str):
        self.fields = fields
        self.object_name = object_name
        error_msg = f"Fields {fields} are not expected by {object_name}."
        super().__init__(error_msg)


class MissingFieldsJsonError(ClinicaDLException):
    """When fields of a json file are missing."""

    def __init__(self, error: MissingFieldsError, json_path: Path):
        error_msg = f"Fields {error.fields} are missing in {str(json_path)}"
        super().__init__(error_msg)


class WrongFieldsJsonError(ClinicaDLException):
    """When some unknown fields are in a json."""

    def __init__(self, error: WrongFieldsError, json_path: Path):
        error_msg = f"Fields {error.fields} in {str(json_path)} are not expected by {error.object_name}."
        super().__init__(error_msg)


class CannotReadFieldError(ClinicaDLException):
    """When a field of a dict cannot be read."""

    def __init__(
        self,
        object_name: str,
        field_names: Optional[list[str]] = None,
        error: Optional[ValidationError] = None,
    ):
        if field_names and not error:
            self.field_names = field_names
        elif error and not field_names:
            self.field_names = sorted(
                list(set([err["loc"][0] for err in error.errors()]))
            )
        else:
            raise ValueError("Pass either the field names OR the pydantic error.")

        self.object_name = object_name
        self.error = error
        error_msg = (
            f"{object_name} cannot read the field(s) {self.field_names}. "
            f"Please pass this field via kwargs."
        )
        super().__init__(error_msg)


class CannotReadJsonFieldError(ClinicaDLException):
    """When a field of a json cannot be read."""

    def __init__(
        self, error: CannotReadFieldError, json_path: Path, mention_kwargs: bool = True
    ):
        self.error = error
        error_msg = f"{error.object_name} cannot read the field(s) {error.field_names} in {str(json_path)}"
        if mention_kwargs:
            error_msg += "\nPlease pass this field via kwargs."
        super().__init__(error_msg)
