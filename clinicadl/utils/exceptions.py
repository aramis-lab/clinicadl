from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError


def add_note(exc: Exception, note: str) -> Exception:
    """
    Adds a note to an exception.

    From Python 3.11, there is exception the method Exception.add_note.
    So, this method is only relevant for Python 3.10.
    """
    if not hasattr(exc, "__notes__"):
        exc.__notes__ = []
    exc.__notes__.append(note)
    return exc


class ClinicaDLException(Exception):
    """Base class for ClinicaDL exceptions."""


class DataLeakageError(ClinicaDLException):
    """Base class for data leakage exceptions."""


class DataFrameError(ClinicaDLException):
    """Base class for exceptions on the DataFrames."""


class TensorConversionError(ClinicaDLException):
    """Base class for tsv files exceptions."""


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
        error: Exception,
        field_names: Optional[list[str]] = None,
    ):
        if field_names:
            self.field_names = field_names
        else:
            if not isinstance(error, ValidationError):
                raise ValueError(
                    "If 'field_names' is not passed, 'error' must be a pydantic.ValidationError."
                )
            self.field_names = sorted(
                list(set([err["loc"][0] for err in error.errors()]))
            )

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


class CannotReadJsonError(ClinicaDLException):
    """When a json cannot be read."""
