from enum import Enum
from typing import Optional, Union

from pydantic import field_validator

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import Preprocessing


class FileType(ClinicaDLConfig):
    """
    Represents a file type with a pattern, description, and optional pipeline requirement.
    """

    pattern: str
    description: str
    needed_pipeline: Optional[Preprocessing] = None

    @field_validator("pattern", mode="before")
    def check_pattern(cls, v):
        if not v:
            raise ValueError("A pattern must be specified")

        elif v[0] == "/":
            raise ValueError(
                "pattern argument cannot start with char: / (does not work in os.path.join function). "
                "If you want to indicate the exact name of the file, use the format "
                "directory_name/filename.extension or filename.extension in the pattern argument."
            )
        return v

    @field_validator("description", mode="before")
    def check_description(cls, v):
        if not v:
            raise ValueError("A description must be specified")
        return v

    @field_validator("needed_pipeline", mode="after")
    def check_needed_pipeline(cls, v: Optional[Union[str, Preprocessing]]):
        if v:
            try:
                v = Preprocessing(v)
            except ValueError:
                raise ValueError(
                    f"Invalid pipeline: {v}. Choose from {[e.value for e in Preprocessing]}"
                )
            return v
