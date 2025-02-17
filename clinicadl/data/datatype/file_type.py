from __future__ import annotations

from typing import Optional, Union

from pydantic import field_validator

from clinicadl.utils.config import ClinicaDLConfig

from .enum import PreprocessingMethod


class FileType(ClinicaDLConfig):
    """
    Represents a file type with a pattern, description, and an optional pipeline requirement.
    """

    pattern: str
    description: str
    needed_pipeline: Optional[Union[PreprocessingMethod, str]] = None

    @field_validator("pattern", mode="after")
    @classmethod
    def check_pattern(cls, v):
        if v[0] == "/":
            raise ValueError(
                "pattern argument cannot start with char: / (does not work in os.path.join function). "
                "If you want to indicate the exact name of the file, use the format "
                "directory_name/filename.extension or filename.extension in the pattern argument."
            )
        return v
