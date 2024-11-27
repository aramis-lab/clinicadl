from typing import Optional

from pydantic import field_validator

from clinicadl.utils.config import ClinicaDLConfig


class FileType(ClinicaDLConfig):
    pattern: str
    description: str
    needed_pipeline: Optional[str] = None

    @field_validator("pattern", mode="before")
    def check_pattern(cls, v):
        if v[0] == "/":
            raise ValueError(
                "pattern argument cannot start with char: / (does not work in os.path.join function). "
                "If you want to indicate the exact name of the file, use the format "
                "directory_name/filename.extension or filename.extension in the pattern argument."
            )
        return v
