from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import computed_field, field_validator

from clinicadl.dictionary.suffixes import NII
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.typing import PathType

from .enum import PreprocessingMethod


class FileType(ClinicaDLConfig):
    """
    Represents a file type with a pattern, description, and an optional pipeline requirement.
    """

    container: str
    pattern: Optional[str] = None
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

    @computed_field
    @property
    @abstractmethod
    def modality(self) -> str:
        """
        The modality of the raw data (e.g., T1, FLAIR, DWI, PET).

        This property must be implemented by subclasses to return the specific
        image modality being handled.
        """

    def get_nii_pattern(self):
        """
        Returns the pattern for NIfTI files.
        This method constructs a pattern string for NIfTI files based on the
        container, pattern, and modality attributes.
        """
        return f"{self.container}/sub-*_ses-*{self.pattern}_{self.modality}.nii*"

    def __str__(self):
        """
        String description of the data.
        """
        return self.description

    @staticmethod
    def get_container_and_filename_from_path(path: PathType) -> Tuple[str, str]:
        """
        Returns the filename from a path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist.")
        if not Path(path).is_file():
            raise IsADirectoryError(f"The path {path} is a directory, not a file.")

        filename = path.name
        container = path.parent.name
        if filename.endswith(NII):
            filename = filename[:-4]
        return container, filename
