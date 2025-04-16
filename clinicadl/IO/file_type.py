from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import computed_field, field_validator

from clinicadl.dictionary.suffixes import NII
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.typing import PathType

from .enum import PreprocessingMethod


class FileType(ABC):
    """
    Represents a file type with a pattern, description, and an optional pipeline requirement.
    """

    def __init__(
        self,
        pattern: str = "",
        needed_pipeline: Optional[Union[PreprocessingMethod, str]] = None,
    ):
        """
        Initializes the FileType instance with the given attributes.
        """
        self.pattern = pattern if pattern == "" else self.check_pattern(pattern)
        self.needed_pipeline = needed_pipeline

    def check_pattern(self, s: str) -> str:
        """
        Validate the pattern string.
        """
        # Check if the pattern matches the expected format
        pattern = r"(_[^_-]+-[^_-]+)+$"
        if bool(re.fullmatch(pattern, s)):
            return s
        raise ValueError(
            f"Pattern '{s}' is not valid. It should match the pattern: {pattern}"
        )

    @classmethod
    def from_filename(cls, filename: str):
        """
        Generate the BIDS-compatible file type pattern and description from a filename.

        Args:
            filename (str): The filename to extract information from.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        # Extract relevant information from the filename
        # This is a placeholder implementation; actual extraction logic will depend on filename format
        match = re.search(
            rf"sub-[^_]+_ses-[^_]+(?P<middle>.+)_{cls.modality}", filename
        )
        pattern = match.group("middle") if match else ""
        return cls(pattern=pattern)

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality of the raw data (e.g., T1, FLAIR, DWI, PET).

        This property must be implemented by subclasses to return the specific
        image modality being handled.
        """
        raise NotImplementedError(
            "The modality property must be implemented by subclasses."
        )

    @computed_field
    @property
    @abstractmethod
    def description(self) -> str:
        """
        The description of the file type.
        """

    @computed_field
    @property
    @abstractmethod
    def container(self) -> str:
        """
        The name of the folder where the file is stored.
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
