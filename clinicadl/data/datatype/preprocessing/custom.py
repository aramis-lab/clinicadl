from logging import getLogger

from pydantic import computed_field

from ..modalities import Custom as CustomModality
from .base import Preprocessing
from .enum import PreprocessingMethod
from .file_type import FileType

logger = getLogger("clinicadl.data.datatype.preprocessing.custom")


class CustomPreprocessing(Preprocessing, CustomModality):
    """
    Configuration for custom preprocessing with a user-defined suffix.
    """

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.CUSTOM.value

    def _get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for custom preprocessing.
        """
        return FileType(
            pattern=f"custom/*{self.custom_suffix}.nii*",
            description="Custom preprocessing",
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of custom images with suffix {self.custom_suffix} "
