from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import Flair

from .base import PreprocessingMethod, _PreprocessingWithCrop

logger = getLogger("clinicadl.preprocessing.flair")


class FlairLinear(_PreprocessingWithCrop, Flair):
    """Config class for Clinica's 'flair-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.FLAIR_LINEAR

    def get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for flair-linear preprocessing.
        """
        return self.linear_nii(
            modality=self.modality,
            needed_pipeline=PreprocessingMethod.FLAIR_LINEAR,
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} Flair images with flair-linear pipeline"
