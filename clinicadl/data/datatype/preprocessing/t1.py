from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import T1w

from .base import PreprocessingMethod, _PreprocessingWithCrop

logger = getLogger("clinicadl.preprocessing.t1")


class T1Linear(_PreprocessingWithCrop, T1w):
    """Config class for Clinica's 't1-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.T1_LINEAR

    def get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for t1-linear preprocessing.
        """
        return self.linear_nii(
            modality=self.modality, needed_pipeline=PreprocessingMethod.T1_LINEAR
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T1 images with t1-linear pipeline"
