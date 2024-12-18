from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.utils.enum import LinearModality, PreprocessingMethod
from clinicadl.utils.iotools.clinica_utils import FileType

from .base import _PreprocessingWithCrop

logger = getLogger("clinicadl.preprocessing.flair")


class PreprocessingFlair(_PreprocessingWithCrop):
    """Config class for Clinica's 'flair-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.FLAIR_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.FLAIR,
            needed_pipeline=PreprocessingMethod.FLAIR_LINEAR,
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} Flair images with flair-linear pipeline"
