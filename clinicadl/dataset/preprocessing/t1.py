from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.dataset.preprocessing.base import Preprocessing
from clinicadl.utils.enum import LinearModality, PreprocessingMethod
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.t1")


class PreprocessingT1(Preprocessing):
    """Config class for Clinica's 't1-linear' preprocessing."""

    use_uncropped_image: bool = True

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.T1_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="anat/sub-*_ses-*_T1w.nii*", description="T1w MRI")

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.T1W, needed_pipeline=PreprocessingMethod.T1_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T1 images with t1-linear pipeline"
