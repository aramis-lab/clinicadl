from logging import getLogger
from typing import Optional

from clinicadl.data.preprocessing.base import BasePreprocessing
from clinicadl.utils.enum import LinearModality, Preprocessing
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.flair")


class PreprocessingFlair(BasePreprocessing):
    preprocessing: Preprocessing = Preprocessing.FLAIR_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.FLAIR, needed_pipeline=Preprocessing.FLAIR_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} Flair images with flair-linear pipeline"
