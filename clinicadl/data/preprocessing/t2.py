from logging import getLogger
from typing import Optional

from clinicadl.data.preprocessing.base import BasePreprocessing
from clinicadl.utils.enum import LinearModality, Preprocessing
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.t2")


class PreprocessingT2(BasePreprocessing):
    preprocessing: Preprocessing = Preprocessing.T2_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        raise NotImplementedError(
            f"Extraction of preprocessing {self.preprocessing.value} is not implemented from BIDS directory."
        )

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.T2W, needed_pipeline=Preprocessing.T2_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T2 images with t2-linear pipeline"
