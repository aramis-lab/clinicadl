from logging import getLogger
from typing import Optional

from clinicadl.data.preprocessing.base import BasePreprocessing
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
)
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.dti")


class PreprocessingDTI(BasePreprocessing):
    """
    Configuration for DTI-based preprocessing
    """

    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    preprocessing: Preprocessing = Preprocessing.DWI_DTI

    def get_bids_filerype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="dwi/sub-*_ses-*_dwi.nii*", description="DWI NIfTI")

    def get_caps_filetype(self) -> FileType:
        """Return the query dict required to capture DWI DTI images.

        Parameters
        ----------
        config: PreprocessingDTI

        Returns
        -------
        FileType :
        """
        measure = self.dti_measure
        space = self.dti_space

        return FileType(
            pattern=f"dwi/dti_based_processing/*/*_space-{space}_{measure.value}.nii.gz",
            description=f"DTI-based {measure.value} in space {space}.",
            needed_pipeline="dwi_dti",
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} DTI images with measure {self.dti_measure.value} and space {self.dti_space.value}. "
