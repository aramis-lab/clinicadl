from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.dataset.preprocessing.base import Preprocessing
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    PreprocessingMethod,
)
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.dti")


class PreprocessingDTI(Preprocessing):
    """Config class for Clinica's 't1-linear' preprocessing."""

    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    use_uncropped_image: bool = True

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.DWI_DTI

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
