from enum import Enum
from logging import getLogger

from pydantic import computed_field

from ..enum import PreprocessingMethod
from ..file_type import FileType
from ..modalities import DWI
from .base import Preprocessing

logger = getLogger("clinicadl.data.datatypes.preprocessing.dti")


class DTIMeasure(str, Enum):
    """Possible DTI measures."""

    FRACTIONAL_ANISOTROPY = "FA"
    MEAN_DIFFUSIVITY = "MD"
    AXIAL_DIFFUSIVITY = "AD"
    RADIAL_DIFFUSIVITY = "RD"


class DTISpace(str, Enum):
    """Possible DTI spaces."""

    NATIVE = "native"
    NORMALIZED = "normalized"


class DWIDTI(Preprocessing, DWI):
    """
    Configuration class to handle Diffusion-Weighted MRI (DWI) images,
    preprocessed with `Clinica dwi-dti <https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/DWI_DTI/>`_
    pipeline.

    Parameters
    ----------
    measure : DTIMeasure
        The DTI-based measure to use, among ``FA`` (fractional anisotropy),
        ``MD`` (mean diffusivity), ``AD`` (axial diffusivity) and ``RD`` (radial diffusivity).
    space : DTISpace
        Either ``native`` (the data in the native space) or ``normalized`` (the data in
        MNI152Lin standard space):\n
        - with ``native``: only the files that match the pattern
          ``dwi/dti_based_processing/native_space/sub-*_ses-*_space-*_{measure}.nii*``
          in the :term:`CAPS` structure will be considered.
        - with ``normalized``: only the files that match the pattern
          ``dwi/dti_based_processing/normalized_space/sub-*_ses-*_space-MNI152Lin_{measure}.nii*``
          in the :term:`CAPS` structure will be considered.
    """

    measure: DTIMeasure
    space: DTISpace

    @computed_field
    @property
    def name(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.DWI_DTI.value

    def _get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for DWI_DTI preprocessing.
        """
        if self.space == DTISpace.NORMALIZED:
            folder = "normalized_space"
            space = "MNI152Lin"
        else:
            folder = "native_space"
            space = "*"

        return FileType(
            pattern=f"dwi/dti_based_processing/{folder}/sub-*_ses-*_space-{space}_{self.measure}.nii*",
            description=f"DTI {self.measure} images in {self.space} space, preprocessed with Clinica's 'dwi-dti' pipeline",
            needed_pipeline=self.name,
        )

    def _get_file_name(self) -> str:
        """
        Builds a suffix for files saving
        information on this preprocessing.
        """
        return f"dwi-dti_{self.measure}_{self.space}"
