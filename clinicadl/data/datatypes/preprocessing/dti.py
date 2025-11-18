import os
import re
from enum import Enum
from typing import Pattern

from .base import Preprocessing


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


class DWIDTI(Preprocessing):
    """
    :py:class:`DataType <clinicadl.data.datatypes.DataType>` to handle Diffusion-Weighted MRI (DWI) images
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

    @property
    def _pipeline_name(self) -> str:
        return "dwi-dti"

    @property
    def _filename(self) -> str:
        return f"{self._pipeline_name}_{self.measure}_{self.space}"

    def _get_pattern(self) -> Pattern:
        if self.space == DTISpace.NORMALIZED:
            folder = "normalized_space"
            space = "MNI152Lin"
        else:
            folder = "native_space"
            space = ".*"

        file_pattern = f"sub-.*_ses-.*_space-{space}_{self.measure}.nii.*"
        pattern = os.path.join("dwi", "dti_based_processing", folder, file_pattern)

        return re.compile(pattern)

    def _get_description(self) -> str:
        return f"DTI {self.measure} images in {self.space} space, preprocessed with Clinica's '{self._pipeline_name}' pipeline"
