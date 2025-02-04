from enum import Enum
from logging import getLogger

from pydantic import computed_field

from ..modalities import DWI
from .base import Preprocessing
from .enum import PreprocessingMethod
from .file_type import FileType

logger = getLogger("clinicadl.data.datatype.preprocessing.dti")


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
    ALL = "*"


class DWIDTI(Preprocessing, DWI):
    """Config class for Clinica's 'dwi-dti' preprocessing."""

    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.DWI_DTI.value

    def _get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for DWI_DTI preprocessing.
        """
        measure = self.dti_measure
        space = self.dti_space

        return FileType(
            pattern=f"dwi/dti_based_processing/*/*_space-{space}_{measure}.nii*",
            description=f"DTI-based {measure} in space {space}.",
            needed_pipeline=self.preprocessing,
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of DTI images with measure {self.dti_measure.value} and space {self.dti_space.value}. "
