from enum import Enum

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import DWI
from clinicadl.data.datatype.utils import PreprocessingMethod

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
    ALL = "*"


class DWIDTI(Preprocessing, DWI):
    """Config class for Clinica's 'dwi-dti' preprocessing."""

    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.DWI_DTI

    def _get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for DWI_DTI preprocessing.
        """

        measure = self.dti_measure
        space = self.dti_space

        return FileType(
            pattern=f"dwi/dti_based_processing/*/*_space-{space}_{measure}.nii.gz",
            description=f"DTI-based {measure} in space {space}.",
            needed_pipeline=PreprocessingMethod.DWI_DTI,
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of DTI images with measure {self.dti_measure.value} and space {self.dti_space.value}. "
