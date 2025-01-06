from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import PET

from .base import PreprocessingMethod, _LinearPreprocessing


class PETLinear(_LinearPreprocessing, PET):
    """Config class for Clinica's 'pet-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.PET_LINEAR

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} PET images with tracer {self.tracer} and suvr reference region {self.suvr_reference_region}. "
