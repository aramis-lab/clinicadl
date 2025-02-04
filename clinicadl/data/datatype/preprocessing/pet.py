from logging import getLogger

from pydantic import computed_field

from ..modalities import PET
from .base import _LinearPreprocessing
from .enum import PreprocessingMethod

logger = getLogger("clinicadl.data.datatype.preprocessing.pet")


class PETLinear(_LinearPreprocessing, PET):
    """Config class for Clinica's 'pet-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.PET_LINEAR.value

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} PET images with tracer {self.tracer} and suvr reference region {self.suvr_reference_region}. "

    def _get_filename(self):
        """
        Constructs the file name depending on the parameters of 'pet-linear'.
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        return f"*_trc-{self.tracer}_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_suvr-{self.suvr_reference_region}_{self.modality.value}.nii*"
