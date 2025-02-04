from logging import getLogger

from pydantic import computed_field

from ..modalities import Flair
from .base import _LinearPreprocessing
from .enum import PreprocessingMethod

logger = getLogger("clinicadl.data.datatype.preprocessing.flair")


class FlairLinear(_LinearPreprocessing, Flair):
    """Config class for Clinica's 'flair-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.FLAIR_LINEAR.value

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} Flair images with 'flair-linear' pipeline"
