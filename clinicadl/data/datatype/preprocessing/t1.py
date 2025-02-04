from logging import getLogger

from pydantic import computed_field

from ..modalities import T1w
from .base import _LinearPreprocessing
from .enum import PreprocessingMethod

logger = getLogger("clinicadl.data.datatype.preprocessing.t1")


class T1Linear(_LinearPreprocessing, T1w):
    """Config class for Clinica's 't1-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.T1_LINEAR.value

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T1 images with 't1-linear' pipeline"
