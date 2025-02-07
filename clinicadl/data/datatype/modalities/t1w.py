from pydantic import computed_field

from ..enum import ImageModality
from .base import Modality


class T1w(Modality):
    """
    Configuration to handle T1-weighted (T1w) images.
    """

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'T1w' here.
        """
        return ImageModality.T1W.value
