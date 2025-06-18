from pydantic import computed_field

from ..enum import ImageModality
from .base import Modality


class Flair(Modality):
    """
    Configuration to handle FLAIR (Fluid-Attenuated Inversion Recovery) images.
    """

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'FLAIR' here.
        """
        return ImageModality.FLAIR.value
