from .base import Modality
from .enum import ImageModality


class Flair(Modality):
    """
    To handle FLAIR (Fluid-Attenuated Inversion Recovery) images.
    """

    @property
    def _modality(self) -> str:
        """
        The modality, always 'FLAIR' here.
        """
        return ImageModality.FLAIR.value
