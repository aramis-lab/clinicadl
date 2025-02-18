from pydantic import computed_field

from ..enum import ImageModality
from .base import Modality


class DWI(Modality):
    """
    Configuration to handle DWI (Diffusion Weighted Imaging) images.
    """

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'dwi' here.
        """
        return ImageModality.DWI.value
