from .base import Modality
from .enum import ImageModality


class T1w(Modality):
    """
    To handle T1-weighted (T1w) images.
    """

    @property
    def _modality(self) -> str:
        """
        The modality, always 'T1w' here.
        """
        return ImageModality.T1W.value
