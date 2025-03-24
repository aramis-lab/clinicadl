from pydantic import computed_field

from ..enum import ImageModality
from .base import Modality


class Custom(Modality):
    """
    Configuration to handle custom images with a user-defined suffix.

    Parameters
    ----------
    custom_suffix : str
        User-defined suffix for custom modality.
    """

    custom_suffix: str

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'custom' here.
        """
        return ImageModality.CUSTOM.value
