from pydantic import computed_field

from .base import ImageModality, Modality


class Custom(Modality):
    """
    Configuration for custom preprocessing with a user-defined suffix.

    Attributes:
        custom_suffix (str): User-defined suffix for custom preprocessing patterns.
    """

    custom_suffix: str = ""

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        Specifies the modality for custom preprocessing.

        Returns:
            ImageModality: The modality, always set to ImageModality.CUSTOM.
        """
        return ImageModality.CUSTOM
