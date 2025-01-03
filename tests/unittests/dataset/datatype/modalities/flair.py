from pydantic import computed_field

from .base import ImageModality, Modality


class Flair(Modality):
    """
    Configuration for FLAIR (Fluid-Attenuated Inversion Recovery) modality preprocessing.

    This class defines the specific settings and attributes for handling
    FLAIR images in a preprocessing pipeline.
    """

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        The image modality for this configuration.

        Returns:
            ImageModality: Always set to ImageModality.FLAIR.
        """
        return ImageModality.FLAIR
