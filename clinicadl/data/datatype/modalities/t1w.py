from pydantic import computed_field

from .base import ImageModality, Modality


class T1w(Modality):
    """
    Configuration for T1-weighted (T1w) modality preprocessing.

    This class defines the specific settings and attributes for handling
    T1-weighted images in a preprocessing pipeline.
    """

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        The image modality for this configuration.

        Returns:
            ImageModality: Always set to ImageModality.T1W.
        """
        return ImageModality.T1W
