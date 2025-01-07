from pydantic import computed_field

from clinicadl.data.datatype.utils import ImageModality

from .base import Modality


class DWI(Modality):
    """
    Configuration for DWI (Diffusion Weighted Imaging) modality preprocessing.

    This class defines the specific settings and attributes for handling
    DWI images in a preprocessing pipeline.
    """

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        The image modality for this configuration.

        Returns:
            ImageModality: Always set to ImageModality.DWI.
        """
        return ImageModality.DWI
