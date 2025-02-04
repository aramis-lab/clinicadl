from enum import Enum

from pydantic import computed_field

from .base import ImageModality, Modality


class Tracer(str, Enum):
    """Possible tracer for pet images in clinicaDL."""

    FFDG = "18FFDG"
    FAV45 = "18FAV45"
    CPIB = "11CPIB"


class SUVRReferenceRegions(str, Enum):
    """Possible SUVR reference region for pet images in clinicaDL."""

    PONS = "pons"
    CEREBELLUMPONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUMPONS2 = "cerebellumPons2"


class PET(Modality):
    """
    Configuration for custom preprocessing with a user-defined suffix.

    Attributes:
        custom_suffix (str): User-defined suffix for custom preprocessing patterns.
    """

    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        Specifies the modality for custom preprocessing.

        Returns:
            ImageModality: The modality, always set to ImageModality.CUSTOM.
        """
        return ImageModality.PET
