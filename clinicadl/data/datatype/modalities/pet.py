from enum import Enum

from pydantic import computed_field

from clinicadl.data.datatype.utils import ImageModality

from .base import Modality


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


# Create a PET configuration with default values
default_pet = PET()
print(default_pet.modality)  # Output: ImageModality.PET
print(default_pet.tracer)  # Output: Tracer.FFDG
print(
    default_pet.suvr_reference_region
)  # Output: SUVRReferenceRegions.CEREBELLUM_PONS2

# Create a PET configuration with a different tracer
custom_pet = PET(tracer=Tracer.FAV45, suvr_reference_region=SUVRReferenceRegions.PONS)
print(custom_pet.tracer)  # Output: Tracer.FAV45
print(custom_pet.suvr_reference_region)  # Output: SUVRReferenceRegions.PONS
