from enum import Enum
from typing import Optional

from pydantic import computed_field

from ..enum import ImageModality
from .base import Modality


class Tracer(str, Enum):
    """BIDS label for PET tracers.

    Follows the convention proposed in the PET section of the BIDS specification.

    See: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html
    """

    PIB = "11CPIB"
    AV1451 = "18FAV1451"
    AV45 = "18FAV45"
    FBB = "18FFBB"
    FDG = "18FFDG"
    FMM = "18FFMM"


class ReconstructionMethod(str, Enum):
    """BIDS label for PET reconstruction methods.

    Follows the convention proposed in the PET section of the BIDS specification.

    See: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html#pet-recording-data

    For ADNI specific reconstruction methods, see:

    https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/pet/
    """

    # Reconstruction methods defined in the BIDS specifications
    STATIC = "nacstat"
    DYNAMIC = "nacdyn"
    STATIC_ATTENUATION_CORRECTION = "acstat"
    DYNAMIC_ATTENUATION_CORRECTION = "acdyn"

    # ADNI specific reconstruction methods
    CO_REGISTERED_DYNAMIC = "coregdyn"  # Corresponds to ADNI processing steps 1
    CO_REGISTERED_AVERAGED = "coregavg"  # Corresponds to ADNI processing steps 2
    CO_REGISTERED_STANDARDIZED = "coregstd"  # Corresponds to ADNI processing steps 3
    COREGISTERED_ISOTROPIC = "coregiso"  # Corresponds to ADNI processing steps 4


class PET(Modality):
    """
    Configuration to handle PET (Positron Emission Tomography) images.

    Parameters
    ----------
    tracer : Tracer
        radioactive tracer used for acquisition.
    reconstruction : Optional[ReconstructionMethod], default=None
        the method used to reconstruct the image.
    """

    tracer: Tracer = Tracer.FDG
    reconstruction: Optional[ReconstructionMethod] = None

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'pet' here.
        """
        return ImageModality.PET.value
