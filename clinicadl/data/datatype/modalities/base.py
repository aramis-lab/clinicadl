import abc
from enum import Enum

from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig


class ImageModality(str, Enum):
    """Possible modality for images in clinicaDL."""

    T1W = "T1w"
    DWI = "dwi"
    PET = "pet"
    FLAIR = "flair"
    CUSTOM = "custom"


class Modality(ClinicaDLConfig, abc.ABC):
    """
    Abstract base class for the preprocessing procedure.

    This class defines the common structure and methods that all preprocessing
    procedures should follow.
    """

    @computed_field
    @property
    @abc.abstractmethod
    def modality(self) -> ImageModality:
        """
        The modality of the raw data (e.g., T1, FLAIR, DWI, PET).

        This property must be implemented by subclasses to return the specific
        image modality being handled.
        """
        raise NotImplementedError("Subclasses must define the `modality` property.")
