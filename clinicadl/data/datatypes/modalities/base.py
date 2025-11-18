import abc

from clinicadl.utils.config import ClinicaDLConfig


class Modality(ClinicaDLConfig, abc.ABC):
    """
    Abstract class for image modalities.
    """

    @property
    @abc.abstractmethod
    def _modality(self) -> str:
        """
        The modality of the raw data (e.g., T1, FLAIR, DWI, PET).

        This property must be implemented by subclasses to return the specific
        image modality being handled.
        """
