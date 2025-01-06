import abc

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.utils.config import ClinicaDLConfig


class RawData(ClinicaDLConfig, abc.ABC):
    """
    Abstract base class for the use of Raw Data (Unprocessed data in BIDS format).
    """

    @computed_field
    @property
    def file_type(self) -> FileType:
        """
        The file type associated with the BIDS dataset.

        This property uses the `get_bids_filetype` method to return the correct
        file type for the modality being handled.
        """
        return self._get_bids_filetype()

    @abc.abstractmethod
    def _get_bids_filetype(self) -> FileType:
        """
        Abstract method to obtain the BIDS-compatible FileType.

        This method must be implemented by subclasses to specify the file type
        associated with the modality.
        """
        raise NotImplementedError(
            "Subclasses must implement the `get_bids_filetype` method."
        )
